"""LLM-SA: a heterogeneous stacking ensemble of survival learners.

The model combines three deliberately complementary base learners, each fitted
on the same training fold:

  1. CoxPH           -- linear proportional hazards on the structured
                        covariates + LLM numeric fields N_i (one-hot encoded).
  2. RSF             -- random survival forest on the same feature block.
  3. DeepSurv + text -- a Cox-style MLP that additionally consumes the
                        PCA-reduced narrative embedding E_i.

Each base learner produces a predicted survival curve on the canonical time
grid. A simplex-constrained meta-learner then combines the three curves into a
single survival curve. The meta-weights w >= 0, sum(w) = 1 are found by
projected gradient descent that minimises squared error against the
validation-fold survival labels, with an L2 prior pulling the weights toward
the uniform vector [1/3, 1/3, 1/3] (strength ``meta_prior_lam``). The prior
is a light regulariser: lam = 0 recovers the vanilla simplex least-squares
stack.

All folds are disjoint: base learners are fitted on train, the meta-weights are
fitted on out-of-sample predictions, and everything is evaluated on test.

``TrainConfig.meta_fit`` selects where those out-of-sample predictions come
from. ``"val"`` uses the held-out validation fold (the original behaviour); on
a small cohort that fold may carry only a handful of events, leaving the three
weights unidentifiable. ``"oof"`` instead uses K-fold out-of-fold predictions
over the training fold -- the standard stacking construction -- which fits the
weights on ~n_train patients rather than ~n_val.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxPH as PCCoxPH
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from .data import DatasetBundle, SurvivalSplit
from .metrics import ipcw_cindex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onehot(split: SurvivalSplit, cardinalities: list) -> np.ndarray:
    pieces = [split.X_num.astype(np.float32)]
    for j, k in enumerate(cardinalities):
        oh = np.zeros((len(split), k), dtype=np.float32)
        oh[np.arange(len(split)), split.X_cat[:, j]] = 1.0
        pieces.append(oh)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _surv_at_grid(times_pred: np.ndarray, surv_pred: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Right-continuous step interpolation of survival curves onto ``grid``."""
    n, T = surv_pred.shape
    idx = np.searchsorted(times_pred, grid, side="right") - 1
    idx_clipped = np.clip(idx, 0, T - 1)
    out = surv_pred[:, idx_clipped]
    before = idx < 0
    if before.any():
        out[:, before] = 1.0
    return out


# ---------------------------------------------------------------------------
# Base learners (each implements .fit(bundle), .surv(bundle, split) -> (n, G))
# ---------------------------------------------------------------------------

class _BaseCoxPH:
    name = "coxph"

    def __init__(self):
        self.model = None
        self.times_ = None

    def fit(self, bundle):
        X = _onehot(bundle.train, bundle.cat_cardinalities)
        y = Surv.from_arrays(bundle.train.event.astype(bool), bundle.train.time)
        self.model = CoxPHSurvivalAnalysis(alpha=0.01, n_iter=200)
        self.model.fit(X, y)
        sf = self.model.predict_survival_function(X[:1])
        self.times_ = np.asarray(sf[0].x)

    def surv(self, bundle, split_name):
        split = getattr(bundle, split_name)
        X = _onehot(split, bundle.cat_cardinalities)
        sf = self.model.predict_survival_function(X)
        surv = np.stack([f(self.times_) for f in sf], axis=0)
        return _surv_at_grid(self.times_, surv, bundle.time_grid)


class _BaseRSF:
    name = "rsf"

    def __init__(self, seed=0, n_estimators=100, max_depth=6):
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = None
        self.times_ = None

    def fit(self, bundle):
        X = _onehot(bundle.train, bundle.cat_cardinalities)
        y = Surv.from_arrays(bundle.train.event.astype(bool), bundle.train.time)
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=15,
            n_jobs=1, random_state=self.seed,
        )
        self.model.fit(X, y)
        sf = self.model.predict_survival_function(X[:1])
        self.times_ = np.asarray(sf[0].x)

    def surv(self, bundle, split_name):
        split = getattr(bundle, split_name)
        X = _onehot(split, bundle.cat_cardinalities)
        sf = self.model.predict_survival_function(X)
        surv = np.stack([f(self.times_) for f in sf], axis=0)
        return _surv_at_grid(self.times_, surv, bundle.time_grid)


#: years -> native time unit of each cohort, for placing the LLM's 2-year and
#: 5-year anchors on the cohort's own time axis.
_TIME_UNIT_PER_YEAR = {
    "gbsg": 365.25, "support": 365.25, "flchain": 365.25, "rotterdam": 365.25,
    "tcga": 365.25, "tcgafull": 365.25, "heartfailure": 365.25,
    "whas500": 365.25, "brcamicro": 365.25, "aids": 365.25, "lung": 365.25,
    "veteran": 365.25,
    "metabric": 12.0, "wpbc": 12.0,
    "larynx": 1.0,
}


#: Anchors actually requested, in years, as (short-horizon risk, long-horizon
#: survival). Keyed by (feature_version, cohort) because a horizon-matched tag
#: such as ``v5hz`` means DIFFERENT horizons for different cohorts -- each pair
#: is chosen to sit inside that cohort's own follow-up. Anything absent uses the
#: standard 2-year / 5-year pair of the default target prompt.
_D = 365.25
_VERSION_ANCHOR_YEARS = {
    ("v5hz", "heartfailure"): (90.0 / _D, 180.0 / _D),   # follow-up 285 d
    ("v5hz", "aids"):         (90.0 / _D, 270.0 / _D),   # follow-up 364 d
}


def anchor_years_for(feature_version: str, dataset: str = ""):
    """(short, long) horizons in years for a cohort/version pair."""
    return _VERSION_ANCHOR_YEARS.get(
        (feature_version, (dataset or "").lower()), (2.0, 5.0))


class _BaseLLM:
    """A survival curve built from the LLM's two-horizon estimates.

    The LLM returns, per patient, S(5y) and F(2y) = 1 - S(2y) in [0, 1]. Two
    modes turn that pair into a curve on the canonical grid:

    ``cox1``   Fit a Cox model on the training fold using ONLY the LLM
               estimates as covariates and take its Breslow baseline. Ranking
               comes from the LLM; calibration and the time scale come from the
               data. Insensitive to the cohort's time unit, and usable even
               where the anchors lie outside follow-up.

    ``anchor`` Solve a per-patient Weibull through the two anchors in closed
               form, using no training data at all -- a genuinely zero-shot
               base learner. With S(t) = exp(-(t/lam)^k) and y = ln(-ln S):
                   k = (y2 - y1) / (ln t2 - ln t1),  ln lam = ln t1 - y1 / k
               Requires the anchors to sit inside follow-up, so it is not
               meaningful on cohorts whose horizon is under ~2 years (ACTG320,
               Heart Failure).
    """

    name = "llm"

    def __init__(self, mode="cox1", unit_per_year=365.25, seed=0,
                 anchor_years=(2.0, 5.0)):
        self.mode = mode
        self.unit_per_year = float(unit_per_year)
        # (short, long) horizons the prompt actually asked for, in YEARS. The
        # default 2/5 matches the standard target prompt; cohorts regenerated
        # with COHORT_HORIZONS need their own pair (heartfailure v5hz asks for
        # 90-day risk and 180-day survival -> (90/365.25, 180/365.25)).
        self.anchor_years = tuple(float(a) for a in anchor_years)
        self.seed = seed
        self.model = None
        self.times_ = None

    @staticmethod
    def _anchors(split):
        """(S at 2y, S at 5y) per patient, clipped and forced monotone."""
        z = np.asarray(split.llm_numerics, dtype=np.float64)
        eps = 1e-3
        s5 = np.clip(z[:, 0], eps, 1 - eps)              # estimated 5yr survival
        s2 = np.clip(1.0 - z[:, 1], eps, 1 - eps)        # 1 - estimated 2yr risk
        # the LLM can return an inconsistent pair (S(5y) >= S(2y)); enforce a
        # strictly decreasing curve rather than discarding the patient
        s5 = np.minimum(s5, s2 - eps)
        return s2, np.clip(s5, eps, 1 - eps)

    def _feats(self, split):
        s2, s5 = self._anchors(split)
        conf = np.asarray(split.llm_numerics, dtype=np.float64)[:, 2]
        # log-log transform puts both anchors on the scale a Cox linear
        # predictor works in; confidence enters untransformed
        return np.column_stack([np.log(-np.log(s5)), np.log(-np.log(s2)),
                                conf]).astype(np.float32)

    def fit(self, bundle):
        if bundle.train.llm_numerics is None:
            raise ValueError("LLM base learner requires llm_numerics on the split")
        if self.mode == "anchor":
            self.times_ = np.asarray(bundle.time_grid, dtype=np.float64)
            return
        X = self._feats(bundle.train)
        y = Surv.from_arrays(bundle.train.event.astype(bool), bundle.train.time)
        self.model = CoxPHSurvivalAnalysis(alpha=0.1, n_iter=200)
        self.model.fit(X, y)
        sf = self.model.predict_survival_function(X[:1])
        self.times_ = np.asarray(sf[0].x)

    def surv(self, bundle, split_name):
        split = getattr(bundle, split_name)
        grid = np.asarray(bundle.time_grid, dtype=np.float64)
        if self.mode == "anchor":
            s2, s5 = self._anchors(split)
            t1 = self.anchor_years[0] * self.unit_per_year
            t2 = self.anchor_years[1] * self.unit_per_year
            y1, y2 = np.log(-np.log(s2)), np.log(-np.log(s5))
            k = np.clip((y2 - y1) / (np.log(t2) - np.log(t1)), 0.1, 10.0)
            log_lam = np.log(t1) - y1 / k
            tt = np.maximum(grid[None, :], 1e-8)
            out = np.exp(-np.exp(k[:, None] * (np.log(tt) - log_lam[:, None])))
            return np.clip(out, 1e-6, 1.0).astype(np.float32)
        X = self._feats(split)
        sf = self.model.predict_survival_function(X)
        surv = np.stack([f(self.times_) for f in sf], axis=0)
        return _surv_at_grid(self.times_, surv, grid)


class _BaseDeepSurvText:
    """DeepSurv on structured covariates + PCA-reduced narrative embedding."""
    name = "deepsurv_text"

    def __init__(self, seed=0, text_pca_dim=64, hidden=(128, 64), dropout=0.2,
                 lr=1e-3, weight_decay=5e-4, batch_size=128, epochs=200,
                 patience=20, use_text=True, n_restarts=2):
        self.seed = seed
        self.text_pca_dim = text_pca_dim
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.use_text = use_text
        self.n_restarts = n_restarts
        self.model = None
        self.pca_components_ = None
        self.pca_mean_ = None

    def _fit_pca(self, X, k):
        mean = X.mean(axis=0, keepdims=True).astype(np.float64)
        Xc = X.astype(np.float64) - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.pca_components_ = Vt[:k].astype(np.float32)
        self.pca_mean_ = mean.astype(np.float32)

    def _transform_pca(self, X):
        Xc = X.astype(np.float32) - self.pca_mean_
        return (Xc @ self.pca_components_.T).astype(np.float32)

    def _build_X(self, bundle, split_name):
        split = getattr(bundle, split_name)
        X_struct = _onehot(split, bundle.cat_cardinalities)
        if self.use_text and split.text_embedding is not None and self.pca_components_ is not None:
            X_text = self._transform_pca(split.text_embedding)
            return np.concatenate([X_struct, X_text], axis=1).astype(np.float32)
        return X_struct

    def fit(self, bundle):
        np.random.seed(self.seed)
        if self.use_text and bundle.text_dim > 0:
            self._fit_pca(bundle.train.text_embedding,
                          min(self.text_pca_dim, bundle.text_dim))
        X_tr = self._build_X(bundle, "train")
        X_va = self._build_X(bundle, "val")
        t_tr = bundle.train.time.astype(np.float32)
        e_tr = bundle.train.event.astype(np.float32)
        t_va = bundle.val.time.astype(np.float32)
        e_va = bundle.val.event.astype(np.float32)

        best_val = float("inf")
        best_model = None
        for r in range(self.n_restarts):
            torch.manual_seed(self.seed * 1000 + r * 17 + 1)
            np.random.seed(self.seed * 1000 + r * 17 + 1)
            net = tt.practical.MLPVanilla(
                X_tr.shape[1], list(self.hidden), 1,
                batch_norm=True, dropout=self.dropout, output_bias=False,
            )
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            m = PCCoxPH(net, optimizer=opt)
            try:
                m.fit(
                    X_tr, (t_tr, e_tr),
                    batch_size=self.batch_size, epochs=self.epochs,
                    callbacks=[tt.callbacks.EarlyStopping(patience=self.patience)],
                    val_data=(X_va, (t_va, e_va)), verbose=False,
                )
                log_df = m.log.to_pandas()
                val_loss = float(log_df["val_loss"].dropna().iloc[-1])
            except Exception:
                continue
            if val_loss < best_val:
                best_val = val_loss
                best_model = m
        if best_model is not None:
            best_model.compute_baseline_hazards()
        self.model = best_model

    def surv(self, bundle, split_name):
        X = self._build_X(bundle, split_name)
        if self.model is None:
            # graceful degradation: all-1 curves so the stacker can still run
            return np.ones((len(X), len(bundle.time_grid)), dtype=np.float32)
        surv_df = self.model.predict_surv_df(X)
        times = np.asarray(surv_df.index)
        surv = surv_df.to_numpy().T
        return _surv_at_grid(times, surv, bundle.time_grid)


# ---------------------------------------------------------------------------
# Simplex meta-learner
# ---------------------------------------------------------------------------

def _simplex_grid(k: int, step: float):
    """Enumerate the k-simplex on a regular grid of the given step."""
    m = int(round(1.0 / step))
    def rec(rem, slots):
        if slots == 1:
            yield [rem]
            return
        for i in range(rem + 1):
            for tail in rec(rem - i, slots - 1):
                yield [i] + tail
    for combo in rec(m, k):
        yield np.array(combo, dtype=np.float64) / m


def _simplex_pgd(S_flat, y_flat, w_prior, lam, lr, steps):
    """Projected GD: min_w ||S w - y||^2 + lam ||w - w_prior||^2  s.t. w in simplex."""
    Ka = S_flat.shape[1]
    w = np.ones(Ka, dtype=np.float64) / Ka
    N = max(1, len(y_flat))
    for _ in range(steps):
        pred = S_flat @ w
        grad_fit = 2.0 * S_flat.T @ (pred - y_flat) / N
        grad_pri = 2.0 * lam * (w - w_prior)
        w = w - lr * (grad_fit + grad_pri)
        w = np.maximum(w, 0)
        s = w.sum()
        w = w / s if s > 0 else np.ones(Ka) / Ka
    return w


@dataclass
class TrainConfig:
    use_text: bool = True
    text_pca_dim: int = 64
    hidden: tuple = (128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 128
    epochs: int = 200
    patience: int = 20
    n_restarts: int = 2
    # L2 prior on the meta-weights toward [1/3, 1/3, 1/3]
    meta_prior_lam: float = 1.0  # headline setting (exp 53); 0.3 was the pre-revision default
    pgd_lr: float = 0.1
    pgd_steps: int = 200
    # Internal RSF base learner (exposed so LLM-SA can be tuned like the baselines)
    rsf_n_estimators: int = 100
    rsf_max_depth: Optional[int] = 6
    # Where the meta-weights are fitted:
    #   "val" -- on the held-out validation fold (original behaviour). On a small
    #            cohort that fold holds only a handful of events, leaving the
    #            three weights unidentifiable: unregularised the solver overfits
    #            it, regularised it returns the uniform prior.
    #   "oof" -- on K-fold out-of-fold predictions over the *training* fold, the
    #            standard stacking construction. Fits the weights on ~n_train
    #            patients instead of ~n_val, at the cost of K extra base fits.
    meta_fit: str = "val"
    meta_folds: int = 5
    # What the meta-weights optimise:
    #   "brier"  -- squared error against the survival labels (original).
    #   "cindex" -- IPCW C-index of the stacked risk, i.e. the metric actually
    #               reported. The original objective rewards calibration, so it
    #               can prefer a well-calibrated but poorly discriminating base
    #               learner over one that ranks patients better.
    meta_objective: str = "brier"
    meta_grid_step: float = 0.05
    # Add the LLM's own estimates as a FOURTH base learner, rather than as
    # covariates of the other three. Independent of ``use_numerics``: the clean
    # contrast is covariates-only learners (use_numerics=False) plus this one,
    # which isolates "LLM as a model" from "LLM as features".
    use_llm_base: bool = False
    llm_base_mode: str = "cox1"          # "cox1" (fitted) or "anchor" (zero-shot)
    #: (short, long) horizons the prompt requested, in years; set from
    #: ``anchor_years_for(feature_version)`` when the cohort was regenerated
    #: with cohort-matched horizons.
    llm_anchor_years: tuple = (2.0, 5.0)


def _subset_split(split: SurvivalSplit, idx: np.ndarray) -> SurvivalSplit:
    """Row subset of a split, preserving None embeddings."""
    return SurvivalSplit(
        X_num=split.X_num[idx],
        X_cat=split.X_cat[idx],
        time=split.time[idx],
        event=split.event[idx],
        text=split.text[idx] if len(split.text) else split.text,
        text_embedding=(None if split.text_embedding is None
                        else split.text_embedding[idx]),
    )


def _subset_bundle(bundle: DatasetBundle, tr_idx, ho_idx) -> DatasetBundle:
    """A view of ``bundle`` whose train fold is ``tr_idx`` and whose val/test
    folds are both the held-out part ``ho_idx``.

    val is set to the held-out rows so the DeepSurv branch still has an
    early-stopping signal, and test to the same rows so ``surv(sub, "test")``
    yields the out-of-fold predictions.
    """
    tr = _subset_split(bundle.train, tr_idx)
    ho = _subset_split(bundle.train, ho_idx)
    return DatasetBundle(
        name=bundle.name, train=tr, val=ho, test=ho,
        cat_cardinalities=bundle.cat_cardinalities,
        num_feature_names=bundle.num_feature_names,
        cat_feature_names=bundle.cat_feature_names,
        text_dim=bundle.text_dim, time_grid=bundle.time_grid,
    )


def _stratified_folds(event: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """Event-stratified K-fold assignment, so every fold carries some events."""
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for cls in (1, 0):
        pos = np.where(event.astype(int) == cls)[0]
        rng.shuffle(pos)
        for j, p in enumerate(pos):
            folds[j % k].append(p)
    return [np.array(sorted(f), dtype=int) for f in folds]


def _alive_targets(time: np.ndarray, event: np.ndarray,
                   grid: np.ndarray) -> np.ndarray:
    """Per (sample, grid) survival label: 1 alive, 0 event by t, NaN masked
    beyond a censoring time."""
    n, G = len(time), len(grid)
    y = np.ones((n, G), dtype=np.float32)
    for i in range(n):
        if event[i] == 1:
            y[i, grid >= time[i]] = 0.0
        else:
            y[i, grid > time[i]] = np.nan
    return y


class LLMSAStacking:
    """Three-base stacking ensemble with a simplex-constrained meta-learner."""

    name = "llmsa"
    BASE_NAMES = ("coxph", "rsf", "deepsurv_text")

    def __init__(self, train_cfg: TrainConfig, seed: int = 0):
        self.train_cfg = train_cfg
        self.seed = seed
        self.bases: List[object] = []
        self.weights_: Optional[np.ndarray] = None

    def _new_bases(self):
        cfg = self.train_cfg
        return [
            _BaseCoxPH(),
            _BaseRSF(seed=self.seed, n_estimators=cfg.rsf_n_estimators,
                     max_depth=cfg.rsf_max_depth),
            _BaseDeepSurvText(
                seed=self.seed,
                text_pca_dim=cfg.text_pca_dim, hidden=cfg.hidden,
                dropout=cfg.dropout, lr=cfg.lr, weight_decay=cfg.weight_decay,
                batch_size=cfg.batch_size, epochs=cfg.epochs,
                patience=cfg.patience, use_text=cfg.use_text,
                n_restarts=cfg.n_restarts,
            ),
        ]

    def _bases_for(self, bundle):
        bases = self._new_bases()
        if self.train_cfg.use_llm_base:
            bases.append(_BaseLLM(
                mode=self.train_cfg.llm_base_mode,
                unit_per_year=_TIME_UNIT_PER_YEAR.get(bundle.name.lower(), 365.25),
                anchor_years=self.train_cfg.llm_anchor_years,
                seed=self.seed))
        return bases

    def _meta_matrices(self, bundle: DatasetBundle):
        """Return (S, time, event): stacked base curves (K, n, G) on held-out
        rows, with the matching survival times and event indicators."""
        cfg = self.train_cfg
        grid = bundle.time_grid
        if cfg.meta_fit == "oof":
            # Out-of-fold predictions over the training fold. Each fold refits
            # the bases on the other folds, so the stacked predictions the
            # weights see are genuinely out-of-sample.
            n_tr = len(bundle.train)
            k = max(2, min(cfg.meta_folds, int(bundle.train.event.sum()), n_tr))
            folds = _stratified_folds(bundle.train.event, k, self.seed)
            S_parts, y_parts = [], []
            for ho_idx in folds:
                if len(ho_idx) == 0:
                    continue
                tr_idx = np.setdiff1d(np.arange(n_tr), ho_idx)
                if len(tr_idx) == 0 or bundle.train.event[tr_idx].sum() == 0:
                    continue
                sub = _subset_bundle(bundle, tr_idx, ho_idx)
                try:
                    fold_bases = self._bases_for(sub)
                    for b in fold_bases:
                        b.fit(sub)
                    S = np.stack([b.surv(sub, "test") for b in fold_bases], axis=0)
                except Exception:
                    continue          # a degenerate fold contributes nothing
                S_parts.append(S)
                y_parts.append(ho_idx)
            if S_parts:
                idx = np.concatenate(y_parts)
                return (np.concatenate(S_parts, axis=1),
                        bundle.train.time[idx], bundle.train.event[idx])
            # every fold failed -> fall back to the validation fold
        S_val = np.stack([b.surv(bundle, "val") for b in self.bases], axis=0)
        return S_val, bundle.val.time, bundle.val.event

    def fit(self, bundle: DatasetBundle):
        np.random.seed(self.seed)
        cfg = self.train_cfg

        # Final base learners are always fitted on the full training fold; the
        # out-of-fold refits below exist only to fit the meta-weights.
        self.bases = self._bases_for(bundle)
        for b in self.bases:
            b.fit(bundle)

        S, m_time, m_event = self._meta_matrices(bundle)
        K = S.shape[0]
        w_prior = np.ones(K, dtype=np.float64) / K

        if cfg.meta_objective == "cindex":
            # Stacked risk is linear in w: risk_i(w) = -sum_k w_k * u_ik with
            # u_ik = sum_g S_k[i, g]. So a coarse simplex grid is cheap, and it
            # optimises the metric actually reported instead of squared error.
            U = S.sum(axis=2).T                      # (n, K)
            best_w, best_s = w_prior, -np.inf
            for w in _simplex_grid(K, cfg.meta_grid_step):
                try:
                    s_ = ipcw_cindex(bundle.train.time, bundle.train.event,
                                     m_time, m_event, -(U @ w))
                except Exception:
                    continue
                pen = cfg.meta_prior_lam * float(np.sum((w - w_prior) ** 2))
                if s_ - pen > best_s:
                    best_w, best_s = w, s_ - pen
            self.weights_ = np.asarray(best_w, dtype=np.float32)
        else:
            y = _alive_targets(m_time, m_event, bundle.time_grid)
            S_flat = S.transpose(1, 2, 0).reshape(-1, K)
            y_flat = y.reshape(-1)
            mask = ~np.isnan(y_flat)
            w = _simplex_pgd(S_flat[mask], y_flat[mask], w_prior,
                             cfg.meta_prior_lam, cfg.pgd_lr, cfg.pgd_steps)
            self.weights_ = w.astype(np.float32)

    def _stack(self, bundle, split_name):
        S = np.stack([b.surv(bundle, split_name) for b in self.bases], axis=0)
        w = self.weights_.reshape(-1, 1, 1)
        return (w * S).sum(axis=0)

    def predict_risk(self, bundle, split_name):
        return -self._stack(bundle, split_name).sum(axis=1)

    def predict_surv(self, bundle, split_name):
        return self._stack(bundle, split_name)
