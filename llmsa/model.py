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
fitted on val, and everything is evaluated on test.
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

    def __init__(self, seed=0):
        self.seed = seed
        self.model = None
        self.times_ = None

    def fit(self, bundle):
        X = _onehot(bundle.train, bundle.cat_cardinalities)
        y = Surv.from_arrays(bundle.train.event.astype(bool), bundle.train.time)
        self.model = RandomSurvivalForest(
            n_estimators=100, max_depth=6, min_samples_leaf=15,
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
    meta_prior_lam: float = 0.3
    pgd_lr: float = 0.1
    pgd_steps: int = 200


class LLMSAStacking:
    """Three-base stacking ensemble with a simplex-constrained meta-learner."""

    name = "llmsa"
    BASE_NAMES = ("coxph", "rsf", "deepsurv_text")

    def __init__(self, train_cfg: TrainConfig, seed: int = 0):
        self.train_cfg = train_cfg
        self.seed = seed
        self.bases: List[object] = []
        self.weights_: Optional[np.ndarray] = None

    def fit(self, bundle: DatasetBundle):
        np.random.seed(self.seed)
        cfg = self.train_cfg

        b1 = _BaseCoxPH()
        b2 = _BaseRSF(seed=self.seed)
        b3 = _BaseDeepSurvText(
            seed=self.seed,
            text_pca_dim=cfg.text_pca_dim, hidden=cfg.hidden, dropout=cfg.dropout,
            lr=cfg.lr, weight_decay=cfg.weight_decay, batch_size=cfg.batch_size,
            epochs=cfg.epochs, patience=cfg.patience, use_text=cfg.use_text,
            n_restarts=cfg.n_restarts,
        )
        self.bases = [b1, b2, b3]
        for b in self.bases:
            b.fit(bundle)

        # Base survival curves on the validation fold -> fit the meta-weights.
        S_val = np.stack([b.surv(bundle, "val") for b in self.bases], axis=0)
        K = S_val.shape[0]

        grid = bundle.time_grid
        n_val = len(bundle.val)
        G = len(grid)
        # Per (sample, grid) survival label: 1 if alive at t, 0 if event by t,
        # NaN (masked) for grid points beyond a censoring time.
        y_alive = np.ones((n_val, G), dtype=np.float32)
        for i in range(n_val):
            t = bundle.val.time[i]; e = bundle.val.event[i]
            if e == 1:
                y_alive[i, grid >= t] = 0.0
            else:
                y_alive[i, grid > t] = np.nan

        S_flat = S_val.transpose(1, 2, 0).reshape(-1, K)
        y_flat = y_alive.reshape(-1)
        mask = ~np.isnan(y_flat)
        S_flat = S_flat[mask]; y_flat = y_flat[mask]

        w_prior = np.ones(K, dtype=np.float64) / K
        w = _simplex_pgd(S_flat, y_flat, w_prior, cfg.meta_prior_lam, cfg.pgd_lr, cfg.pgd_steps)
        self.weights_ = w.astype(np.float32)

    def _stack(self, bundle, split_name):
        S = np.stack([b.surv(bundle, split_name) for b in self.bases], axis=0)
        w = self.weights_.reshape(-1, 1, 1)
        return (w * S).sum(axis=0)

    def predict_risk(self, bundle, split_name):
        return -self._stack(bundle, split_name).sum(axis=1)

    def predict_surv(self, bundle, split_name):
        return self._stack(bundle, split_name)
