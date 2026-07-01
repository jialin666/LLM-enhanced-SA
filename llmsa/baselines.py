"""Baseline survival models used to populate baselines.tsv.

Each baseline implements the SurvivalModel protocol from eval.py:
    fit(bundle)
    predict_risk(bundle, "test")  -> (n,) higher = more risk
    predict_surv(bundle, "test")  -> (n, len(time_grid))

All models train on bundle.train (with bundle.val available for early stopping),
ignoring text embeddings - these are the "structured-covariate-only" baselines.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .data import DatasetBundle, SurvivalSplit  # noqa: E402


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _fuse_features(split: SurvivalSplit) -> np.ndarray:
    """Concatenate categorical + numerical features into a single matrix.

    Cat columns are kept as int (one-hot encoded by callers if needed); for
    the linear / tree baselines that's adequate because cardinalities are tiny.
    """
    return np.concatenate([split.X_num, split.X_cat.astype(np.float32)], axis=1)


def _onehot(split: SurvivalSplit, cardinalities: list) -> np.ndarray:
    pieces = [split.X_num]
    for j, k in enumerate(cardinalities):
        oh = np.zeros((len(split), k), dtype=np.float32)
        oh[np.arange(len(split)), split.X_cat[:, j]] = 1.0
        pieces.append(oh)
    return np.concatenate(pieces, axis=1).astype(np.float32)


def _surv_at_grid(times_pred: np.ndarray, surv_pred: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Step-function interpolation of survival curves onto `grid`.

    times_pred: (T,) increasing times at which surv_pred is defined
    surv_pred:  (n, T) survival probabilities
    grid:       (G,) target times
    """
    n, T = surv_pred.shape
    out = np.zeros((n, len(grid)), dtype=np.float32)
    # Right-continuous step: for each grid point t, find largest times_pred <= t.
    idx = np.searchsorted(times_pred, grid, side="right") - 1
    idx_clipped = np.clip(idx, 0, T - 1)
    out = surv_pred[:, idx_clipped]
    # For grid points before times_pred[0], survival = 1.
    before = idx < 0
    if before.any():
        out[:, before] = 1.0
    return out


# -----------------------------------------------------------------------------
# CoxPH (sksurv)
# -----------------------------------------------------------------------------

class CoxPH:
    name = "coxph"

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.model = None
        self.times_ = None
        self.cardinalities_ = None

    def fit(self, bundle: DatasetBundle) -> None:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
        self.cardinalities_ = bundle.cat_cardinalities
        X = _onehot(bundle.train, bundle.cat_cardinalities)
        y = Surv.from_arrays(bundle.train.event.astype(bool), bundle.train.time)
        self.model = CoxPHSurvivalAnalysis(alpha=self.alpha, n_iter=200)
        self.model.fit(X, y)
        sf = self.model.predict_survival_function(X[:1])
        self.times_ = np.asarray(sf[0].x)

    def predict_risk(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = _onehot(split, bundle.cat_cardinalities)
        return self.model.predict(X)

    def predict_surv(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = _onehot(split, bundle.cat_cardinalities)
        sf = self.model.predict_survival_function(X)
        surv = np.stack([f(self.times_) for f in sf], axis=0)
        return _surv_at_grid(self.times_, surv, bundle.time_grid)


# -----------------------------------------------------------------------------
# Random Survival Forest (sksurv)
# -----------------------------------------------------------------------------

class RSF:
    name = "rsf"

    def __init__(self, n_estimators: int = 100, max_depth: int = 6, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.model = None
        self.times_ = None

    def fit(self, bundle: DatasetBundle) -> None:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        X = _onehot(bundle.train, bundle.cat_cardinalities)
        y = Surv.from_arrays(bundle.train.event.astype(bool), bundle.train.time)
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=15,
            n_jobs=1,
            random_state=self.seed,
        )
        self.model.fit(X, y)
        sf = self.model.predict_survival_function(X[:1])
        self.times_ = np.asarray(sf[0].x)

    def predict_risk(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = _onehot(split, bundle.cat_cardinalities)
        return self.model.predict(X)

    def predict_surv(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = _onehot(split, bundle.cat_cardinalities)
        sf = self.model.predict_survival_function(X)
        surv = np.stack([f(self.times_) for f in sf], axis=0)
        return _surv_at_grid(self.times_, surv, bundle.time_grid)


# -----------------------------------------------------------------------------
# pycox base helper
# -----------------------------------------------------------------------------

class _PycoxBase:
    """Common fit logic for DeepSurv / DeepHit / CoxTime.

    Subclasses implement `_build_model(in_features, num_durations)`.
    """
    name = "pycox-base"

    def __init__(self, hidden=(64, 64), lr=1e-3, batch_size=128, epochs=200, seed: int = 0):
        self.hidden = hidden
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.model = None
        self.times_ = None

    def _onehot(self, split: SurvivalSplit, cards: list) -> np.ndarray:
        return _onehot(split, cards)

    def fit(self, bundle: DatasetBundle) -> None:
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_tr = self._onehot(bundle.train, bundle.cat_cardinalities).astype(np.float32)
        X_va = self._onehot(bundle.val, bundle.cat_cardinalities).astype(np.float32)
        t_tr = bundle.train.time.astype(np.float32)
        e_tr = bundle.train.event.astype(np.float32)
        t_va = bundle.val.time.astype(np.float32)
        e_va = bundle.val.event.astype(np.float32)

        self.model = self._make_pycox_model(
            in_features=X_tr.shape[1],
            train_time=t_tr,
            train_event=e_tr,
        )
        y_tr = self._prepare_targets(t_tr, e_tr)
        y_va = self._prepare_targets(t_va, e_va, fit=False)
        val = (X_va, y_va) if y_va is not None else None

        callbacks = self._callbacks()
        self.model.fit(
            X_tr, y_tr,
            batch_size=self.batch_size, epochs=self.epochs,
            callbacks=callbacks, val_data=val, verbose=False,
        )
        self._finalize(X_tr, t_tr, e_tr)

    def _callbacks(self):
        try:
            import torchtuples as tt
            return [tt.callbacks.EarlyStopping(patience=10)]
        except Exception:
            return None


# -----------------------------------------------------------------------------
# DeepSurv (Cox-style proportional-hazards NN)
# -----------------------------------------------------------------------------

class DeepSurv(_PycoxBase):
    name = "deepsurv"

    def _make_pycox_model(self, in_features, train_time, train_event):
        import torch
        import torchtuples as tt
        from pycox.models import CoxPH as PCCoxPH
        net = tt.practical.MLPVanilla(
            in_features, list(self.hidden), 1,
            batch_norm=True, dropout=0.1, output_bias=False,
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        model = PCCoxPH(net, optimizer=opt)
        return model

    def _prepare_targets(self, time, event, fit: bool = True):
        return (time, event)

    def _finalize(self, X_tr, t_tr, e_tr):
        self.model.compute_baseline_hazards()

    def predict_risk(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = self._onehot(split, bundle.cat_cardinalities).astype(np.float32)
        risk = self.model.predict(X).reshape(-1)
        return risk

    def predict_surv(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = self._onehot(split, bundle.cat_cardinalities).astype(np.float32)
        surv_df = self.model.predict_surv_df(X)
        times = np.asarray(surv_df.index)
        surv = surv_df.to_numpy().T  # (n, T)
        return _surv_at_grid(times, surv, bundle.time_grid)


# -----------------------------------------------------------------------------
# DeepHit (discrete-time, single-event)
# -----------------------------------------------------------------------------

class DeepHit(_PycoxBase):
    name = "deephit"

    def __init__(self, *args, num_durations: int = 50, **kw):
        super().__init__(*args, **kw)
        self.num_durations = num_durations
        self.labtrans = None

    def _make_pycox_model(self, in_features, train_time, train_event):
        import torch
        import torchtuples as tt
        from pycox.models import DeepHitSingle
        # Build discretizer from training durations.
        self.labtrans = DeepHitSingle.label_transform(self.num_durations)
        self.labtrans.fit_transform(train_time, train_event)
        out_features = self.labtrans.out_features
        net = tt.practical.MLPVanilla(
            in_features, list(self.hidden), out_features,
            batch_norm=True, dropout=0.1, output_bias=False,
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        model = DeepHitSingle(
            net, optimizer=opt, alpha=0.2, sigma=0.1,
            duration_index=self.labtrans.cuts,
        )
        return model

    def _prepare_targets(self, time, event, fit: bool = True):
        if fit:
            return self.labtrans.transform(time, event)
        return self.labtrans.transform(time, event)

    def _finalize(self, X_tr, t_tr, e_tr):
        pass

    def predict_risk(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = self._onehot(split, bundle.cat_cardinalities).astype(np.float32)
        surv_df = self.model.predict_surv_df(X)
        # Risk = 1 - integrated survival up to the last bin, i.e. CDF mass at max.
        return 1.0 - surv_df.to_numpy().sum(axis=0)

    def predict_surv(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = self._onehot(split, bundle.cat_cardinalities).astype(np.float32)
        surv_df = self.model.predict_surv_df(X)
        times = np.asarray(surv_df.index)
        surv = surv_df.to_numpy().T
        return _surv_at_grid(times, surv, bundle.time_grid)


# -----------------------------------------------------------------------------
# CoxTime
# -----------------------------------------------------------------------------

class CoxTime(_PycoxBase):
    name = "coxtime"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.labtrans = None

    def _make_pycox_model(self, in_features, train_time, train_event):
        import torch
        import torchtuples as tt
        from pycox.models import CoxTime as PCCoxTime
        from pycox.models.cox_time import MLPVanillaCoxTime
        self.labtrans = PCCoxTime.label_transform()
        self.labtrans.fit_transform(train_time, train_event)
        net = MLPVanillaCoxTime(
            in_features, list(self.hidden),
            batch_norm=True, dropout=0.1,
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        model = PCCoxTime(net, optimizer=opt, labtrans=self.labtrans)
        return model

    def _prepare_targets(self, time, event, fit: bool = True):
        if fit:
            return self.labtrans.transform(time, event)
        return self.labtrans.transform(time, event)

    def _finalize(self, X_tr, t_tr, e_tr):
        self.model.compute_baseline_hazards()

    def predict_risk(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = self._onehot(split, bundle.cat_cardinalities).astype(np.float32)
        surv_df = self.model.predict_surv_df(X)
        return 1.0 - surv_df.to_numpy().mean(axis=0)

    def predict_surv(self, bundle: DatasetBundle, split_name: str) -> np.ndarray:
        split = getattr(bundle, split_name)
        X = self._onehot(split, bundle.cat_cardinalities).astype(np.float32)
        surv_df = self.model.predict_surv_df(X)
        times = np.asarray(surv_df.index)
        surv = surv_df.to_numpy().T
        return _surv_at_grid(times, surv, bundle.time_grid)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

BASELINE_FACTORIES = {
    "coxph": lambda seed: CoxPH(),
    "rsf": lambda seed: RSF(seed=seed),
    "deepsurv": lambda seed: DeepSurv(seed=seed),
    "deephit": lambda seed: DeepHit(seed=seed),
    "coxtime": lambda seed: CoxTime(seed=seed),
}
