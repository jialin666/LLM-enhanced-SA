"""
DeepHit (PyTorch) — DataFrame-only, preprocessed inputs
Metrics computed with scikit-survival (sksurv):
  - C-index: sksurv.metrics.concordance_index_censored
  - IBS:     sksurv.metrics.integrated_brier_score

Assumptions:
  - You already preprocessed covariates (numeric, standardized/encoded/embeddings merged, etc.)
  - Inputs are pandas.DataFrame: train_data, validation_data, test_data
  - Each DataFrame contains covariates + time_col + event_col
  - event: 1=event, 0=censored

Model:
  - Discretizes time into n_bins based on TRAIN time (quantile or linear).
  - Predicts PMF over time bins; survival curve S(t) derived from PMF.

Evaluation:
  - C-index (sksurv) from scalar risk = -E[T] (expected failure time from PMF).
  - IBS (sksurv) from survival curves interpolated onto a common time grid.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Data preprocessing (self-contained)
# -----------------------------
def prepare_survival_data(
    data_path: str,
    categorical_columns: List[str],
    numerical_columns: List[str],
    time_col: str = "time",
    event_col: str = "event",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load CSV, split into train/val/test, one-hot encode categoricals,
    StandardScale numericals (fit on train only), and return processed
    DataFrames together with the list of covariate column names.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Each contains covariate columns + time_col + event_col.
    covar_cols : List[str]
        Names of the covariate columns (one-hot cats + scaled numericals),
        with constant/near-zero-variance features removed.
    """
    df = pd.read_csv(data_path)

    # Split train+val / test, then train / val
    trainval_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_frac = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(trainval_df, test_size=val_frac, random_state=random_state)

    # One-hot encode categoricals (fit on train only, drop='first' to avoid collinearity)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(train_df[categorical_columns])
    cat_names = encoder.get_feature_names_out(categorical_columns).tolist()

    def _encode_cat(split_df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            encoder.transform(split_df[categorical_columns]),
            columns=cat_names,
            index=split_df.index,
        )

    train_cat = _encode_cat(train_df)
    val_cat = _encode_cat(val_df)
    test_cat = _encode_cat(test_df)

    # StandardScale numericals (fit on train only)
    scaler = StandardScaler()
    scaler.fit(train_df[numerical_columns])

    for split, cat in [(train_df, train_cat), (val_df, val_cat), (test_df, test_cat)]:
        split[numerical_columns] = scaler.transform(split[numerical_columns])

    # Reassemble: [one-hot cats | scaled numericals | time | event]
    keep_cols = [time_col, event_col]
    train_out = pd.concat([train_cat, train_df[numerical_columns], train_df[keep_cols]], axis=1)
    val_out   = pd.concat([val_cat,   val_df[numerical_columns],   val_df[keep_cols]],   axis=1)
    test_out  = pd.concat([test_cat,  test_df[numerical_columns],  test_df[keep_cols]],  axis=1)

    val_out  = val_out[train_out.columns]
    test_out = test_out[train_out.columns]

    # Covariate column list, excluding constant/near-zero-variance features
    covar_cols = cat_names + numerical_columns
    constant_features = [
        col for col in covar_cols
        if col in train_out.columns and (
            train_out[col].nunique() <= 1 or train_out[col].std() < 1e-8
        )
    ]
    if constant_features:
        print(f"Warning: Removing constant/near-constant features: {constant_features}")
        covar_cols = [c for c in covar_cols if c not in constant_features]

    return train_out, val_out, test_out, covar_cols


# -----------------------------
# Device helper (self-contained)
# -----------------------------
def get_best_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Time grid helper (self-contained)
# -----------------------------
def create_time_grid(times: np.ndarray, n_points: int = 200, eps: float = 1e-8) -> np.ndarray:
    """
    Create a time grid for IBS based on test times.
    Uses quantiles to reduce tail sensitivity.
    """
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    t = t[t > 0]
    if t.size == 0:
        return np.linspace(0.1, 1.0, n_points)

    t_min = max(np.quantile(t, 0.01), eps)
    t_max = max(np.quantile(t, 0.99), t_min + eps)
    return np.linspace(t_min, t_max, int(max(n_points, 3)))


# -----------------------------
# Dataset (DataFrame-only)
# -----------------------------
class SurvivalDataFrameDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        time_col: str = "time",
        event_col: str = "event",
    ):
        if time_col not in df.columns or event_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{time_col}' and '{event_col}' columns.")

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in (time_col, event_col)]

        x = df[feature_cols].to_numpy(dtype=np.float32)
        time = df[time_col].to_numpy(dtype=np.float32)
        event = df[event_col].to_numpy(dtype=np.int64)

        self.feature_cols = feature_cols
        self.x = torch.from_numpy(x)
        self.time = torch.from_numpy(time)
        self.event = torch.from_numpy(event)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.time[i], self.event[i]


# -----------------------------
# Time discretization
# -----------------------------
def make_time_bins(time: np.ndarray, n_bins: int, strategy: str = "quantile"):
    """
    Returns: (bin_edges, encode_fn, bin_centers)
      - encode_fn(t): map continuous times -> integer bin indices [0..K-1]
      - bin_centers: (K,) representative times (for evaluation / interpolation)
    """
    t = np.asarray(time, dtype=float)
    if t.size == 0:
        raise ValueError("Empty time array.")

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(t, qs)
        edges = np.maximum.accumulate(edges)  # enforce non-decreasing
    else:
        edges = np.linspace(t.min(), t.max(), n_bins + 1)

    # Make sure edges are strictly increasing enough for interpolation stability
    # If duplicates exist (common in quantiles with ties), nudge slightly
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-8

    centers = 0.5 * (edges[:-1] + edges[1:])

    def encode(times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        # idx in [0..K-1]
        idx = np.searchsorted(edges[1:-1], times, side="right")
        return idx.astype(np.int64)

    return edges, encode, centers


# -----------------------------
# DeepHit model
# -----------------------------
class DeepHit(nn.Module):
    """
    Outputs logits over K time bins.
    Softmax(logits) = PMF over bins.
    Survival at bin k: 1 - cumsum(PMF)_k
    """
    def __init__(
        self,
        in_features: int,
        n_bins: int,
        hidden: Iterable[int] = (512, 256),
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation.__class__())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_bins))  # logits over bins
        self.net = nn.Sequential(*layers)
        self.n_bins = n_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (N, K) logits


# -----------------------------
# DeepHit loss (event/censor + optional ranking)
# -----------------------------
class DeepHitLoss(nn.Module):
    """
    Negative log-likelihood for DeepHit (single-risk), with optional ranking loss.

    Targets:
      - bin_index: integer bin for each sample (0..K-1)
      - event: 1 if event observed, 0 if censored

    PMF p_k(x) from softmax(logits):
      - For events:   -log p_k
      - For censored: -log sum_{j > k} p_j  (= -log S(t))
    """
    def __init__(self, alpha: float = 0.0, sigma: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.sigma = float(sigma)

    def forward(self, logits: torch.Tensor, bin_index: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        pmf = torch.softmax(logits, dim=1)  # (N, K)
        N, K = pmf.shape

        idx = bin_index.unsqueeze(1)
        p_at_k = torch.gather(pmf, 1, idx).squeeze(1)  # (N,)

        device = logits.device
        tri = torch.triu(torch.ones(K, K, device=device), diagonal=1)  # strict upper triangle
        tail = pmf @ tri  # (N, K), tail prob for each k
        s_at_k = torch.gather(tail, 1, idx).squeeze(1)

        eps = 1e-8
        nll_event = -torch.log(p_at_k + eps)
        nll_cens = -torch.log(s_at_k + eps)
        nll = torch.where(event.bool(), nll_event, nll_cens)
        loss = nll.mean()

        # Optional ranking loss (simple surrogate)
        if self.alpha > 0:
            with torch.no_grad():
                bins = torch.arange(K, device=device, dtype=pmf.dtype)
            expected_bin = (pmf * bins.unsqueeze(0)).sum(dim=1)  # (N,)
            ev_mask = event.bool()
            i_idx = torch.nonzero(ev_mask, as_tuple=False).squeeze(1)
            if i_idx.numel() > 0:
                t_i = expected_bin[i_idx].unsqueeze(1)
                t_all = expected_bin.unsqueeze(0)
                comp = (t_all > t_i).float()
                margin = (t_all - t_i) / (self.sigma + 1e-12)
                rank_term = torch.log1p(torch.exp(-margin)) * comp
                if comp.sum() > 0:
                    loss_rank = rank_term.sum() / (comp.sum() + 1e-8)
                    loss = loss + self.alpha * loss_rank

        return loss


# -----------------------------
# Inference helpers
# -----------------------------
@torch.no_grad()
def deephit_pmf(model: DeepHit, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    logits = model(t)
    return torch.softmax(logits, dim=1).cpu().numpy()  # (N, K)

@torch.no_grad()
def deephit_survival_from_pmf(pmf: np.ndarray) -> np.ndarray:
    surv = 1.0 - np.cumsum(pmf, axis=1)
    return np.clip(surv, 0.0, 1.0)


# -----------------------------
# sksurv metrics: C-index + IBS
# -----------------------------
def sksurv_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    from sksurv.metrics import concordance_index_censored
    c, *_ = concordance_index_censored(np.asarray(event).astype(bool), np.asarray(time).astype(float), np.asarray(risk).astype(float))
    return float(c)

def ibs_with_sksurv(
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    surv_test: np.ndarray,     # (N_test, M) on time_grid
    time_grid: np.ndarray,     # (M,)
) -> float:
    from sksurv.util import Surv
    from sksurv.metrics import integrated_brier_score

    y_tr = Surv.from_arrays(np.asarray(event_train).astype(bool), np.asarray(time_train).astype(float))
    y_te = Surv.from_arrays(np.asarray(event_test).astype(bool), np.asarray(time_test).astype(float))

    # Keep grid inside test range (optional)
    tmin, tmax = float(np.min(time_test)), float(np.max(time_test))
    mask = (time_grid >= tmin) & (time_grid <= tmax)
    tg = time_grid[mask]
    st = surv_test[:, mask]

    if tg.size < 2:
        tg = time_grid
        st = surv_test

    return float(integrated_brier_score(y_tr, y_te, st, tg))


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10
    device: str = get_best_device()


# -----------------------------
# Train DeepHit from DataFrames (no X standardization here)
# -----------------------------
def train_deephit_from_dfs(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    time_col: str = "time",
    event_col: str = "event",
    n_bins: int = 50,
    hidden: Iterable[int] = (512, 256),
    dropout: float = 0.1,
    batch_norm: bool = True,
    alpha_rank: float = 0.0,
    bin_strategy: str = "quantile",
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[DeepHit, dict]:
    """
    Train DeepHit using preprocessed DataFrames.
    - Covariates are taken as-is (already preprocessed).
    - Time discretization is learned from TRAIN times.
    """
    set_seed(42)

    train_ds = SurvivalDataFrameDataset(train_df, feature_cols, time_col, event_col)
    feature_cols_used = train_ds.feature_cols

    # Discretize using TRAIN times
    edges, encode, centers = make_time_bins(train_df[time_col].to_numpy(), n_bins=n_bins, strategy=bin_strategy)
    bin_train = encode(train_df[time_col].to_numpy())

    # Torch dataset: (x, bin_index, event)
    class _D(Dataset):
        def __init__(self, x: torch.Tensor, b: np.ndarray, e: np.ndarray):
            self.x = x
            self.b = torch.from_numpy(np.asarray(b, dtype=np.int64))
            self.e = torch.from_numpy(np.asarray(e, dtype=np.int64))
        def __len__(self): return self.x.shape[0]
        def __getitem__(self, i): return self.x[i], self.b[i], self.e[i]

    ds_tr = _D(train_ds.x, bin_train, train_df[event_col].to_numpy())
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # Validation tensors prepared once (full-batch)
    if val_df is not None:
        # Encode val times using TRAIN binning
        bin_val = encode(val_df[time_col].to_numpy())
        val_ds = SurvivalDataFrameDataset(val_df, feature_cols_used, time_col, event_col)
        x_val = val_ds.x.to(cfg.device)
        b_val = torch.from_numpy(bin_val.astype(np.int64)).to(cfg.device)
        e_val = torch.from_numpy(val_df[event_col].to_numpy(dtype=np.int64)).to(cfg.device)
    else:
        x_val = b_val = e_val = None

    model = DeepHit(
        in_features=train_ds.x.shape[1],
        n_bins=n_bins,
        hidden=hidden,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(cfg.device)

    criterion = DeepHitLoss(alpha=alpha_rank)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val = math.inf
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0

        for xb, bb, eb in dl_tr:
            xb = xb.to(cfg.device)
            bb = bb.to(cfg.device)
            eb = eb.to(cfg.device)

            logits = model(xb)
            loss = criterion(logits, bb, eb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)

        tr_loss = total / max(count, 1)

        val_loss = None
        if val_df is not None:
            model.eval()
            with torch.no_grad():
                logits = model(x_val)
                val_loss = float(criterion(logits, b_val, e_val).item())

            if val_loss < best_val:
                best_val = val_loss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    break

        print(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.4f}"
            + (f" | val_loss={val_loss:.4f}" if val_loss is not None else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {
        "edges": edges,
        "centers": centers,
        "encode_time_to_bin": encode,  # function
        "bin_strategy": bin_strategy,
        "n_bins": n_bins,
        "config": cfg,
        "feature_cols": feature_cols_used,
        "time_col": time_col,
        "event_col": event_col,
        "hidden": tuple(hidden),
        "dropout": dropout,
        "alpha_rank": alpha_rank,
    }
    return model, info


# -----------------------------
# End-to-end evaluation (DataFrames in, sksurv metrics out)
# -----------------------------
def evaluate_deephit_with_sksurv(
    model: DeepHit,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
    centers: np.ndarray,           # bin centers from TRAIN
    device: str,
    n_times: int = 200,
) -> dict:
    # PMF on test
    Xte = test_df[feature_cols].to_numpy(dtype=np.float32)
    pmf_te = deephit_pmf(model, Xte, device=device)            # (N_test, K)
    surv_bins = deephit_survival_from_pmf(pmf_te)              # (N_test, K) on bin-index grid (centers)

    # C-index (sksurv) from risk = -E[T]
    exp_t = (pmf_te * centers.reshape(1, -1)).sum(axis=1)
    risk = -exp_t
    cindex = sksurv_c_index(test_df[time_col].to_numpy(), test_df[event_col].to_numpy(), risk)

    # IBS (sksurv): interpolate survival from bin centers -> common time grid
    grid = create_time_grid(test_df[time_col].to_numpy(), n_points=n_times)

    # Interpolate each survival curve
    surv_interp = np.zeros((surv_bins.shape[0], grid.size), dtype=np.float64)
    # Ensure centers strictly increasing for np.interp
    centers_inc = np.asarray(centers, dtype=float)
    if np.any(np.diff(centers_inc) <= 0):
        order = np.argsort(centers_inc)
        centers_inc = centers_inc[order]
        surv_bins = surv_bins[:, order]

    for i in range(surv_bins.shape[0]):
        # np.interp extrapolates with boundary values by default (left/right)
        surv_interp[i] = np.interp(grid, centers_inc, surv_bins[i])

    # IBS
    ibs = ibs_with_sksurv(
        time_train=train_df[time_col].to_numpy(),
        event_train=train_df[event_col].to_numpy(),
        time_test=test_df[time_col].to_numpy(),
        event_test=test_df[event_col].to_numpy(),
        surv_test=surv_interp,
        time_grid=grid,
    )

    return {
        "time_grid": grid,
        "pmf_test": pmf_te,
        "surv_test_on_bins": surv_bins,
        "surv_test_on_grid": surv_interp,
        "c_index_test": cindex,
        "ibs_test": ibs,
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # You already have:
    # train_data, validation_data, test_data (all pandas.DataFrame)

    # Optional: specify feature columns explicitly (recommended)
    # feature_cols = ['meno','grade','hormon','age','size','nodes','pgr','er']
    feature_cols = None  # if None, inferred as all columns except time/event
    data_path = "data/1_gbsg_all.csv"
    categorical_columns = ['meno', 'grade', 'hormon']
    numerical_columns = ['age', 'size', 'nodes', 'pgr', 'er']
    train_df, val_df, test_df, covar_cols = prepare_survival_data(
        data_path, categorical_columns, numerical_columns
    )
    cfg = TrainConfig(batch_size=64, epochs=30, patience=10)

    model, info = train_deephit_from_dfs(
        train_df=train_df,
        val_df=val_df,
        feature_cols=covar_cols,
        time_col="time",
        event_col="event",
        n_bins=50,
        alpha_rank=0.0,
        bin_strategy="quantile",
        cfg=cfg,
    )

    results = evaluate_deephit_with_sksurv(
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=info["feature_cols"],
        time_col=info["time_col"],
        event_col=info["event_col"],
        centers=info["centers"],
        device=info["config"].device,
        n_times=200,
    )

    print("Test C-index (sksurv):", results["c_index_test"])
    print("Test IBS (sksurv):", results["ibs_test"])
