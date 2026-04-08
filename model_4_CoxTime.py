"""
CoxTime (PyTorch) — DataFrame-only, preprocessed inputs
Metrics computed with scikit-survival (sksurv):
  - C-index: sksurv.metrics.concordance_index_censored
  - IBS:     sksurv.metrics.integrated_brier_score

Assumptions:
  - You already preprocessed covariates (numeric, standardized/encoded/embeddings merged, etc.)
  - Inputs are pandas.DataFrame: train_data, validation_data, test_data
  - Each DataFrame contains covariates + time_col + event_col
  - event: 1=event, 0=censored

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
# CoxTime model
# -----------------------------
class CoxTimeNet(nn.Module):
    """
    Cox-Time network: f_theta(x, t) -> log-risk at time t.
    We concatenate covariates x and a standardized scalar time input.
    """
    def __init__(
        self,
        in_features: int,
        hidden: Iterable[int] = (128, 64),
        dropout: float = 0.1,
        batch_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.time_norm_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.time_norm_std = nn.Parameter(torch.ones(1), requires_grad=False)

        layers: List[nn.Module] = []
        prev = in_features + 1  # + time
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(activation.__class__())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def set_time_norm(self, mean: float, std: float):
        self.time_norm_mean.data = torch.tensor([float(mean)])
        self.time_norm_std.data = torch.tensor([float(std) if std > 0 else 1.0])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t_std = (t - self.time_norm_mean) / (self.time_norm_std + 1e-8)
        z = torch.cat([x, t_std], dim=1)
        return self.net(z).squeeze(-1)  # (N,)


class CoxTimeLoss(nn.Module):
    """
    Negative log partial likelihood for Cox-Time (Breslow ties).

    For each unique event time tau:
        - sum_{i: event at tau} f(x_i, tau)
        + d(tau) * log( sum_{j: T_j >= tau} exp(f(x_j, tau)) )
    """
    def forward(
        self,
        f_t: callable,          # function (x, t) -> log-risk
        x: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device

        # Sort by descending time for risk sets
        order = torch.argsort(time, descending=True)
        x = x[order]
        time = time[order]
        event = event[order]

        ev_mask = event == 1
        if ev_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        t_e = time[ev_mask]
        uniq, inv, counts = torch.unique_consecutive(t_e, return_inverse=True, return_counts=True)

        K = uniq.shape[0]
        sum_logrisk_events = torch.zeros(K, device=device)
        log_denom = torch.zeros(K, device=device)

        idx_e = torch.nonzero(ev_mask, as_tuple=False).squeeze(1)

        for k in range(K):
            tau = uniq[k]
            risk_mask = time >= tau
            tau_vec = torch.full((x.size(0),), tau, device=device, dtype=time.dtype)

            f_all = f_t(x, tau_vec)  # (N,)

            # Events occurring at this tau (within sorted arrays)
            at_tau_mask = torch.zeros_like(event, dtype=torch.bool)
            at_tau_mask[idx_e[inv == k]] = True

            sum_logrisk_events[k] = f_all[at_tau_mask].sum()
            log_denom[k] = torch.log(torch.sum(torch.exp(f_all[risk_mask])) + 1e-12)

        loss = -(sum_logrisk_events - counts.to(device) * log_denom).sum()
        loss = loss / counts.sum()
        return loss


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 100
    patience: int = 10
    device: str = get_best_device()


# -----------------------------
# Train CoxTime from DataFrames (no X standardization here)
# -----------------------------
def train_coxtime_from_dfs(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    time_col: str = "time",
    event_col: str = "event",
    hidden: Iterable[int] = (128, 64),
    dropout: float = 0.1,
    batch_norm: bool = True,
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[CoxTimeNet, dict]:
    """
    Train CoxTime using preprocessed DataFrames.
    - Covariates are taken as-is (already preprocessed).
    - Time normalization stats are computed from TRAIN time and stored in the model.
    """
    set_seed(42)

    train_ds = SurvivalDataFrameDataset(train_df, feature_cols, time_col, event_col)
    feature_cols_used = train_ds.feature_cols

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    # Time normalization based on training set
    t_mean = float(train_df[time_col].to_numpy(dtype=float).mean())
    t_std = float(train_df[time_col].to_numpy(dtype=float).std() + 1e-8)

    model = CoxTimeNet(
        in_features=train_ds.x.shape[1],
        hidden=hidden,
        dropout=dropout,
        batch_norm=batch_norm,
    )
    model.set_time_norm(t_mean, t_std)
    model.to(cfg.device)

    criterion = CoxTimeLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def f_t(Xb: torch.Tensor, Tb: torch.Tensor) -> torch.Tensor:
        return model(Xb, Tb)

    # Prepare validation tensors once (full-batch eval)
    if val_df is not None:
        val_ds = SurvivalDataFrameDataset(val_df, feature_cols_used, time_col, event_col)
        x_val = val_ds.x.to(cfg.device)
        t_val = val_ds.time.to(cfg.device)
        e_val = val_ds.event.to(cfg.device)
    else:
        x_val = t_val = e_val = None

    best_state = None
    best_val = math.inf
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for xb, tb, eb in train_dl:
            xb = xb.to(cfg.device)
            tb = tb.to(cfg.device)
            eb = eb.to(cfg.device)

            loss = criterion(f_t, xb, tb, eb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        tr_loss = total / max(n, 1)

        val_loss = None
        if val_df is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(f_t, x_val, t_val, e_val).item())

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
        "config": cfg,
        "feature_cols": feature_cols_used,
        "time_col": time_col,
        "event_col": event_col,
        "time_mean": t_mean,
        "time_std": t_std,
    }
    return model, info


# -----------------------------
# Baseline increments + survival curves (for IBS)
# -----------------------------
@torch.no_grad()
def coxtime_baseline_hazard(
    model: CoxTimeNet,
    x_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate baseline cumulative hazard increments on unique event times using Breslow-type estimator.
    Returns (uniq_event_times, dLambda).
    """
    order = np.argsort(time_train)
    times = time_train[order]
    events = event_train[order].astype(bool)

    uniq = np.unique(times[events])
    if uniq.size == 0:
        return uniq, np.zeros((0,), dtype=np.float64)

    X = torch.from_numpy(x_train.astype(np.float32)).to(device)
    T = torch.from_numpy(time_train.astype(np.float32)).to(device)

    dLambda = np.zeros_like(uniq, dtype=np.float64)

    for k, tau in enumerate(uniq):
        tau_t = torch.full((X.size(0),), float(tau), device=device)
        f_all = model(X, tau_t)  # (N,)
        risk = torch.exp(f_all)

        at_risk = (T >= tau_t).float()
        denom = (risk * at_risk).sum().item() + 1e-12

        d_k = int((times[events] == tau).sum())
        dLambda[k] = float(d_k) / denom

    return uniq, dLambda


@torch.no_grad()
def coxtime_survival(
    model: CoxTimeNet,
    x: np.ndarray,
    time_grid: np.ndarray,
    uniq_train_times: np.ndarray,
    dLambda: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Compute survival S_i(t) on time_grid using baseline increments from train set.
    S_i(t) = exp( - sum_{tau_k <= t} dLambda_k * exp(f_i(tau_k)) )
    """
    X = torch.from_numpy(x.astype(np.float32)).to(device)
    N = X.size(0)
    K = len(uniq_train_times)
    M = len(time_grid)

    if K == 0:
        return np.ones((N, M), dtype=np.float64)

    # Compute increments per subject at each uniq time
    H_inc = np.zeros((N, K), dtype=np.float64)
    for k, tau in enumerate(uniq_train_times):
        tau_t = torch.full((N,), float(tau), device=device)
        f_i = model(X, tau_t)  # (N,)
        H_inc[:, k] = np.exp(f_i.detach().cpu().numpy()) * float(dLambda[k])

    cumH = np.cumsum(H_inc, axis=1)  # (N, K)

    # Map each grid time to the last uniq time index <= grid time
    idx = np.searchsorted(uniq_train_times, time_grid, side="right") - 1
    idx = np.clip(idx, -1, K - 1)

    S = np.ones((N, M), dtype=np.float64)
    valid = idx >= 0
    # For each grid position m, use cumH at index idx[m]
    S[:, valid] = np.exp(-cumH[:, idx[valid]])
    return S


# -----------------------------
# sksurv metrics: C-index + IBS from survival curves
# -----------------------------
def c_index_from_survival_sksurv(
    time: np.ndarray,
    event: np.ndarray,
    surv: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    """
    Convert survival curves to a scalar risk (negative expected time) then compute C-index using sksurv.
    """
    from sksurv.metrics import concordance_index_censored

    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(bool)

    # expected time = integral S(t) dt
    et = np.trapezoid(surv, time_grid, axis=1)
    risk = -et  # shorter expected time => higher risk

    c, *_ = concordance_index_censored(event, time, risk)
    return float(c)


def ibs_from_survival_sksurv(
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    surv_test: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    """
    IBS using sksurv.integrated_brier_score.
    """
    from sksurv.util import Surv
    from sksurv.metrics import integrated_brier_score

    y_tr = Surv.from_arrays(np.asarray(event_train).astype(bool), np.asarray(time_train).astype(float))
    y_te = Surv.from_arrays(np.asarray(event_test).astype(bool), np.asarray(time_test).astype(float))

    # Ensure time_grid within test range (optional, keeps integrated_brier_score well-behaved)
    tmin, tmax = float(np.min(time_test)), float(np.max(time_test))
    mask = (time_grid >= tmin) & (time_grid <= tmax)
    tg = time_grid[mask]
    st = surv_test[:, mask]

    if tg.size < 2:
        # Fallback: use full grid
        tg = time_grid
        st = surv_test

    return float(integrated_brier_score(y_tr, y_te, st, tg))


# -----------------------------
# End-to-end evaluation (DataFrames in, sksurv metrics out)
# -----------------------------
def evaluate_coxtime_with_sksurv(
    model: CoxTimeNet,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
    device: str,
    n_times: int = 200,
) -> dict:
    # Extract arrays from DataFrames (covariates already preprocessed)
    Xtr = train_df[feature_cols].to_numpy(dtype=np.float32)
    Ttr = train_df[time_col].to_numpy(dtype=np.float32)
    Etr = train_df[event_col].to_numpy(dtype=np.int64)

    Xte = test_df[feature_cols].to_numpy(dtype=np.float32)
    Tte = test_df[time_col].to_numpy(dtype=np.float32)
    Ete = test_df[event_col].to_numpy(dtype=np.int64)

    # Baseline increments from train
    uniq, dLam = coxtime_baseline_hazard(model, Xtr, Ttr, Etr, device)

    # Unified time grid and survival curves for test
    grid = create_time_grid(Tte, n_points=n_times)
    S_te = coxtime_survival(model, Xte, grid, uniq, dLam, device)

    # sksurv metrics
    cindex = c_index_from_survival_sksurv(Tte, Ete, S_te, grid)
    ibs = ibs_from_survival_sksurv(Ttr, Etr, Tte, Ete, S_te, grid)

    return {
        "time_grid": grid,
        "surv_test": S_te,
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
    cfg = TrainConfig(batch_size=512, epochs=100, patience=10)

    model, info = train_coxtime_from_dfs(
        train_df=train_df,
        val_df=val_df,
        feature_cols=covar_cols,
        time_col="time",
        event_col="event",
        cfg=cfg,
    )

    results = evaluate_coxtime_with_sksurv(
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=info["feature_cols"],
        time_col=info["time_col"],
        event_col=info["event_col"],
        device=info["config"].device,
        n_times=200,
    )

    print("Test C-index (sksurv):", results["c_index_test"])
    print("Test IBS (sksurv):", results["ibs_test"])
