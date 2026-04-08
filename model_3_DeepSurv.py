"""
DeepSurv (PyTorch) — DataFrame-only — C-index + IBS computed with scikit-survival (sksurv)

Inputs:
  train_df, test_df  : pandas.DataFrame
Each df must contain:
  - covariate columns (already preprocessed; numeric)
  - time column (default: 'time')
  - event column (default: 'event', 1=event, 0=censored)

Metrics:
  - C-index: sksurv.metrics.concordance_index_censored
  - IBS: sksurv.metrics.integrated_brier_score (via CoxPH fitted on DeepSurv risk)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
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
def create_time_grid(
    times: np.ndarray,
    n_points: int = 200,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Create a monotonically increasing time grid for IBS.
    Uses quantiles to avoid extreme tails dominating.
    """
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    t = t[t > 0]
    if t.size == 0:
        # Fallback
        return np.linspace(0.1, 1.0, n_points)

    t_min = max(np.quantile(t, 0.01), eps)
    t_max = max(np.quantile(t, 0.99), t_min + eps)
    grid = np.linspace(t_min, t_max, n_points)
    return grid


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

    def __getitem__(self, idx):
        return self.x[idx], self.time[idx], self.event[idx]


# -----------------------------
# Model (DeepSurv MLP)
# -----------------------------
class DeepSurv(nn.Module):
    """
    DeepSurv MLP outputs log-risk f(x). Higher => higher risk.
    """

    def __init__(
        self,
        in_features: int,
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
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -----------------------------
# CoxPH loss (negative log partial likelihood) with Breslow ties
# -----------------------------
class CoxPHLoss(nn.Module):
    def forward(self, pred: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(time, descending=True)
        time_sorted = time[order]
        event_sorted = event[order]
        pred_sorted = pred[order]

        exp_pred = torch.exp(pred_sorted)
        log_cumsum = torch.log(torch.cumsum(exp_pred, dim=0))

        event_mask = event_sorted == 1
        times_e = time_sorted[event_mask]
        pred_e = pred_sorted[event_mask]

        if times_e.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        uniq_times, inverse, counts = torch.unique_consecutive(
            times_e, return_inverse=True, return_counts=True
        )

        idx_e = torch.nonzero(event_mask, as_tuple=False).squeeze(1)
        log_denoms_per_event = log_cumsum[idx_e]

        sum_pred_by_tie = torch.zeros(uniq_times.shape[0], device=pred.device)
        sum_logden_by_tie = torch.zeros_like(sum_pred_by_tie)
        sum_pred_by_tie.index_add_(0, inverse, pred_e)
        sum_logden_by_tie.index_add_(0, inverse, log_denoms_per_event)

        loss = -(sum_pred_by_tie - sum_logden_by_tie).sum()
        loss = loss / counts.sum()
        return loss


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 60
    device: str = get_best_device()


# -----------------------------
# Training (DataFrame splits)
# -----------------------------
def train_deepsurv_from_dfs(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    time_col: str = "time",
    event_col: str = "event",
    hidden: Iterable[int] = (512, 256),
    dropout: float = 0.1,
    batch_norm: bool = True,
    cfg: TrainConfig = TrainConfig(),
) -> Tuple[DeepSurv, dict]:
    set_seed(42)

    train_ds = SurvivalDataFrameDataset(train_df, feature_cols, time_col, event_col)
    feature_cols_used = train_ds.feature_cols

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    model = DeepSurv(
        in_features=train_ds.x.shape[1],
        hidden=hidden,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(cfg.device)

    criterion = CoxPHLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for xb, tb, eb in train_dl:
            xb = xb.to(cfg.device)
            tb = tb.to(cfg.device)
            eb = eb.to(cfg.device)

            pred = model(xb)
            loss = criterion(pred, tb, eb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)

        print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.4f}")

    history = {
        "config": cfg,
        "feature_cols": feature_cols_used,
        "time_col": time_col,
        "event_col": event_col,
    }
    return model, history


# -----------------------------
# Risk prediction
# -----------------------------
@torch.no_grad()
def predict_risk_on_df(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: List[str],
    device: str,
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(df[feature_cols].to_numpy(dtype=np.float32)).to(device)
    return model(x).detach().cpu().numpy()


# -----------------------------
# sksurv metrics: C-index + IBS
# -----------------------------
def sksurv_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    """
    C-index via sksurv (concordance_index_censored).
    event must be boolean (True=event).
    risk: higher => higher risk.
    """
    from sksurv.metrics import concordance_index_censored

    e = np.asarray(event).astype(bool)
    t = np.asarray(time).astype(float)
    r = np.asarray(risk).astype(float)

    cindex, *_ = concordance_index_censored(e, t, r)
    return float(cindex)


def sksurv_ibs_via_risk_coxph(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    risk_train: np.ndarray,
    risk_test: np.ndarray,
    time_col: str = "time",
    event_col: str = "event",
    n_times: int = 200,
) -> float:
    """
    IBS via sksurv.integrated_brier_score using the common trick:
      - fit CoxPHSurvivalAnalysis on a single feature 'risk' (DeepSurv output)
      - predict survival functions for test set
      - compute IBS on a time grid
    """
    from sksurv.util import Surv
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import integrated_brier_score

    y_train = Surv.from_arrays(
        event=train_df[event_col].to_numpy().astype(bool),
        time=train_df[time_col].to_numpy().astype(float),
    )
    y_test = Surv.from_arrays(
        event=test_df[event_col].to_numpy().astype(bool),
        time=test_df[time_col].to_numpy().astype(float),
    )

    X_train_risk = pd.DataFrame({"risk": np.asarray(risk_train, dtype=float)})
    X_test_risk = pd.DataFrame({"risk": np.asarray(risk_test, dtype=float)})

    cox = CoxPHSurvivalAnalysis()
    cox.fit(X_train_risk[["risk"]], y_train)

    surv_funcs = cox.predict_survival_function(X_test_risk[["risk"]])

    time_grid = create_time_grid(test_df[time_col].to_numpy(), n_points=max(int(n_times), 3))

    # Ensure grid is within the predicted survival functions' domain
    dom_min, dom_max = surv_funcs[0].domain
    time_grid = time_grid[(time_grid >= dom_min) & (time_grid <= dom_max)]
    if time_grid.size < 2:
        # Fallback: use the domain itself
        time_grid = np.linspace(dom_min, dom_max, max(int(n_times), 3))

    surv_probs = np.vstack([fn(time_grid) for fn in surv_funcs])  # (n_test, n_times)

    ibs = integrated_brier_score(y_train, y_test, surv_probs, time_grid)
    return float(ibs)


# -----------------------------
# End-to-end evaluation
# -----------------------------
def evaluate_deepsurv_with_sksurv(
    model: nn.Module,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    time_col: str,
    event_col: str,
    device: str,
    n_times: int = 200,
) -> dict:
    risk_train = predict_risk_on_df(model, train_df, feature_cols, device)
    risk_test = predict_risk_on_df(model, test_df, feature_cols, device)

    cindex_test = sksurv_c_index(
        time=test_df[time_col].to_numpy(),
        event=test_df[event_col].to_numpy(),
        risk=risk_test,
    )

    ibs_test = sksurv_ibs_via_risk_coxph(
        train_df=train_df,
        test_df=test_df,
        risk_train=risk_train,
        risk_test=risk_test,
        time_col=time_col,
        event_col=event_col,
        n_times=n_times,
    )

    return {
        "c_index_test": cindex_test,
        "ibs_test": ibs_test,
    }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # You already have:
    # train_data, test_data (all are pandas.DataFrame)

    # Optionally specify the covariate columns explicitly (recommended)
    # feature_cols = ['meno','grade','hormon','age','size','nodes','pgr','er']
    feature_cols = None  # if None, inferred as all cols except time/event

    data_path = "data/1_gbsg_all.csv"
    categorical_columns = ['meno', 'grade', 'hormon']
    numerical_columns = ['age', 'size', 'nodes', 'pgr', 'er']
    train_df, val_df, test_df, covar_cols = prepare_survival_data(
        data_path, categorical_columns, numerical_columns
    )
    cfg = TrainConfig(batch_size=256, epochs=60)

    model, hist = train_deepsurv_from_dfs(
        train_df=train_df,
        feature_cols=covar_cols,
        time_col="time",
        event_col="event",
        cfg=cfg,
    )

    feature_cols_used = hist["feature_cols"]
    results = evaluate_deepsurv_with_sksurv(
        model=model,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols_used,
        time_col=hist["time_col"],
        event_col=hist["event_col"],
        device=hist["config"].device,
        n_times=200,
    )

    print("Test C-index (sksurv):", results["c_index_test"])
    print("Test IBS (sksurv):", results["ibs_test"])
