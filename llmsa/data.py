"""Canonical data loaders for LLM-SA.

Every model imports its data through this module so that all comparisons use
the exact same per-seed train/val/test splits.

Two loaders are exposed:

* ``load_dataset``    -- structured covariates only (used by the baselines).
* ``load_dataset_v4`` -- structured covariates + the three LLM numeric
  prognostic fields ``N_i = (p_5y, r_2y, c)`` + the 1536-d narrative
  embedding ``E_i``. This is the input consumed by the LLM-SA stacking
  ensemble.

Data files are expected under ``<repo>/data`` (set ``LLMSA_REPO_ROOT`` to
point elsewhere). They are produced by ``text_generation/generate_features.py``
and are not tracked in git.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Repo root; override with LLMSA_REPO_ROOT so `data/` can live outside the repo.
REPO_ROOT = os.environ.get("LLMSA_REPO_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)


@dataclass
class SurvivalSplit:
    """One split (train / val / test) of one dataset for one seed."""

    X_num: np.ndarray  # (n, p_num) float32, standardized using train statistics
    X_cat: np.ndarray  # (n, p_cat) int64, integer-encoded categorical features
    time: np.ndarray  # (n,) float32
    event: np.ndarray  # (n,) int32 (0/1)
    text: np.ndarray  # (n,) object array of strings (may be empty)
    text_embedding: Optional[np.ndarray]  # (n, d_text) float32 or None

    def __len__(self) -> int:
        return len(self.time)


@dataclass
class DatasetBundle:
    name: str
    train: SurvivalSplit
    val: SurvivalSplit
    test: SurvivalSplit
    cat_cardinalities: list  # [|cat_1|, |cat_2|, ...]
    num_feature_names: list
    cat_feature_names: list
    text_dim: int  # 0 when no embedding
    time_grid: np.ndarray  # canonical grid for IPCW IBS evaluation


# -----------------------------------------------------------------------------
# Dataset definitions (feature splits follow the manuscript)
# -----------------------------------------------------------------------------

GBSG_CAT = ["meno", "grade", "hormon"]
GBSG_NUM = ["age", "size", "nodes", "pgr", "er"]

METABRIC_CAT = ["hormone_treatment", "radiotherapy", "chemotherapy", "ER_positive"]
METABRIC_NUM = ["MKI67", "EGFR", "PGR", "ERBB2", "age"]

SUPPORT_CAT = ["sex", "dzclass"]
SUPPORT_NUM = [
    "age", "num.co", "meanbp", "wblc", "hrt", "resp",
    "temp", "pafi", "alb", "bili", "crea", "sod",
]

# SUPPORT is large (9105 patients); by default we evaluate on a fixed
# 1500-patient event-stratified subsample so every experiment sees the same
# cohort. Set the seed once and never change it -- comparability depends on it.
SUPPORT_SUBSAMPLE_SIZE = 1500
SUPPORT_SUBSAMPLE_SEED = 20260101

ALL_DATASETS = ("metabric", "gbsg", "support")

# The three LLM numeric prognostic fields appended to the structured covariates.
V4_NUMS = ("llm_5yr_surv_v4", "llm_2yr_event_risk_v4", "llm_confidence_v4")


def _csv_path(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


def _cols(name: str):
    name = name.lower()
    if name == "gbsg":
        return list(GBSG_CAT), list(GBSG_NUM)
    if name == "metabric":
        return list(METABRIC_CAT), list(METABRIC_NUM)
    if name == "support":
        return list(SUPPORT_CAT), list(SUPPORT_NUM)
    raise ValueError(f"unknown dataset name: {name!r}")


def _read_cohort_csv(name: str) -> pd.DataFrame:
    """Load the per-cohort CSV holding covariates + time/event (+ text/emb)."""
    name = name.lower()
    if name == "gbsg":
        return pd.read_csv(_csv_path("data", "1_gbsg_all_new_with_embeddings.csv"))
    if name == "metabric":
        return pd.read_csv(_csv_path("data", "5_metabric_all_new_with_embeddings.csv"))
    if name == "support":
        tr = pd.read_csv(_csv_path("data", "2_support_train_new_with_embeddings_preprocessed_DL.csv"))
        te = pd.read_csv(_csv_path("data", "2_support_test_new_with_embeddings_preprocessed_DL.csv"))
        return pd.concat([tr, te], axis=0, ignore_index=True)
    raise ValueError(name)


def _parse_embedding_cell(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=np.float32)
    if isinstance(cell, str):
        try:
            return np.asarray(json.loads(cell), dtype=np.float32)
        except Exception:
            return None
    return None


def _support_subsample(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SUPPORT_SUBSAMPLE_SEED)
    ev1 = df.index[df["event"] == 1].to_numpy()
    ev0 = df.index[df["event"] == 0].to_numpy()
    n_ev1 = int(round(SUPPORT_SUBSAMPLE_SIZE * len(ev1) / len(df)))
    n_ev0 = SUPPORT_SUBSAMPLE_SIZE - n_ev1
    pick_ev1 = rng.choice(ev1, size=n_ev1, replace=False)
    pick_ev0 = rng.choice(ev0, size=n_ev0, replace=False)
    keep_idx = np.sort(np.concatenate([pick_ev1, pick_ev0]))
    return df.loc[keep_idx].reset_index(drop=True)


def _make_time_grid(times: np.ndarray, events: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Quantile time grid over the training-fold event times."""
    ev_times = times[events == 1]
    if len(ev_times) < n_bins:
        ev_times = times
    qs = np.linspace(0.05, 0.95, n_bins)
    grid = np.quantile(ev_times, qs)
    return np.unique(grid).astype(np.float64)


# -----------------------------------------------------------------------------
# Structured-only loader (baselines)
# -----------------------------------------------------------------------------

def _build_split(df, cat_cols, num_cols, cat_cardinalities, scaler, has_text) -> SurvivalSplit:
    X_cat = (df[cat_cols].astype(int).to_numpy(dtype=np.int64)
             if cat_cols else np.zeros((len(df), 0), dtype=np.int64))
    for j, k in enumerate(cat_cardinalities):
        X_cat[:, j] = np.clip(X_cat[:, j], 0, k - 1)

    X_num_raw = (df[num_cols].to_numpy(dtype=np.float32)
                 if num_cols else np.zeros((len(df), 0), dtype=np.float32))
    X_num = scaler.transform(X_num_raw).astype(np.float32) if num_cols else X_num_raw

    if has_text:
        embs = [_parse_embedding_cell(c) for c in df["embeddings"].tolist()]
        d = max((e.shape[0] for e in embs if e is not None), default=0)
        out = np.zeros((len(embs), d), dtype=np.float32)
        for i, e in enumerate(embs):
            if e is not None and e.shape[0] == d:
                out[i] = e
        text_embedding = out
        text = df["generated_texts"].fillna("").astype(str).to_numpy()
    else:
        text_embedding = None
        text = np.array([""] * len(df), dtype=object)

    return SurvivalSplit(
        X_num=X_num, X_cat=X_cat,
        time=df["time"].to_numpy(dtype=np.float32),
        event=df["event"].to_numpy(dtype=np.int32),
        text=text, text_embedding=text_embedding,
    )


def load_dataset(name: str, seed: int) -> DatasetBundle:
    """Structured-covariate-only loader with a reproducible 70/10/20 split."""
    name = name.lower()
    cat_cols, num_cols = _cols(name)

    df = _read_cohort_csv(name)
    keep = cat_cols + num_cols + ["time", "event", "generated_texts", "embeddings"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int).clip(0, 1)
    if name == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE:
        df = _support_subsample(df)

    has_text = "embeddings" in df.columns

    df_tmp, df_test = train_test_split(df, test_size=0.2, stratify=df["event"], random_state=seed)
    df_train, df_val = train_test_split(
        df_tmp, test_size=0.125, stratify=df_tmp["event"], random_state=seed)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    cat_cardinalities = [int(df[c].astype(int).max()) + 1 for c in cat_cols]
    scaler = StandardScaler()
    if num_cols:
        scaler.fit(df_train[num_cols].to_numpy(dtype=np.float32))

    train = _build_split(df_train, cat_cols, num_cols, cat_cardinalities, scaler, has_text)
    val = _build_split(df_val, cat_cols, num_cols, cat_cardinalities, scaler, has_text)
    test = _build_split(df_test, cat_cols, num_cols, cat_cardinalities, scaler, has_text)

    time_grid = _make_time_grid(train.time, train.event, n_bins=50)
    text_dim = train.text_embedding.shape[1] if train.text_embedding is not None else 0

    return DatasetBundle(
        name=name, train=train, val=val, test=test,
        cat_cardinalities=cat_cardinalities,
        num_feature_names=num_cols, cat_feature_names=cat_cols,
        text_dim=text_dim, time_grid=time_grid,
    )


# -----------------------------------------------------------------------------
# LLM-SA loader: covariates + N_i numerics + narrative embedding E_i
# -----------------------------------------------------------------------------

def load_dataset_v4(name: str, seed: int, full_support: bool = False) -> DatasetBundle:
    """Load covariates augmented with the LLM outputs.

    Reads two artefacts produced by ``text_generation/generate_features.py``:

    * ``data/v4_structured_<name>.csv`` -- the three LLM numeric fields
      ``N_i = (llm_5yr_surv_v4, llm_2yr_event_risk_v4, llm_confidence_v4)``,
      keyed by ``patient_idx``.
    * ``data/embeddings_v4_<name>.npy`` -- the (n, 1536) narrative embeddings
      ``E_i``, in the same patient order as the structured CSV.

    The numerics are concatenated to the standardized covariates; the embedding
    is attached to each split for the DeepSurv base learner.
    """
    name = name.lower()
    cat_cols, num_cols = _cols(name)

    df = _read_cohort_csv(name)
    df = df[cat_cols + num_cols + ["time", "event"]]
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int).clip(0, 1)
    df["_patient_idx"] = df.index.copy()

    if name == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE and not full_support:
        df = _support_subsample(df)

    struct = pd.read_csv(_csv_path("data", f"v4_structured_{name}.csv"))
    text_emb = np.load(_csv_path("data", f"embeddings_v4_{name}.npy"))
    struct_idx_to_row = {int(pi): i for i, pi in enumerate(struct["patient_idx"].tolist())}
    struct_map = struct.set_index("patient_idx").to_dict("index")

    n = len(df)
    new_num = {c: np.zeros(n, dtype=np.float32) for c in V4_NUMS}
    view_rows = []
    for i, pi in enumerate(df["_patient_idx"].tolist()):
        rec = struct_map.get(int(pi))
        if rec is None:
            view_rows.append(np.zeros(text_emb.shape[1], dtype=np.float32))
            for c in V4_NUMS:
                new_num[c][i] = 0.5
        else:
            for c in V4_NUMS:
                new_num[c][i] = float(rec.get(c, 0.5))
            view_rows.append(text_emb[struct_idx_to_row[int(pi)]])
    text_emb_aligned = np.stack(view_rows, axis=0).astype(np.float32)
    for c, v in new_num.items():
        df[c] = v
    df = df.drop(columns=["_patient_idx"])

    all_num_cols = num_cols + list(V4_NUMS)
    cat_cardinalities = [int(df[c].astype(int).max()) + 1 for c in cat_cols]

    df_pos = df.copy(); df_pos["_pos"] = np.arange(len(df_pos))
    df_tmp, df_test = train_test_split(df_pos, test_size=0.2, stratify=df_pos["event"], random_state=seed)
    df_train, df_val = train_test_split(
        df_tmp, test_size=0.125, stratify=df_tmp["event"], random_state=seed)

    def _with_view(df_sub):
        positions = df_sub["_pos"].to_numpy()
        return df_sub.drop(columns=["_pos"]).reset_index(drop=True), text_emb_aligned[positions]

    df_train, ve_train = _with_view(df_train)
    df_val, ve_val = _with_view(df_val)
    df_test, ve_test = _with_view(df_test)

    scaler = StandardScaler()
    if all_num_cols:
        scaler.fit(df_train[all_num_cols].to_numpy(dtype=np.float32))

    def _build(df_sub, view):
        X_cat = (np.ascontiguousarray(df_sub[cat_cols].astype(int).to_numpy(dtype=np.int64))
                 if cat_cols else np.zeros((len(df_sub), 0), dtype=np.int64))
        for j, k in enumerate(cat_cardinalities):
            X_cat[:, j] = np.clip(X_cat[:, j], 0, k - 1)
        X_num_raw = (df_sub[all_num_cols].to_numpy(dtype=np.float32)
                     if all_num_cols else np.zeros((len(df_sub), 0), dtype=np.float32))
        X_num = scaler.transform(X_num_raw).astype(np.float32) if all_num_cols else X_num_raw
        return SurvivalSplit(
            X_num=X_num, X_cat=X_cat,
            time=df_sub["time"].to_numpy(dtype=np.float32),
            event=df_sub["event"].to_numpy(dtype=np.int32),
            text=np.array([""] * len(df_sub), dtype=object),
            text_embedding=view.astype(np.float32),
        )

    train = _build(df_train, ve_train)
    val = _build(df_val, ve_val)
    test = _build(df_test, ve_test)
    time_grid = _make_time_grid(train.time, train.event, n_bins=50)

    return DatasetBundle(
        name=name, train=train, val=val, test=test,
        cat_cardinalities=cat_cardinalities,
        num_feature_names=all_num_cols, cat_feature_names=cat_cols,
        text_dim=text_emb.shape[1], time_grid=time_grid,
    )
