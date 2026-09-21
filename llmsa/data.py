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
    # Raw (unstandardized) LLM numeric estimates, always attached when the
    # artefacts exist -- independent of ``use_numerics``, which controls only
    # whether they enter the covariate block of the base learners. This lets a
    # dedicated LLM base learner consume them in a cell where the other
    # learners see no LLM input at all.
    llm_numerics: Optional[np.ndarray] = None  # (n, 3) float32

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

# FLCHAIN (Olmsted County serum free light chain study): classical clinical
# lab covariates, used as a fourth cohort to test when LLM augmentation helps.
# The cause-of-death "chapter" column is deliberately NOT a covariate: it is
# defined only for patients who died, i.e. outcome leakage.
FLCHAIN_CAT = ["sex", "mgus"]
FLCHAIN_NUM = ["age", "sample.yr", "flc.grp", "kappa", "lambda", "creatinine"]

# Reduced-covariate variants used for the covariate-sparsity experiment: the
# same patients and outcomes, with only a subset of covariates exposed to BOTH
# the LLM (via the {FEATURES} block) and the downstream models. This tests the
# manuscript's motivating claim that LLM augmentation helps when covariates are
# sparse.
GBSG3_CAT, GBSG3_NUM = ["grade"], ["age", "nodes"]
GBSG5_CAT, GBSG5_NUM = ["grade", "hormon"], ["age", "size", "nodes"]
METABRIC3_CAT, METABRIC3_NUM = ["ER_positive"], ["age", "MKI67"]
METABRIC5_CAT, METABRIC5_NUM = ["ER_positive", "hormone_treatment"], ["age", "MKI67", "ERBB2"]

# Rotterdam breast cancer cohort (n=2,982): the standard companion/validation
# cohort for GBSG, with the same covariates and the same recurrence-free
# survival endpoint. Added to test whether the LLM benefit observed on GBSG
# replicates on an independent cohort of the same disease.
# TCGA-CDR pan-cancer pilot (n=2,000): 8 raw covariates that expand to ~200
# one-hot columns, dominated by high-cardinality, mostly-rare levels (33 cancer
# types, 94 histological subtypes). Tests whether LLM augmentation helps when
# covariates are semantically rich but statistically sparse.
TCGA_CAT = ["cancer_type", "histology", "stage", "grade", "race", "gender"]
TCGA_NUM = ["age", "dx_year"]

ROTTERDAM_CAT = ["meno", "size", "grade", "hormon"]
ROTTERDAM_NUM = ["age", "nodes", "pgr", "er"]


# Small public survival cohorts (panel testing the low-data regime).
SMALL_COHORTS = {
    "heartfailure": (["anaemia","diabetes","hypertension","smoking","sex"],
                     ["age","cpk","ejection_fraction","platelets","serum_creatinine","serum_sodium"]),
    "wpbc": ([], ["tumor_size","lymph_node_status","radius_worst","texture_worst","perimeter_worst",
                  "area_worst","smoothness_worst","compactness_worst","concavity_worst",
                  "concave_points_worst","symmetry_worst","fractal_dim_worst"]),
    "veteran": (['celltype', 'prior_therapy', 'treatment'], ['age', 'karnofsky', 'months_from_dx']),
    "lung": (['sex_lbl'], ['age', 'ecog', 'karnofsky_phys', 'karnofsky_pat', 'meal_calories', 'weight_loss']),
    "whas500": (['afb', 'av3', 'chf', 'cvd', 'gender', 'miord', 'mitype', 'sho'], ['age', 'bmi', 'diasbp', 'hr', 'sysbp']),
    "larynx": (['stage'], ['age']),
    "aids": (['hemophil', 'ivdrug', 'karnof', 'raceth', 'sex', 'strat2', 'tx', 'txgrp'], ['age', 'cd4', 'priorzdv']),
    "brcamicro": (['er', 'grade'], ['X200726_at', 'X200965_s_at', 'X201068_s_at', 'X201091_s_at', 'X201288_at', 'X201368_at', 'X201663_s_at', 'X201664_at', 'X202239_at', 'X202240_at']),
}

REDUCED_PARENT = {"gbsg3": "gbsg", "gbsg5": "gbsg",
                  "metabric3": "metabric", "metabric5": "metabric"}

ALL_DATASETS = ("metabric", "gbsg", "support", "flchain", "rotterdam", "tcga", "tcgafull")

# Development hold-out used for prompt selection (KM fit, tertile cut-points,
# grading subset). These rows are pinned to the training fold of every per-seed
# protocol split, so no information from prompt selection can reach a
# validation or test set. Fixed fraction and seed -- never change them.
DEV_HOLDOUT_FRAC = 0.15
DEV_HOLDOUT_SEED = 20260903

# The three LLM numeric prognostic fields appended to the structured covariates.
# Feature files are versioned (v4, v5, v5nobrief, ...); column names carry
# the version suffix: llm_5yr_surv_<version>, etc.
def _num_cols_for_version(version: str):
    return tuple(f"llm_{s}_{version}" for s in ("5yr_surv", "2yr_event_risk", "confidence"))


V4_NUMS = _num_cols_for_version("v4")


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
    if name == "flchain":
        return list(FLCHAIN_CAT), list(FLCHAIN_NUM)
    if name == "rotterdam":
        return list(ROTTERDAM_CAT), list(ROTTERDAM_NUM)
    if name in ("tcga", "tcgafull"):
        return list(TCGA_CAT), list(TCGA_NUM)
    if name in SMALL_COHORTS:
        cat, num = SMALL_COHORTS[name]
        return list(cat), list(num)
    if name == "gbsg3":
        return list(GBSG3_CAT), list(GBSG3_NUM)
    if name == "gbsg5":
        return list(GBSG5_CAT), list(GBSG5_NUM)
    if name == "metabric3":
        return list(METABRIC3_CAT), list(METABRIC3_NUM)
    if name == "metabric5":
        return list(METABRIC5_CAT), list(METABRIC5_NUM)
    raise ValueError(f"unknown dataset name: {name!r}")


def _read_cohort_csv(name: str) -> pd.DataFrame:
    """Load the per-cohort CSV holding covariates + time/event (+ text/emb)."""
    name = name.lower()
    name = REDUCED_PARENT.get(name, name)  # reduced variants share the parent CSV
    if name == "gbsg":
        return pd.read_csv(_csv_path("data", "1_gbsg_all_new_with_embeddings.csv"))
    if name == "metabric":
        return pd.read_csv(_csv_path("data", "5_metabric_all_new_with_embeddings.csv"))
    if name == "flchain":
        return pd.read_csv(_csv_path("data", "4_flchain_all.csv"))
    if name == "rotterdam":
        return pd.read_csv(_csv_path("data", "6_rotterdam_all.csv"))
    if name == "tcga":
        return pd.read_csv(_csv_path("data", "7_tcga_pilot.csv"))
    if name == "tcgafull":
        return pd.read_csv(_csv_path("data", "7_tcga_full.csv"))
    if name in ("heartfailure", "wpbc"):
        return pd.read_csv(_csv_path("data", f"9_{name}.csv"))
    if name in SMALL_COHORTS:
        return pd.read_csv(_csv_path("data", f"8_{name}.csv"))
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


def dev_holdout_positions(df: pd.DataFrame) -> np.ndarray:
    """Positions (row indices after preprocessing) of the fixed development hold-out.

    Event-stratified sample of ``DEV_HOLDOUT_FRAC`` of the prepared cohort,
    drawn once with ``DEV_HOLDOUT_SEED``. Positions refer to the dataframe
    after column selection, ``dropna(subset=["time", "event"])``,
    ``reset_index(drop=True)`` and (for SUPPORT) the fixed subsample -- the
    same preparation used by ``load_dataset``, ``load_dataset_v4`` and
    ``text_generation.cohorts.Cohort.load_full``, so the three pipelines agree
    on which patients are development rows.
    """
    rng = np.random.default_rng(DEV_HOLDOUT_SEED)
    n_dev = int(round(DEV_HOLDOUT_FRAC * len(df)))
    ev1 = np.flatnonzero(df["event"].to_numpy() == 1)
    ev0 = np.flatnonzero(df["event"].to_numpy() == 0)
    n_ev1 = int(round(n_dev * len(ev1) / len(df)))
    n_ev0 = n_dev - n_ev1
    pick_ev1 = rng.choice(ev1, size=n_ev1, replace=False)
    pick_ev0 = rng.choice(ev0, size=n_ev0, replace=False)
    return np.sort(np.concatenate([pick_ev1, pick_ev0]))


def _support_subsample_positions(df: pd.DataFrame) -> np.ndarray:
    """Positions (full-frame) of the fixed SUPPORT subsample, without building it."""
    rng = np.random.default_rng(SUPPORT_SUBSAMPLE_SEED)
    ev1 = df.index[df["event"] == 1].to_numpy()
    ev0 = df.index[df["event"] == 0].to_numpy()
    n_ev1 = int(round(SUPPORT_SUBSAMPLE_SIZE * len(ev1) / len(df)))
    n_ev0 = SUPPORT_SUBSAMPLE_SIZE - n_ev1
    pick_ev1 = rng.choice(ev1, size=n_ev1, replace=False)
    pick_ev0 = rng.choice(ev0, size=n_ev0, replace=False)
    return np.sort(np.concatenate([pick_ev1, pick_ev0]))


def support_full_dev_positions(df_full: pd.DataFrame) -> np.ndarray:
    """Full-frame positions of the SUPPORT prompt-selection development rows.

    Prompt selection operates on the fixed 1500-patient subsample
    (``Cohort.load_full``), so its development hold-out lives on the SUBSAMPLE
    frame. When the evaluation cohort is the FULL 9105-patient SUPPORT, the
    rows that must be pinned to the training fold are those same selection
    patients, mapped back to full-frame positions. (Pinning a freshly drawn
    15% of the full frame instead would leave most selection rows eligible
    for test folds -- a partial leak.)
    """
    keep = _support_subsample_positions(df_full)
    df_sub = df_full.iloc[keep].reset_index(drop=True)
    dev_sub = dev_holdout_positions(df_sub)
    return keep[dev_sub]


def _dev_aware_split(df: pd.DataFrame, seed: int, dev_pos: np.ndarray = None):
    """70/10/20 split with the development hold-out pinned to the training fold.

    Test and validation keep their global fractions (20% / 10% of the full
    cohort) but are drawn from non-development rows only; the training fold is
    the remainder plus all development rows. ``dev_pos`` overrides the default
    hold-out (used for full-cohort SUPPORT, where the selection rows live on
    the subsample frame).
    """
    n = len(df)
    dev_pos = dev_holdout_positions(df) if dev_pos is None else dev_pos
    dev_mask = np.zeros(n, dtype=bool)
    dev_mask[dev_pos] = True
    df_dev = df.iloc[dev_pos]
    df_rest = df.iloc[np.flatnonzero(~dev_mask)]

    test_frac = 0.2 * n / len(df_rest)
    df_tmp, df_test = train_test_split(
        df_rest, test_size=test_frac, stratify=df_rest["event"], random_state=seed)
    val_frac = 0.1 * n / len(df_tmp)
    df_tmp, df_val = train_test_split(
        df_tmp, test_size=val_frac, stratify=df_tmp["event"], random_state=seed)
    df_train = pd.concat([df_tmp, df_dev], axis=0)
    return df_train, df_val, df_test


def _cohort_subsample_positions(df: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Event-stratified random subsample of ``n`` rows of the WHOLE cohort.

    Used for the small-data experiments: a random ``n``-patient cohort is drawn
    from the full frame once per ``seed`` (independent of the split seed), and
    the usual 70/10/20 dev-aware split is then applied to that cohort. Unlike
    ``_stratified_subsample`` (training fold only), this shrinks train, val and
    test together, i.e. it simulates having collected only ``n`` patients.
    """
    if n is None or n >= len(df):
        return np.arange(len(df))
    rng = np.random.default_rng(seed)
    ev1 = np.flatnonzero(df["event"].to_numpy() == 1)
    ev0 = np.flatnonzero(df["event"].to_numpy() == 0)
    n1 = int(round(n * len(ev1) / len(df)))
    n0 = n - n1
    pos = np.concatenate([rng.choice(ev1, n1, replace=False), rng.choice(ev0, n0, replace=False)])
    return np.sort(pos)


def _stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Event-stratified subsample of ``n`` rows (learning-curve experiments).

    Applied to the TRAINING fold only; validation and test folds are never
    touched, so curves at different training sizes remain comparable.
    """
    if n >= len(df):
        return df
    keep, _ = train_test_split(df, train_size=n, stratify=df["event"], random_state=seed)
    return keep


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


def load_dataset(name: str, seed: int, dev_holdout: bool = True,
                 full_support: bool = False,
                 cohort_subsample: tuple = None) -> DatasetBundle:
    """Structured-covariate-only loader with a reproducible 70/10/20 split.

    With ``dev_holdout=True`` (default, the corrected protocol) the fixed
    development rows used for prompt selection are pinned to the training
    fold; ``dev_holdout=False`` reproduces the as-submitted legacy splits.
    ``full_support=True`` evaluates on the full 9105-patient SUPPORT cohort
    (default: the fixed 1500-patient subsample).
    """
    name = name.lower()
    cat_cols, num_cols = _cols(name)

    df = _read_cohort_csv(name)
    keep = cat_cols + num_cols + ["time", "event", "generated_texts", "embeddings"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int).clip(0, 1)
    is_full_support = name == "support" and full_support
    if name == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE and not full_support:
        df = _support_subsample(df)

    has_text = "embeddings" in df.columns

    # ``cohort_subsample=(n, draw_seed)``: random event-stratified n-patient
    # cohort drawn from the full frame (small-data experiments).
    if cohort_subsample is not None:
        cs_n, cs_seed = cohort_subsample
        df = df.iloc[_cohort_subsample_positions(df, cs_n, cs_seed)].reset_index(drop=True)
        is_full_support = False

    if dev_holdout:
        dev_pos = support_full_dev_positions(df) if is_full_support else None
        df_train, df_val, df_test = _dev_aware_split(df, seed, dev_pos=dev_pos)
    else:
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

def load_dataset_v4(name: str, seed: int, full_support: bool = False,
                    dev_holdout: bool = True,
                    feature_version: str = "v4",
                    use_numerics: bool = True,
                    train_subsample_n: int = None,
                    subset: tuple = None,
                    cohort_subsample: tuple = None) -> DatasetBundle:
    """Load covariates augmented with the LLM outputs.

    Reads two artefacts produced by ``text_generation/generate_features.py``:

    * ``data/<version>_structured_<name>.csv`` -- the three LLM numeric fields
      ``N_i`` (columns ``llm_5yr_surv_<version>`` etc.), keyed by ``patient_idx``.
    * ``data/embeddings_<version>_<name>.npy`` -- the (n, 1536) narrative
      embeddings ``E_i``, in the same patient order as the structured CSV.

    ``feature_version`` selects the generation: ``v4`` (as-published two-call
    pipeline) or the unified v5 arms (``v5``, ``v5nobrief``, tag suffixes such as ``v5fixmini``).

    The numerics are concatenated to the standardized covariates; the embedding
    is attached to each split for the DeepSurv base learner.
    """
    name = name.lower()
    cat_cols, num_cols = _cols(name)
    version_nums = _num_cols_for_version(feature_version)

    df = _read_cohort_csv(name)
    df = df[cat_cols + num_cols + ["time", "event"]]
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int).clip(0, 1)
    df["_patient_idx"] = df.index.copy()

    if name == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE and not full_support:
        df = _support_subsample(df)

    struct = pd.read_csv(_csv_path("data", f"{feature_version}_structured_{name}.csv"))
    text_emb = np.load(_csv_path("data", f"embeddings_{feature_version}_{name}.npy"))
    struct_idx_to_row = {int(pi): i for i, pi in enumerate(struct["patient_idx"].tolist())}
    struct_map = struct.set_index("patient_idx").to_dict("index")

    n = len(df)
    new_num = {c: np.zeros(n, dtype=np.float32) for c in version_nums}
    view_rows = []
    for i, pi in enumerate(df["_patient_idx"].tolist()):
        rec = struct_map.get(int(pi))
        if rec is None:
            view_rows.append(np.zeros(text_emb.shape[1], dtype=np.float32))
            for c in version_nums:
                new_num[c][i] = 0.5
        else:
            for c in version_nums:
                new_num[c][i] = float(rec.get(c, 0.5))
            view_rows.append(text_emb[struct_idx_to_row[int(pi)]])
    text_emb_aligned = np.stack(view_rows, axis=0).astype(np.float32)
    for c, v in new_num.items():
        df[c] = v
    df = df.drop(columns=["_patient_idx"])

    # ``use_numerics=False`` withholds N_i from every base learner (ablation
    # cells A and C); the narrative embedding is controlled separately by
    # ``TrainConfig.use_text``.
    # ``subset=(column, value)`` restricts the cohort to one stratum (e.g. a single
    # TCGA cancer type) AFTER the LLM features have been aligned by patient index,
    # so each stratum becomes its own small prediction problem.
    if subset is not None:
        col, val = subset
        if col in df.columns:
            series = df[col]
        else:
            # raw label columns (e.g. cancer_type_raw) are dropped during column
            # selection; recover them from the source file, which has the same
            # row order after the identical dropna/reset preprocessing.
            _raw = _read_cohort_csv(name)
            _raw = _raw.dropna(subset=["time", "event"]).reset_index(drop=True)
            _raw = _raw[_raw["time"].astype(float) > 0].reset_index(drop=True) \
                if (_raw["time"].astype(float) <= 0).any() else _raw
            series = _raw[col].iloc[: len(df)].reset_index(drop=True)
        keep = np.flatnonzero((series.astype(str) == str(val)).to_numpy())
        if len(keep) < 40:
            raise ValueError(f"subset {col}={val!r} has only {len(keep)} patients")
        df = df.iloc[keep].reset_index(drop=True)
        text_emb_aligned = text_emb_aligned[keep]

    # ``cohort_subsample=(n, draw_seed)``: random event-stratified n-patient
    # cohort drawn from the full frame AFTER the LLM features are aligned, so
    # the same draw can be used with and without the LLM inputs.
    if cohort_subsample is not None:
        cs_n, cs_seed = cohort_subsample
        keep = _cohort_subsample_positions(df, cs_n, cs_seed)
        df = df.iloc[keep].reset_index(drop=True)
        text_emb_aligned = text_emb_aligned[keep]

    all_num_cols = num_cols + (list(version_nums) if use_numerics else [])
    cat_cardinalities = [int(df[c].astype(int).max()) + 1 for c in cat_cols]

    df_pos = df.copy(); df_pos["_pos"] = np.arange(len(df_pos))
    if dev_holdout:
        is_full_support = (name == "support" and full_support and len(df_pos) > SUPPORT_SUBSAMPLE_SIZE
                           and cohort_subsample is None)
        dev_pos = support_full_dev_positions(df_pos) if is_full_support else None
        df_train, df_val, df_test = _dev_aware_split(df_pos, seed, dev_pos=dev_pos)
    else:
        df_tmp, df_test = train_test_split(df_pos, test_size=0.2, stratify=df_pos["event"], random_state=seed)
        df_train, df_val = train_test_split(
            df_tmp, test_size=0.125, stratify=df_tmp["event"], random_state=seed)

    if train_subsample_n is not None:
        df_train = _stratified_subsample(df_train, train_subsample_n, seed)

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
            llm_numerics=df_sub[list(version_nums)].to_numpy(dtype=np.float32),
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
