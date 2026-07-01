"""IPCW C-index and IPCW IBS -- shared by the baselines and LLM-SA.

These wrap `sksurv` and convert to the (time, event, risk) / (time, event,
surv_curve_at_grid) convention used throughout the codebase.

Every model is scored under exactly the same definition so the comparison is
fair.
"""

from __future__ import annotations

import numpy as np
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv


def _y_structured(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """sksurv structured array (event:bool, time:float)."""
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))


def ipcw_cindex(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_risk: np.ndarray,
) -> float:
    """Uno's IPCW concordance.

    `test_risk` is a per-sample scalar; higher = more risk = shorter survival.
    Training time/event are needed so the IPCW estimator can fit the censoring
    distribution from the training fold (avoids test-leakage).

    sksurv refuses to score when a test EVENT time exceeds max(train_time)
    (the IPCW weight is undefined past that point). We clip such test rows out
    to keep the metric defined; that is the standard practice.
    """
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event)
    test_risk = np.asarray(test_risk, dtype=float)

    train_tmax = float(np.asarray(train_time).max())
    # Drop test rows whose EVENT time exceeds train_tmax; censor them at
    # train_tmax otherwise concordance_index_ipcw raises.
    keep = ~((test_event.astype(bool)) & (test_time > train_tmax))
    if keep.sum() == 0:
        return float("nan")
    test_time = test_time[keep]
    test_event = test_event[keep]
    test_risk = test_risk[keep]

    # For any still-too-large censoring times, clip down so sksurv's IPCW is
    # defined. This affects very few rows in practice.
    eps = 1e-3
    test_time = np.minimum(test_time, train_tmax - eps)

    y_train = _y_structured(train_time, train_event)
    y_test = _y_structured(test_time, test_event)
    c, *_ = concordance_index_ipcw(y_train, y_test, test_risk)
    return float(c)


def ipcw_ibs(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_surv: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    """IPCW Integrated Brier Score on `time_grid`.

    `test_surv` has shape (n_test, len(time_grid)) with predicted survival
    probabilities. `time_grid` must be strictly increasing and lie inside the
    observed follow-up range of the training fold (sksurv requirement).
    """
    grid = np.asarray(time_grid, dtype=float)
    surv = np.asarray(test_surv, dtype=float)
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event)
    if surv.shape[1] != grid.shape[0]:
        raise ValueError(
            f"surv has {surv.shape[1]} columns but time_grid has {grid.shape[0]} points"
        )

    # 1) Clip the grid to lie within the overlap of train + test follow-up.
    train_tmax = float(np.asarray(train_time).max())
    upper = min(train_tmax, float(test_time.max())) - 1e-3
    lower = max(float(np.asarray(train_time).min()), float(test_time.min())) + 1e-3
    keep_grid = (grid > lower) & (grid < upper)
    if not keep_grid.any():
        return float("nan")
    grid_k = grid[keep_grid]
    surv_k = surv[:, keep_grid]

    # 2) Drop test rows whose event time exceeds train_tmax (sksurv requirement).
    keep_rows = ~((test_event.astype(bool)) & (test_time > train_tmax))
    if keep_rows.sum() == 0:
        return float("nan")
    test_time_k = test_time[keep_rows]
    test_event_k = test_event[keep_rows]
    surv_kr = surv_k[keep_rows]
    # Clip any leftover censoring rows so all observation times fit within train range.
    test_time_k = np.minimum(test_time_k, train_tmax - 1e-3)

    y_train = _y_structured(train_time, train_event)
    y_test = _y_structured(test_time_k, test_event_k)
    return float(integrated_brier_score(y_train, y_test, surv_kr, grid_k))


def smoke_test() -> None:
    rng = np.random.default_rng(0)
    n = 200
    time = rng.uniform(1, 100, size=n)
    event = (rng.uniform(size=n) > 0.4).astype(int)
    risk = rng.normal(size=n)
    grid = np.linspace(5, 90, 25)
    surv = np.clip(
        1 - np.outer(rng.uniform(0.3, 0.7, size=n), grid / grid.max()), 1e-3, 1 - 1e-3
    )
    c = ipcw_cindex(time, event, time, event, risk)
    b = ipcw_ibs(time, event, time, event, surv, grid)
    print(f"smoke: C={c:.3f}  IBS={b:.3f}")


if __name__ == "__main__":
    smoke_test()
