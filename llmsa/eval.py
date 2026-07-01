"""Multi-seed evaluation harness for baselines and LLM-SA.

Every model implements a small protocol:

    fit(bundle)                        -> None
    predict_risk(bundle, split_name)   -> (n,) risk score, higher = more risk
    predict_surv(bundle, split_name)   -> (n, len(time_grid)) survival probs

``evaluate_model_on_dataset`` runs ``n_seeds`` independent fit-evaluate cycles
under IPCW C-index and IPCW IBS and returns per-seed metrics.
"""

from __future__ import annotations

import gc
import time as _time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Protocol

import numpy as np

from .data import ALL_DATASETS, DatasetBundle, load_dataset, load_dataset_v4
from .metrics import ipcw_cindex, ipcw_ibs


class SurvivalModel(Protocol):
    name: str

    def fit(self, bundle: DatasetBundle) -> None: ...
    def predict_risk(self, bundle: DatasetBundle, split_name: str) -> np.ndarray: ...
    def predict_surv(self, bundle: DatasetBundle, split_name: str) -> np.ndarray: ...


@dataclass
class SeedResult:
    seed: int
    cindex: float
    ibs: float
    fit_seconds: float
    note: str = ""


@dataclass
class DatasetResult:
    dataset: str
    seeds: List[SeedResult] = field(default_factory=list)

    @property
    def cindex_mean(self) -> float:
        return float(np.mean([s.cindex for s in self.seeds]))

    @property
    def cindex_std(self) -> float:
        return float(np.std([s.cindex for s in self.seeds]))

    @property
    def ibs_mean(self) -> float:
        return float(np.mean([s.ibs for s in self.seeds]))

    @property
    def ibs_std(self) -> float:
        return float(np.std([s.ibs for s in self.seeds]))


def evaluate_model_on_dataset(
    model_factory: Callable[[int], SurvivalModel],
    dataset_name: str,
    seeds: Iterable[int] = range(10),
    verbose: bool = True,
    use_v4: bool = False,
    full_support: bool = False,
) -> DatasetResult:
    """Fit-evaluate ``model_factory(seed)`` on ``dataset_name`` for each seed.

    ``use_v4=True`` loads the LLM-augmented bundle (covariates + N_i + narrative
    embedding); otherwise the structured-only bundle is used (for baselines).
    """
    seeds = list(seeds)
    out = DatasetResult(dataset=dataset_name)
    for seed in seeds:
        if use_v4:
            bundle = load_dataset_v4(dataset_name, seed=seed, full_support=full_support)
        else:
            bundle = load_dataset(dataset_name, seed=seed)
        t0 = _time.time()
        note = ""
        try:
            try:
                model = model_factory(seed)
            except TypeError:
                model = model_factory()
            model.fit(bundle)
            risk = np.asarray(model.predict_risk(bundle, "test"), dtype=float)
            surv = np.asarray(model.predict_surv(bundle, "test"), dtype=float)
            grid = bundle.time_grid
            if surv.ndim != 2 or surv.shape[0] != len(bundle.test):
                raise ValueError(f"bad surv shape {surv.shape}")
            if surv.shape[1] != len(grid):
                raise ValueError(f"surv has {surv.shape[1]} cols but grid has {len(grid)}")
            c = ipcw_cindex(
                bundle.train.time, bundle.train.event,
                bundle.test.time, bundle.test.event, risk,
            )
            b = ipcw_ibs(
                bundle.train.time, bundle.train.event,
                bundle.test.time, bundle.test.event, surv, grid,
            )
        except Exception as exc:
            note = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[{dataset_name} seed={seed}] FAILED  {note}", flush=True)
                traceback.print_exc()
            c, b = float("nan"), float("nan")
        fit_s = _time.time() - t0
        out.seeds.append(SeedResult(seed=seed, cindex=c, ibs=b, fit_seconds=fit_s, note=note))
        if verbose:
            print(f"[{dataset_name} seed={seed}] C={c:.4f}  IBS={b:.4f}  ({fit_s:.1f}s) {note}",
                  flush=True)
        del bundle
        gc.collect()
    return out


def format_summary_block(
    experiment_id: str,
    results: Dict[str, DatasetResult],
    peak_memory_mb: float = float("nan"),
    runtime_seconds: float = float("nan"),
) -> str:
    lines = ["---", f"experiment:              {experiment_id}"]
    for ds in ("metabric", "gbsg", "support"):
        r = results.get(ds)
        if r is None:
            cm = cs = bm = bs = float("nan")
        else:
            cm, cs, bm, bs = r.cindex_mean, r.cindex_std, r.ibs_mean, r.ibs_std
        lines += [
            f"{ds}_cindex_mean:    {cm:.4f}",
            f"{ds}_cindex_std:     {cs:.4f}",
            f"{ds}_ibs_mean:       {bm:.4f}",
            f"{ds}_ibs_std:        {bs:.4f}",
        ]
    lines += [
        f"peak_memory_mb:          {peak_memory_mb:.1f}",
        f"runtime_seconds:         {runtime_seconds:.1f}",
    ]
    return "\n".join(lines)
