"""Hyperparameter search for the baseline survival models (reviewer request R2-M5).

Protocol (identical for every baseline, and matched to the search budget used for
LLM-SA's own deep branch):

  1. For each cohort and model, every configuration in the grid is fitted on the
     TRAINING fold and scored by IPCW C-index on the VALIDATION fold, averaged
     over ``--tune-seeds`` seeds. Test folds are never touched during selection.
  2. The winning configuration is then evaluated over the full 30-seed protocol
     and reported; this is the number that belongs in the results table.

Usage:
    python -m llmsa.tune_baselines --datasets metabric,gbsg,support,flchain \
        --models deepsurv,deephit,coxtime,rsf,coxph --tune-seeds 5 --seeds 30 \
        --full-support
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from .baselines import CoxPH, RSF, DeepSurv, DeepHit, CoxTime
from .data import load_dataset
from .eval import evaluate_model_on_dataset, format_summary_block
from .metrics import ipcw_cindex

GRIDS = {
    "deepsurv": {"hidden": [(64, 64), (128, 64), (128, 128)], "lr": [1e-3, 5e-4, 1e-4]},
    "deephit":  {"hidden": [(64, 64), (128, 64), (128, 128)], "lr": [1e-3, 5e-4, 1e-4]},
    "coxtime":  {"hidden": [(64, 64), (128, 64), (128, 128)], "lr": [1e-3, 5e-4, 1e-4]},
    "rsf":      {"n_estimators": [100, 300], "max_depth": [4, 6, 10]},
    "coxph":    {"alpha": [0.001, 0.01, 0.1]},
}
CLASSES = {"deepsurv": DeepSurv, "deephit": DeepHit, "coxtime": CoxTime,
           "rsf": RSF, "coxph": CoxPH}


def _configs(model):
    grid = GRIDS[model]
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _val_score(model_name, cfg, ds, seed, full_support):
    """IPCW C-index on the validation fold (selection metric; test never used)."""
    bundle = load_dataset(ds, seed=seed, full_support=full_support)
    kw = dict(cfg)
    if model_name != "coxph":
        kw["seed"] = seed
    model = CLASSES[model_name](**kw)
    model.fit(bundle)
    risk = np.asarray(model.predict_risk(bundle, "val"), dtype=float)
    return ipcw_cindex(bundle.train.time, bundle.train.event,
                       bundle.val.time, bundle.val.event, risk)


def main(datasets, models, tune_seeds, seeds, full_support):
    t0 = time.time()
    chosen = {}
    for ds in datasets:
        for model_name in models:
            print(f"\n===== tuning {model_name} on {ds} =====", flush=True)
            best, best_score = None, -np.inf
            for cfg in _configs(model_name):
                scores = []
                for s in range(tune_seeds):
                    try:
                        scores.append(_val_score(model_name, cfg, ds, s, full_support))
                    except Exception as exc:
                        print(f"  {cfg} seed {s} FAILED: {type(exc).__name__}", flush=True)
                if not scores:
                    continue
                m = float(np.nanmean(scores))
                print(f"  {cfg}: val C = {m:.4f}", flush=True)
                if m > best_score:
                    best, best_score = cfg, m
            chosen[(ds, model_name)] = (best, best_score)
            print(f"  -> selected {best} (val C = {best_score:.4f})", flush=True)

    print("\n\n########## 30-seed evaluation of the tuned baselines ##########", flush=True)
    summary = {}
    for (ds, model_name), (cfg, _) in chosen.items():
        if cfg is None:
            continue
        def factory(seed, _c=dict(cfg), _m=model_name):
            kw = dict(_c)
            if _m != "coxph":
                kw["seed"] = seed
            return CLASSES[_m](**kw)
        print(f"\n============ tuned baseline: {model_name} on {ds} "
              f"({cfg}) ============", flush=True)
        res = evaluate_model_on_dataset(factory, ds, seeds=list(range(seeds)),
                                        verbose=True, full_support=full_support)
        summary[f"{ds}:{model_name}"] = {
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
            "cindex_mean": res.cindex_mean, "cindex_std": res.cindex_std,
            "ibs_mean": res.ibs_mean, "ibs_std": res.ibs_std,
        }
        print(format_summary_block(f"tuned_{model_name}_{ds}", {ds: res}), flush=True)
    print("\nSELECTED CONFIGURATIONS AND RESULTS")
    print(json.dumps(summary, indent=2))
    print(f"\ntuning finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="metabric,gbsg,support,flchain")
    p.add_argument("--models", default="deepsurv,deephit,coxtime,rsf,coxph")
    p.add_argument("--tune-seeds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--full-support", action="store_true")
    args = p.parse_args()
    main(args.datasets.split(","), args.models.split(","),
         args.tune_seeds, args.seeds, args.full_support)
