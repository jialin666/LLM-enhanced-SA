"""Validation-fold tuning of the LLM-SA PCA dimension and batch size
(same protocol as tune_baselines.py).

Everything else is held at the Table 2 configuration (per_cohort_cfg). For the
full framework (configuration D) a grid of d_PCA x batch size is scored by the
IPCW C-index on the validation fold over --tune-seeds seeds (test folds never
used, exactly as for the baselines); the winner is then evaluated on the
30-split protocol of Table 2 (dev hold-out splits, C-index truncated at the
evaluation grid), so its test results are directly comparable with Table 2.

Usage:
    python -m llmsa.tune_llmsa_pca_batch --dataset gbsg --feature-version v5fixmini
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

from .data import load_dataset_v4
from .eval import evaluate_model_on_dataset
from .metrics import ipcw_cindex
from .model import LLMSAStacking, TrainConfig
from .train_eval import per_cohort_cfg

PCA_GRID = (16, 32, 64, 128, 256)
BATCH_GRID = (32, 64, 128)


def make_cfg(ds, lam, pca, batch):
    base = per_cohort_cfg(lam)[ds]
    return TrainConfig(text_pca_dim=pca, batch_size=batch,
                       meta_prior_lam=base.meta_prior_lam, use_text=True)


def val_score(cfg, ds, seed, fv, full_support=False):
    b = load_dataset_v4(ds, seed=seed, feature_version=fv, use_numerics=True,
                        full_support=full_support)
    m = LLMSAStacking(cfg, seed=seed)
    m.fit(b)
    r = np.asarray(m.predict_risk(b, "val"), dtype=float)
    return ipcw_cindex(b.train.time, b.train.event, b.val.time, b.val.event, r)


def main(ds, fv, lam, tune_seeds, seeds, out, full_support=False):
    t0 = time.time()
    base = per_cohort_cfg(lam)[ds]
    print(f"######## {ds}: Table 2 setting d_PCA={base.text_pca_dim}, "
          f"batch={base.batch_size}; features={fv} ########", flush=True)
    grid = {}
    for pca, batch in itertools.product(PCA_GRID, BATCH_GRID):
        cfg = make_cfg(ds, lam, pca, batch)
        s = [val_score(cfg, ds, k, fv, full_support) for k in range(tune_seeds)]
        grid[(pca, batch)] = float(np.mean(s))
        print(f"  d_PCA={pca:<4} batch={batch:<4} val C = {grid[(pca, batch)]:.4f}", flush=True)
    best = max(grid, key=grid.get)
    print(f"  -> selected d_PCA={best[0]}, batch={best[1]} (val C = {grid[best]:.4f}); "
          f"Table 2 setting val C = {grid.get((base.text_pca_dim, base.batch_size), float('nan')):.4f}",
          flush=True)
    cfg = make_cfg(ds, lam, *best)
    print(f"\n@@@@@@@@@@@@ ablation cell D (N_i=+, E_i=+, features={fv}, tuned "
          f"d_pca={best[0]} batch={best[1]}, cindex_tau=grid) @@@@@@@@@@@@", flush=True)
    print(f"\n############ {ds} (cell D) ############", flush=True)
    res = evaluate_model_on_dataset(lambda seed: LLMSAStacking(cfg, seed=seed), ds,
                                    seeds=list(range(seeds)), verbose=True, use_v4=True,
                                    dev_holdout=True, feature_version=fv, full_support=full_support,
                                    use_numerics=True, cindex_tau="grid")
    print(f"  == {ds}: tuned D test C = {res.cindex_mean:.4f}, IBS = {res.ibs_mean:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    json.dump({"dataset": ds, "features": fv,
               "table2": [base.text_pca_dim, base.batch_size], "selected": list(best),
               "val_grid": {f"{p}_{b}": v for (p, b), v in grid.items()},
               "test_cindex_mean": res.cindex_mean, "test_ibs_mean": res.ibs_mean},
              open(out, "w"), indent=2)
    print("TUNE_PCA_BATCH_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--feature-version", required=True)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--tune-seeds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--out", default=None)
    p.add_argument("--full-support", action="store_true")
    a = p.parse_args()
    out = a.out or os.path.join("analysis", "logs_round2", f"tune_pca_batch_{a.dataset}.json")
    main(a.dataset, a.feature_version, a.lam, a.tune_seeds, a.seeds, out, a.full_support)
