"""Train-vs-test performance gap (reviewer request R1-6).

R1-6 asks for stronger evidence that the reported gains are not overfitting,
given modest cohort sizes and a high-dimensional narrative embedding. This
script fits each ablation cell and reports the IPCW C-index on the TRAINING
fold and on the TEST fold for the same fitted model, so the optimism gap
(train - test) can be compared across cells:

    A = covariates only            (no LLM inputs)
    C = covariates + embedding     (the high-dimensional channel R1-6 worries about)
    D = covariates + numerics + embedding

A larger gap for C/D than for A would indicate the embedding induces
overfitting; a similar gap indicates it does not.

Usage:
    python -m llmsa.run_traintest_gap --seeds 30 --datasets gbsg,metabric,flchain,support \
        --full-support --feature-version v5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .data import load_dataset_v4
from .metrics import ipcw_cindex
from .model import LLMSAStacking, TrainConfig
from .run_ablation import CELLS
from .train_eval import per_cohort_cfg


def main(seeds, datasets, lam, full_support, feature_version, cells):
    t0 = time.time()
    print(f"{'cohort':<10}{'cell':<6}{'train C':<20}{'test C':<20}{'gap (train-test)'}")
    for ds in datasets:
        base = per_cohort_cfg(lam)[ds]
        for cell in cells:
            spec = CELLS[cell]
            tr_scores, te_scores = [], []
            for seed in seeds:
                bundle = load_dataset_v4(ds, seed=seed, full_support=full_support,
                                         feature_version=feature_version,
                                         use_numerics=spec["use_numerics"])
                cfg = TrainConfig(text_pca_dim=base.text_pca_dim, batch_size=base.batch_size,
                                  meta_prior_lam=base.meta_prior_lam,
                                  use_text=spec["use_text"])
                model = LLMSAStacking(cfg, seed=seed)
                try:
                    model.fit(bundle)
                    r_tr = np.asarray(model.predict_risk(bundle, "train"), dtype=float)
                    r_te = np.asarray(model.predict_risk(bundle, "test"), dtype=float)
                    # censoring distribution always estimated from the training fold
                    c_tr = ipcw_cindex(bundle.train.time, bundle.train.event,
                                       bundle.train.time, bundle.train.event, r_tr)
                    c_te = ipcw_cindex(bundle.train.time, bundle.train.event,
                                       bundle.test.time, bundle.test.event, r_te)
                    tr_scores.append(c_tr); te_scores.append(c_te)
                except Exception as exc:
                    print(f"  [{ds} {cell} seed={seed}] FAILED {type(exc).__name__}: {exc}",
                          flush=True)
                del bundle
            if not tr_scores:
                continue
            tr, te = np.array(tr_scores), np.array(te_scores)
            gap = tr - te
            print(f"{ds:<10}{cell:<6}"
                  f"{tr.mean():.4f} ± {tr.std(ddof=1):.4f}   "
                  f"{te.mean():.4f} ± {te.std(ddof=1):.4f}   "
                  f"{gap.mean():+.4f} ± {gap.std(ddof=1):.4f}", flush=True)
    print(f"\nfinished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--datasets", default="gbsg,metabric,flchain,support")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--full-support", action="store_true")
    p.add_argument("--feature-version", default="v5")
    p.add_argument("--cells", default="A,C,D")
    args = p.parse_args()
    main(list(range(args.seeds)), args.datasets.split(","), args.lam,
         args.full_support, args.feature_version, args.cells.split(","))
