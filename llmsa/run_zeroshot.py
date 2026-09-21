"""Zero-shot LLM risk vs. a trained survival model as a function of training size.

The component ablation asks whether LLM features help a model that already has a
full training set. This script asks a different and more favourable question: how
much labelled data does a conventional survival model need before it overtakes the
LLM's zero-shot prognostic estimate, which uses no training labels at all?

For each cohort we score, on the same test folds:
  * ``llm_zeroshot``  -- risk = 1 - p_5y taken directly from the LLM output
                         (no fitting, no training labels),
  * ``llm_zeroshot_2y`` -- risk = r_2y (the short-horizon estimate),
  * ``coxph_n``       -- a CoxPH model fitted on n training patients,
  * ``stack_n``       -- the full stacking ensemble (cell A) fitted on n patients.

Usage:
    python -m llmsa.run_zeroshot --seeds 30 --datasets gbsg,metabric,flchain,support \
        --full-support --feature-version v5
"""

from __future__ import annotations

import argparse

import numpy as np

from .baselines import CoxPH
from .data import load_dataset_v4, _num_cols_for_version
from .metrics import ipcw_cindex
from .model import LLMSAStacking, TrainConfig
from .train_eval import per_cohort_cfg


def main(seeds, datasets, sizes, feature_version, full_support, lam):
    print(f"{'cohort':<10}{'method':<20}{'C-index':<22}{'vs zero-shot'}")
    for ds in datasets:
        num_cols = _num_cols_for_version(feature_version)
        zs, zs2, fitted = [], [], {n: {"cox": [], "stack": []} for n in sizes}
        for seed in seeds:
            b = load_dataset_v4(ds, seed=seed, full_support=full_support,
                                feature_version=feature_version)
            # locate the LLM numeric columns inside the standardized X_num block
            names = b.num_feature_names
            i5 = names.index(num_cols[0])
            i2 = names.index(num_cols[1])
            # higher predicted 5y survival -> lower risk; standardized values keep order
            zs.append(ipcw_cindex(b.train.time, b.train.event, b.test.time, b.test.event,
                                  -b.test.X_num[:, i5]))
            zs2.append(ipcw_cindex(b.train.time, b.train.event, b.test.time, b.test.event,
                                   b.test.X_num[:, i2]))
            for n in sizes:
                bn = load_dataset_v4(ds, seed=seed, full_support=full_support,
                                     feature_version=feature_version,
                                     use_numerics=False, train_subsample_n=n)
                try:
                    m = CoxPH()
                    m.fit(bn)
                    fitted[n]["cox"].append(ipcw_cindex(
                        bn.train.time, bn.train.event, bn.test.time, bn.test.event,
                        np.asarray(m.predict_risk(bn, "test"), dtype=float)))
                except Exception:
                    fitted[n]["cox"].append(np.nan)
                try:
                    cfg = per_cohort_cfg(lam)[ds]
                    st = LLMSAStacking(TrainConfig(text_pca_dim=cfg.text_pca_dim,
                                                   batch_size=cfg.batch_size,
                                                   meta_prior_lam=cfg.meta_prior_lam,
                                                   use_text=False), seed=seed)
                    st.fit(bn)
                    fitted[n]["stack"].append(ipcw_cindex(
                        bn.train.time, bn.train.event, bn.test.time, bn.test.event,
                        np.asarray(st.predict_risk(bn, "test"), dtype=float)))
                except Exception:
                    fitted[n]["stack"].append(np.nan)
                del bn
            del b
        z = float(np.nanmean(zs))
        print(f"{ds:<10}{'LLM zero-shot (5y)':<20}{z:.4f} ± {np.nanstd(zs, ddof=1):.4f}      --")
        print(f"{ds:<10}{'LLM zero-shot (2y)':<20}{np.nanmean(zs2):.4f} ± "
              f"{np.nanstd(zs2, ddof=1):.4f}      {np.nanmean(zs2) - z:+.4f}")
        for n in sizes:
            for key, label in (("cox", f"CoxPH n={n}"), ("stack", f"stack n={n}")):
                v = np.array(fitted[n][key], dtype=float)
                print(f"{ds:<10}{label:<20}{np.nanmean(v):.4f} ± {np.nanstd(v, ddof=1):.4f}      "
                      f"{np.nanmean(v) - z:+.4f}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--datasets", default="gbsg,metabric,flchain,support")
    p.add_argument("--sizes", default="25,50,100,200,400")
    p.add_argument("--feature-version", default="v5")
    p.add_argument("--full-support", action="store_true")
    p.add_argument("--lam", type=float, default=1.0)
    args = p.parse_args()
    main(list(range(args.seeds)), args.datasets.split(","),
         [int(x) for x in args.sizes.split(",")], args.feature_version,
         args.full_support, args.lam)
