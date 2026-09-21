"""Component ablation of the LLM-derived representations (manuscript Sec 4.3).

The stacking ensemble is held FIXED (CoxPH + RSF + DeepSurv, simplex
meta-learner, identical hyperparameters and identical per-seed splits); only
the LLM-derived inputs vary. This isolates the contribution of the LLM
representations from the contribution of stacking itself:

    cell  X_i  N_i  E_i   meaning
    A      +    -    -    structured covariates only (the no-LLM control)
    B      +    +    -    + numeric prognostic estimates
    C      +    -    +    + narrative embedding
    D      +    +    +    full LLM-SA

Usage:
    python -m llmsa.run_ablation --seeds 30 --datasets metabric,gbsg,support \
        --full-support --feature-version v5
"""

from __future__ import annotations

import argparse
import time

from .eval import evaluate_model_on_dataset, format_summary_block
from .model import LLMSAStacking, TrainConfig
from .train_eval import per_cohort_cfg

CELLS = {
    "A": dict(use_numerics=False, use_text=False),
    "B": dict(use_numerics=True,  use_text=False),
    "C": dict(use_numerics=False, use_text=True),
    "D": dict(use_numerics=True,  use_text=True),
}


def main(seeds, datasets, lam, full_support, feature_version, cells, dev_holdout=True,
         text_pca_dim=None, train_n=None, cohort_n=None, cohort_seed=0, cindex_tau="none"):
    t0 = time.time()
    for cell in cells:
        spec = CELLS[cell]
        results = {}
        tag = (f", d_pca={text_pca_dim}" if text_pca_dim else "") + \
              (f", train_n={train_n}" if train_n else "") + \
              (f", cohort_n={cohort_n} draw={cohort_seed}" if cohort_n else "") + \
              (f", cindex_tau={cindex_tau}" if cindex_tau != "none" else "")
        print(f"\n@@@@@@@@@@@@ ablation cell {cell} "
              f"(N_i={'+' if spec['use_numerics'] else '-'}, "
              f"E_i={'+' if spec['use_text'] else '-'}, "
              f"features={feature_version}{tag}) @@@@@@@@@@@@", flush=True)
        for ds in datasets:
            base = per_cohort_cfg(lam)[ds]
            cfg = TrainConfig(text_pca_dim=text_pca_dim or base.text_pca_dim,
                              batch_size=base.batch_size,
                              meta_prior_lam=base.meta_prior_lam,
                              use_text=spec["use_text"])

            def factory(seed, _cfg=cfg):
                return LLMSAStacking(_cfg, seed=seed)

            print(f"\n############ {ds} (cell {cell}) ############", flush=True)
            results[ds] = evaluate_model_on_dataset(
                factory, ds, seeds=seeds, verbose=True, use_v4=True,
                full_support=full_support, dev_holdout=dev_holdout,
                feature_version=feature_version,
                use_numerics=spec["use_numerics"],
                train_subsample_n=train_n,
                cohort_subsample=(cohort_n, cohort_seed) if cohort_n else None,
                cindex_tau=cindex_tau,
            )
        print()
        suffix = (f"_d{text_pca_dim}" if text_pca_dim else "") + (f"_n{train_n}" if train_n else "") + \
                 (f"_cohort{cohort_n}_draw{cohort_seed}" if cohort_n else "")
        print(format_summary_block(f"ablation_{cell}_{feature_version}{suffix}", results),
              flush=True)
    print(f"\nablation finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--datasets", type=str, default="metabric,gbsg,support")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--full-support", action="store_true")
    p.add_argument("--feature-version", default="v5",
                   help="artefact tag of the LLM features, e.g. v4, v5, v5nobrief, "
                        "or a --tag-suffix variant such as v5mini")
    p.add_argument("--cells", type=str, default="A,B,C,D")
    p.add_argument("--legacy-splits", action="store_true")
    p.add_argument("--text-pca-dim", type=int, default=None,
                   help="override the per-cohort embedding PCA dimension "
                        "(d^PCA sensitivity analysis)")
    p.add_argument("--train-n", type=int, default=None,
                   help="subsample the TRAINING fold to this many patients "
                        "(learning-curve experiments); val/test untouched")
    p.add_argument("--cohort-n", type=int, default=None,
                   help="draw a random event-stratified cohort of this many patients "
                        "from the full frame (train/val/test all shrink); small-data experiments")
    p.add_argument("--cohort-seed", type=int, default=0,
                   help="seed of the random cohort draw (independent of the split seeds)")
    p.add_argument("--cindex-tau", choices=["none", "grid"], default="none",
                   help="Uno C truncation: 'grid' = evaluation-grid maximum (IBS horizon); "
                        "'none' = as-submitted untruncated metric")
    args = p.parse_args()
    main(seeds=list(range(args.seeds)), datasets=args.datasets.split(","),
         lam=args.lam, full_support=args.full_support,
         feature_version=args.feature_version, cells=args.cells.split(","),
         dev_holdout=not args.legacy_splits, text_pca_dim=args.text_pca_dim,
         train_n=args.train_n, cohort_n=args.cohort_n, cohort_seed=args.cohort_seed,
         cindex_tau=args.cindex_tau)
