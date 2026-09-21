"""Evaluate the structured-covariate-only baselines across the three cohorts.

Usage:
    python -m llmsa.run_baselines --seeds 30 --datasets metabric,gbsg,support \
        --models coxph,rsf,deepsurv,deephit,coxtime --full-support

``--full-support`` must match the setting used for LLM-SA: the comparison is
only meaningful when every model is evaluated on the same cohort and the same
per-seed splits.
"""

from __future__ import annotations

import argparse

from .baselines import BASELINE_FACTORIES
from .eval import evaluate_model_on_dataset, format_summary_block


def main(seeds, datasets, models, full_support=False, dev_holdout=True,
         cohort_n=None, cohort_seed=0, cindex_tau="none"):
    for model_name in models:
        if model_name not in BASELINE_FACTORIES:
            raise SystemExit(f"unknown model {model_name!r}; choices: {list(BASELINE_FACTORIES)}")
        factory = BASELINE_FACTORIES[model_name]
        results = {}
        print(f"\n============ baseline: {model_name} "
              f"(full_support={full_support}, dev_holdout={dev_holdout}) ============",
              flush=True)
        for ds in datasets:
            results[ds] = evaluate_model_on_dataset(
                factory, ds, seeds=seeds, verbose=True,
                full_support=full_support, dev_holdout=dev_holdout,
                cohort_subsample=(cohort_n, cohort_seed) if cohort_n else None,
                cindex_tau=cindex_tau,
            )
        print()
        suffix = f"_cohort{cohort_n}_draw{cohort_seed}" if cohort_n else ""
        print(format_summary_block(f"baseline_{model_name}{suffix}", results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--datasets", type=str, default="metabric,gbsg,support")
    p.add_argument("--models", type=str, default="coxph,rsf,deepsurv,deephit,coxtime")
    p.add_argument("--full-support", action="store_true",
                   help="evaluate SUPPORT on the full 9105-patient cohort "
                        "(must match the LLM-SA setting being compared against)")
    p.add_argument("--legacy-splits", action="store_true",
                   help="reproduce the as-submitted splits (no dev holdout pinning)")
    p.add_argument("--cohort-n", type=int, default=None,
                   help="random event-stratified cohort of this many patients drawn from the full frame")
    p.add_argument("--cohort-seed", type=int, default=0, help="seed of the cohort draw")
    p.add_argument("--cindex-tau", choices=["none", "grid"], default="none",
                   help="Uno C truncation: 'grid' = evaluation-grid maximum (IBS horizon); "
                        "'none' = as-submitted untruncated metric")
    args = p.parse_args()
    main(seeds=list(range(args.seeds)), datasets=args.datasets.split(","),
         models=args.models.split(","), full_support=args.full_support,
         dev_holdout=not args.legacy_splits, cohort_n=args.cohort_n, cohort_seed=args.cohort_seed,
         cindex_tau=args.cindex_tau)
