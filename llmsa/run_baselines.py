"""Evaluate the structured-covariate-only baselines across the three cohorts.

Usage:
    python -m llmsa.run_baselines --seeds 10 --datasets metabric,gbsg,support \
        --models coxph,rsf,deepsurv,deephit,coxtime
"""

from __future__ import annotations

import argparse

from .baselines import BASELINE_FACTORIES
from .eval import evaluate_model_on_dataset, format_summary_block


def main(seeds, datasets, models):
    for model_name in models:
        if model_name not in BASELINE_FACTORIES:
            raise SystemExit(f"unknown model {model_name!r}; choices: {list(BASELINE_FACTORIES)}")
        factory = BASELINE_FACTORIES[model_name]
        results = {}
        print(f"\n============ baseline: {model_name} ============", flush=True)
        for ds in datasets:
            results[ds] = evaluate_model_on_dataset(factory, ds, seeds=seeds, verbose=True)
        print()
        print(format_summary_block(f"baseline_{model_name}", results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--datasets", type=str, default="metabric,gbsg,support")
    p.add_argument("--models", type=str, default="coxph,rsf,deepsurv,deephit,coxtime")
    args = p.parse_args()
    main(seeds=list(range(args.seeds)), datasets=args.datasets.split(","),
         models=args.models.split(","))
