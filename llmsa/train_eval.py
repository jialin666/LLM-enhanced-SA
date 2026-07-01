"""Train and evaluate the LLM-SA stacking ensemble across the three cohorts.

Usage:
    python -m llmsa.train_eval --seeds 10 --datasets metabric,gbsg,support --lam 0.3

Requires the LLM artefacts (data/v4_structured_<ds>.csv and
data/embeddings_v4_<ds>.npy) produced by text_generation/generate_features.py.
"""

from __future__ import annotations

import argparse
import time

from .eval import evaluate_model_on_dataset, format_summary_block
from .model import LLMSAStacking, TrainConfig


def per_cohort_cfg(lam: float):
    """Per-cohort hyperparameters (model family is fixed across datasets)."""
    return {
        "metabric": TrainConfig(text_pca_dim=64,  batch_size=128, meta_prior_lam=lam),
        "gbsg":     TrainConfig(text_pca_dim=256, batch_size=64,  meta_prior_lam=lam),
        "support":  TrainConfig(text_pca_dim=64,  batch_size=128, meta_prior_lam=lam),
    }


def main(seeds, datasets, lam, full_support):
    t0 = time.time()
    cfg_per_ds = per_cohort_cfg(lam)
    results = {}
    for ds in datasets:
        print(f"\n############ {ds} (LLM-SA, lam={lam}) ############", flush=True)
        cfg = cfg_per_ds[ds]

        def factory(seed, _cfg=cfg):
            return LLMSAStacking(_cfg, seed=seed)

        results[ds] = evaluate_model_on_dataset(
            factory, ds, seeds=seeds, verbose=True,
            use_v4=True, full_support=full_support,
        )

    runtime = time.time() - t0
    print()
    print(format_summary_block(f"llmsa_lam{lam}", results, runtime_seconds=runtime), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--datasets", type=str, default="metabric,gbsg,support")
    p.add_argument("--lam", type=float, default=0.3,
                   help="L2 prior strength on meta-weights toward [1/3,1/3,1/3]")
    p.add_argument("--full-support", action="store_true",
                   help="use the full 9105-patient SUPPORT cohort instead of the 1500 subsample")
    args = p.parse_args()
    main(seeds=list(range(args.seeds)), datasets=args.datasets.split(","),
         lam=args.lam, full_support=args.full_support)
