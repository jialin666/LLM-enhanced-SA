"""Train and evaluate the LLM-SA stacking ensemble across the three cohorts.

Usage:
    python -m llmsa.train_eval --seeds 30 --datasets metabric,gbsg,support --full-support

Requires the LLM artefacts (data/<version>_structured_<ds>.csv and
data/embeddings_<version>_<ds>.npy) produced by
text_generation/generate_features.py; select the generation with
``--feature-version`` (v4 = as-published two-call pipeline; v5 = unified
reasoning-conditioned pipeline; v5nobrief = its no-briefing ablation arm).
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
        "flchain":  TrainConfig(text_pca_dim=64,  batch_size=128, meta_prior_lam=lam),
        "rotterdam": TrainConfig(text_pca_dim=64, batch_size=128, meta_prior_lam=lam),
        "tcga":     TrainConfig(text_pca_dim=64,  batch_size=128, meta_prior_lam=lam),
        "tcgafull": TrainConfig(text_pca_dim=64,  batch_size=128, meta_prior_lam=lam),
        "heartfailure": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "wpbc":     TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "veteran": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "lung": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "whas500": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "larynx": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "aids": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "brcamicro": TrainConfig(text_pca_dim=32, batch_size=32, meta_prior_lam=lam),
        "gbsg3":    TrainConfig(text_pca_dim=64,  batch_size=64,  meta_prior_lam=lam),
        "gbsg5":    TrainConfig(text_pca_dim=64,  batch_size=64,  meta_prior_lam=lam),
        "metabric3": TrainConfig(text_pca_dim=64, batch_size=128, meta_prior_lam=lam),
        "metabric5": TrainConfig(text_pca_dim=64, batch_size=128, meta_prior_lam=lam),
    }


def main(seeds, datasets, lam, full_support, feature_version="v4", dev_holdout=True):
    t0 = time.time()
    cfg_per_ds = per_cohort_cfg(lam)
    results = {}
    for ds in datasets:
        print(f"\n############ {ds} (LLM-SA, lam={lam}, features={feature_version}, "
              f"dev_holdout={dev_holdout}) ############", flush=True)
        cfg = cfg_per_ds[ds]

        def factory(seed, _cfg=cfg):
            return LLMSAStacking(_cfg, seed=seed)

        results[ds] = evaluate_model_on_dataset(
            factory, ds, seeds=seeds, verbose=True,
            use_v4=True, full_support=full_support,
            feature_version=feature_version, dev_holdout=dev_holdout,
        )

    runtime = time.time() - t0
    print()
    print(format_summary_block(f"llmsa_lam{lam}_{feature_version}", results,
                               runtime_seconds=runtime), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--datasets", type=str, default="metabric,gbsg,support")
    p.add_argument("--lam", type=float, default=1.0,
                   help="L2 prior strength on meta-weights toward [1/3,1/3,1/3] "
                        "(1.0 is the headline setting)")
    p.add_argument("--full-support", action="store_true",
                   help="use the full 9105-patient SUPPORT cohort instead of the 1500 subsample")
    p.add_argument("--feature-version", default="v4",
                   help="artefact tag of the LLM features to load, e.g. v4, v5, v5nobrief, or a --tag-suffix variant such as v5mini")
    p.add_argument("--legacy-splits", action="store_true",
                   help="reproduce the as-submitted splits (no dev holdout pinning)")
    args = p.parse_args()
    main(seeds=list(range(args.seeds)), datasets=args.datasets.split(","),
         lam=args.lam, full_support=args.full_support,
         feature_version=args.feature_version, dev_holdout=not args.legacy_splits)
