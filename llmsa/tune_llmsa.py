"""Per-cohort hyperparameter tuning for LLM-SA (protocol parity with R2-M5).

``tune_baselines.py`` grid-searched every baseline per cohort, but LLM-SA itself
was left on stock settings, so the published comparison tuned the competitors
and not our own method. This script closes that gap using the *same* protocol:
candidates are scored by IPCW C-index on the **validation fold only**, over
``--tune-seeds`` seeds, and the winner is then evaluated over ``--seeds`` seeds.
Test folds are never touched during selection.

The search is staged coordinate descent rather than a full product, keeping it
to ~14 fits per cohort instead of ~144:

    stage 1  internal RSF     n_estimators x max_depth
    stage 2  DeepSurv branch  hidden x lr
    stage 3  meta-learner     (meta_fit, meta_objective) x meta_prior_lam
    stage 4  embedding        text_pca_dim            [cells C/D only]

Stage 3 matters most. The meta-weights are three parameters; fitted on a
held-out validation fold they come from ~n_val patients, which on a small cohort
is a handful of events. ``meta_fit="oof"`` fits them on K-fold out-of-fold
predictions over the training fold instead. Independently, the default
``meta_objective="brier"`` optimises squared error against the survival labels
while we report C-index, so it can prefer a well-calibrated but poorly
discriminating base learner; ``"cindex"`` optimises the reported metric.

Usage:
    python -m llmsa.tune_llmsa --datasets heartfailure,wpbc --cell A \
        --tune-seeds 5 --seeds 30
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np

from .data import load_dataset_v4, _csv_path
from .metrics import ipcw_cindex, ipcw_ibs
from .model import LLMSAStacking, TrainConfig
from .train_eval import per_cohort_cfg

CELLS = {"A": (False, False), "B": (True, False),
         "C": (False, True), "D": (True, True)}


def _score(cfg, ds, seeds, use_num, subset, feature_version, split):
    """Mean IPCW C-index of ``cfg`` on ``split`` ('val' for selection)."""
    vals = []
    for seed in seeds:
        try:
            b = load_dataset_v4(ds, seed=seed, feature_version=feature_version,
                                use_numerics=use_num, subset=subset)
            c = replace(cfg, text_pca_dim=min(cfg.text_pca_dim,
                                              max(4, len(b.train.time) // 10)))
            m = LLMSAStacking(c, seed=seed)
            m.fit(b)
            r = np.asarray(m.predict_risk(b, split), dtype=float)
            tgt = getattr(b, split)
            vals.append(ipcw_cindex(b.train.time, b.train.event,
                                    tgt.time, tgt.event, r))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else -np.inf


def _evaluate(cfg, ds, seeds, use_num, subset, feature_version):
    cs, ibs = [], []
    for seed in seeds:
        try:
            b = load_dataset_v4(ds, seed=seed, feature_version=feature_version,
                                use_numerics=use_num, subset=subset)
            c = replace(cfg, text_pca_dim=min(cfg.text_pca_dim,
                                              max(4, len(b.train.time) // 10)))
            m = LLMSAStacking(c, seed=seed)
            m.fit(b)
            r = np.asarray(m.predict_risk(b, "test"), dtype=float)
            s = np.asarray(m.predict_surv(b, "test"), dtype=float)
            cs.append(ipcw_cindex(b.train.time, b.train.event,
                                  b.test.time, b.test.event, r))
            ibs.append(ipcw_ibs(b.train.time, b.train.event, b.test.time,
                                b.test.event, s, b.time_grid))
        except Exception:
            continue
    return cs, ibs


def main(datasets, cell, tune_seeds, seeds, feature_version, lam, stages, out_tag,
         meta_fit="oof"):
    use_num, use_text = CELLS[cell]
    chosen, results = {}, {}
    for ds in datasets:
        t0 = time.time()
        base = per_cohort_cfg(lam).get(ds)
        cfg = TrainConfig(
            text_pca_dim=getattr(base, "text_pca_dim", 32) if base else 32,
            batch_size=getattr(base, "batch_size", 32) if base else 32,
            meta_prior_lam=lam, use_text=use_text,
            meta_fit=meta_fit)  # see the "meta" grid below for why this is fixed
        print(f"\n######## tuning LLM-SA on {ds} (cell {cell}) ########", flush=True)
        cur = _score(cfg, ds, tune_seeds, use_num, None, feature_version, "val")
        print(f"  start: val C = {cur:.4f}  ({cfg.rsf_n_estimators}/"
              f"{cfg.rsf_max_depth}, {cfg.hidden}, lr={cfg.lr}, "
              f"{cfg.meta_fit}/{cfg.meta_objective}, lam={cfg.meta_prior_lam})",
              flush=True)

        grids = {
            "rsf": [dict(rsf_n_estimators=n, rsf_max_depth=d)
                    for n in (100, 300) for d in (3, 6, None)],
            "deep": [dict(hidden=h, lr=l)
                     for h in ((64, 64), (128, 64)) for l in (1e-3, 5e-4)],
            # meta_fit is FIXED to "oof" here and is not a tunable. With
            # meta_fit="val" the meta-weights are fitted on the validation fold
            # and the selection score is then read off that same fold, which is
            # wildly optimistic (WPBC: val C = 0.81 against test C = 0.60) and
            # would make stage 3 pick "val" purely because it scores itself.
            # Out-of-fold stacking leaves val untouched by weight fitting, which
            # both restores a clean selection metric and matches the baselines,
            # none of which fit combination weights on val.
            "meta": [dict(meta_objective=o, meta_prior_lam=l)
                     for o in ("brier", "cindex") for l in (0.0, 0.1, 0.3, 1.0)],
            "pca": [dict(text_pca_dim=d) for d in (8, 16, 32)],
        }
        for stage in stages:
            if stage == "pca" and not use_text:
                continue
            best, best_s = None, cur
            for upd in grids[stage]:
                cand = replace(cfg, **upd)
                s = _score(cand, ds, tune_seeds, use_num, None,
                           feature_version, "val")
                flag = ""
                if s > best_s:
                    best, best_s, flag = upd, s, "  <-"
                print(f"    [{stage}] {upd} val C = {s:.4f}{flag}", flush=True)
            if best is not None:
                cfg, cur = replace(cfg, **best), best_s
                print(f"  -> stage {stage}: adopted {best} (val C = {cur:.4f})",
                      flush=True)
            else:
                print(f"  -> stage {stage}: no improvement, kept current",
                      flush=True)

        cs, ibs = _evaluate(cfg, ds, seeds, use_num, None, feature_version)
        chosen[ds] = {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in cfg.__dict__.items()}
        results[ds] = dict(cindex_mean=float(np.mean(cs)),
                           cindex_std=float(np.std(cs, ddof=1)),
                           ibs_mean=float(np.mean(ibs)),
                           ibs_std=float(np.std(ibs, ddof=1)),
                           n_seeds=len(cs), val_cindex=cur)
        print(f"  == {ds}: tuned LLM-SA test C = {np.mean(cs):.4f} "
              f"+/- {np.std(cs, ddof=1):.4f}, IBS = {np.mean(ibs):.4f} "
              f"({len(cs)} seeds, {time.time()-t0:.0f}s)", flush=True)

    out = _csv_path("data", f"tuned_llmsa_{cell}{out_tag}.json")
    with open(out, "w") as f:
        json.dump({"cell": cell, "chosen": chosen, "results": results}, f,
                  indent=2, default=str)
    print(f"\nwritten -> {out}")
    print("TUNE_LLMSA_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="heartfailure,wpbc")
    p.add_argument("--cell", default="A", choices=list(CELLS))
    p.add_argument("--tune-seeds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--feature-version", default="v5")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--stages", default="rsf,deep,meta,pca")
    p.add_argument("--meta-fit", default="oof", choices=("oof", "val"),
                   help="weight-fitting scheme held fixed during tuning; "
                        "'val' is contaminated as a selection metric")
    p.add_argument("--out-tag", default="")
    a = p.parse_args()
    main(a.datasets.split(","), a.cell, list(range(a.tune_seeds)),
         list(range(a.seeds)), a.feature_version, a.lam,
         a.stages.split(","), a.out_tag, a.meta_fit)
