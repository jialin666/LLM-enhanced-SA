"""Re-evaluate already-tuned baseline configurations (no re-tuning).

``tune_baselines.py`` selects a configuration per (cohort, model) on the
validation fold and prints a JSON summary block whose entries carry the chosen
``config``. This script reads those blocks back from one or more tuning logs
(later logs override earlier ones for the same key, so a re-selection log can
be passed last) and evaluates the chosen configurations over ``--seeds`` splits,
optionally with Uno's C truncated at the evaluation-grid maximum
(``--cindex-tau grid``). It exists so the metric can be changed without
re-spending the tuning compute; the selection itself is untouched.

Usage:
    python -m llmsa.run_tuned_baselines --logs a.log,b.log \
        --datasets metabric,gbsg --seeds 30 --cindex-tau grid --full-support
"""

from __future__ import annotations

import argparse
import json
import re
import time

from .baselines import DeepSurv, DeepHit, CoxTime, RSF, CoxPH
from .eval import evaluate_model_on_dataset, format_summary_block

CLASSES = {"deepsurv": DeepSurv, "deephit": DeepHit, "coxtime": CoxTime,
           "rsf": RSF, "coxph": CoxPH}


def _chosen_from_logs(paths):
    """Return {(ds, model): config} from the JSON summary blocks of tuning logs."""
    chosen = {}
    for path in paths:
        text = open(path, encoding="utf-8").read()
        # every top-level {...} block that contains '"config"'
        for m in re.finditer(r"\{\s*\n\s*\"[a-z0-9]+:[a-z]+\": \{.*?\n\}", text, re.S):
            try:
                block = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            for key, val in block.items():
                if "config" not in val:
                    continue
                ds, model = key.split(":")
                cfg = dict(val["config"])
                if "hidden" in cfg:
                    cfg["hidden"] = tuple(cfg["hidden"])
                chosen[(ds, model)] = cfg
    return chosen


def main(logs, datasets, models, seeds, full_support, cindex_tau):
    t0 = time.time()
    chosen = _chosen_from_logs(logs)
    summary = {}
    for ds in datasets:
        for model_name in models:
            cfg = chosen.get((ds, model_name))
            if cfg is None:
                print(f"\n!!! no tuned config for {model_name} on {ds} in the given logs", flush=True)
                continue

            def factory(seed, _c=dict(cfg), _m=model_name):
                kw = dict(_c)
                if _m != "coxph":
                    kw["seed"] = seed
                return CLASSES[_m](**kw)

            print(f"\n============ tuned baseline: {model_name} on {ds} ({cfg}) "
                  f"[cindex_tau={cindex_tau}] ============", flush=True)
            res = evaluate_model_on_dataset(
                factory, ds, seeds=list(range(seeds)), verbose=True,
                full_support=full_support, cindex_tau=cindex_tau)
            summary[f"{ds}:{model_name}"] = {
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                "cindex_mean": res.cindex_mean, "cindex_std": res.cindex_std,
                "ibs_mean": res.ibs_mean, "ibs_std": res.ibs_std,
            }
            print(format_summary_block(f"tuned_{model_name}_tau_{cindex_tau}", {ds: res}), flush=True)
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"\ntuned re-evaluation finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True,
                   help="comma-separated tuning logs whose JSON summaries hold the chosen configs "
                        "(later files override earlier ones)")
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", default="coxph,rsf,deepsurv,deephit,coxtime")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--full-support", action="store_true")
    p.add_argument("--cindex-tau", choices=["none", "grid"], default="grid")
    a = p.parse_args()
    main(a.logs.split(","), a.datasets.split(","), a.models.split(","),
         a.seeds, a.full_support, a.cindex_tau)
