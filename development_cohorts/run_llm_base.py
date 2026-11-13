"""The LLM's estimates as a FOURTH base learner, instead of as covariates.

Cells B/C/D feed the LLM output into the covariate block of the existing three
learners, where dense covariates can render it redundant. Here the LLM instead
competes as a model in its own right: the three conventional learners see no LLM
input at all, and a dedicated LLM learner joins them under the same simplex
meta-learner.

Arms (all share seeds, splits and meta-learner settings):

    A     covariates only, 3 learners              -- the no-LLM control
    A+L   covariates only, 3 learners + LLM base   -- LLM as a MODEL
    B     covariates + N_i, 3 learners             -- LLM as FEATURES (reference)
    B+L   covariates + N_i, 3 learners + LLM base  -- both
    solo  the LLM base learner alone               -- how good is the 4th learner?

Pre-registered prediction (recorded before running, from the zero-shot analysis).
At lam=1.0 with few events the meta-weights sit at uniform, so a fourth learner
receives 1/4 of the weight whether or not it deserves it. Hence A+L should beat
A where the LLM's standalone estimate is competitive with the fitted learners,
and lose where it is near chance:

    expect GAIN  larynx    (zero-shot C 0.6491; never overtaken by a fitted
                            model at full cohort size)
    expect GAIN  whas500   (zero-shot C 0.7409)
    expect LOSS  brcamicro (zero-shot C 0.5256; probe-set intensities the LLM
                            cannot interpret -- pre-specified negative control)

Usage:
    python -m development_cohorts.run_llm_base --datasets larynx,whas500,brcamicro --seeds 30
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

from llmsa.data import load_dataset_v4

from development_cohorts import DATA_DIR  # importing the package registers the cohorts
from llmsa.metrics import ipcw_cindex, ipcw_ibs
from llmsa.model import (LLMSAStacking, TrainConfig, _BaseLLM,
                    _TIME_UNIT_PER_YEAR, anchor_years_for)

ARMS = {
    #        use_numerics, use_llm_base
    "A":    (False, False),
    "A+L":  (False, True),
    "B":    (True,  False),
    "B+L":  (True,  True),
}


def _ci(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = tdist.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def main(datasets, seeds, lam, feature_version, mode, arms, out_tag=""):
    rows = []
    for ds in datasets:
        res = {a: {} for a in arms}
        solo, weights = {}, {a: [] for a in arms}
        for seed in seeds:
            for arm in arms:
                use_num, use_llm = ARMS[arm]
                try:
                    b = load_dataset_v4(ds, seed=seed, feature_version=feature_version,
                                        use_numerics=use_num)
                    d = int(min(32, max(4, len(b.train.time) // 10)))
                    cfg = TrainConfig(text_pca_dim=d, batch_size=32,
                                      meta_prior_lam=lam, use_text=False,
                                      use_llm_base=use_llm, llm_base_mode=mode,
                                      llm_anchor_years=anchor_years_for(feature_version, ds))
                    m = LLMSAStacking(cfg, seed=seed)
                    m.fit(b)
                    r = np.asarray(m.predict_risk(b, "test"), dtype=float)
                    s = np.asarray(m.predict_surv(b, "test"), dtype=float)
                    res[arm][seed] = (
                        ipcw_cindex(b.train.time, b.train.event,
                                    b.test.time, b.test.event, r),
                        ipcw_ibs(b.train.time, b.train.event, b.test.time,
                                 b.test.event, s, b.time_grid))
                    weights[arm].append(m.weights_)
                    if arm == "A+L" and seed not in solo:
                        # the fourth learner on its own, same split
                        lb = _BaseLLM(mode=mode, seed=seed,
                                      unit_per_year=_TIME_UNIT_PER_YEAR.get(
                                          ds.lower(), 365.25),
                                      anchor_years=anchor_years_for(feature_version, ds))
                        lb.fit(b)
                        S = np.asarray(lb.surv(b, "test"), dtype=float)
                        solo[seed] = (
                            ipcw_cindex(b.train.time, b.train.event, b.test.time,
                                        b.test.event, -S.sum(axis=1)),
                            ipcw_ibs(b.train.time, b.train.event, b.test.time,
                                     b.test.event, S, b.time_grid))
                except Exception:
                    continue

        print(f"\n=== {ds} (mode={mode}, lam={lam}) ===", flush=True)
        for arm in arms:
            if not res[arm]:
                continue
            c = np.array([v[0] for v in res[arm].values()])
            i_ = np.array([v[1] for v in res[arm].values()])
            w = np.stack(weights[arm]).mean(0) if weights[arm] else None
            lo, hi = _ci(c)
            ws = "[" + ",".join(f"{x:.2f}" for x in w) + "]" if w is not None else ""
            print(f"  {arm:<5} C={c.mean():.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"IBS={i_.mean():.4f}  n={len(c)}  w={ws}", flush=True)
        if solo:
            c = np.array([v[0] for v in solo.values()])
            lo, hi = _ci(c)
            print(f"  {'solo':<5} C={c.mean():.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"IBS={np.mean([v[1] for v in solo.values()]):.4f}  "
                  f"(LLM base learner alone)", flush=True)

        for a, base in (("A+L", "A"), ("B+L", "B")):
            if a not in res or base not in res or not res[a] or not res[base]:
                continue
            sh = sorted(set(res[a]) & set(res[base]))
            if len(sh) < 5:
                continue
            d = np.array([res[a][s][0] - res[base][s][0] for s in sh])
            di = np.array([res[a][s][1] - res[base][s][1] for s in sh])
            lo, hi = _ci(d)
            p = wilcoxon(d).pvalue if len(sh) > 5 and np.any(d != 0) else np.nan
            print(f"  {a} - {base}: dC = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"p={p:.4f}   dIBS = {di.mean():+.4f}", flush=True)
            rows.append(dict(dataset=ds, contrast=f"{a}-{base}", mode=mode,
                             seeds=len(sh), delta_c=d.mean(), lo=lo, hi=hi, p=p,
                             delta_ibs=di.mean(),
                             c_with=np.mean([res[a][s][0] for s in sh]),
                             c_without=np.mean([res[base][s][0] for s in sh]),
                             solo_c=(np.mean([solo[s][0] for s in sh if s in solo])
                                     if solo else np.nan)))
    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(DATA_DIR, f"llm_base_{mode}{out_tag}.csv")
        df.to_csv(out, index=False)
        print(f"\nwritten -> {out}")
    print("LLM_BASE_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="larynx,whas500,brcamicro")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--feature-version", default="v5")
    p.add_argument("--mode", default="cox1", choices=("cox1", "anchor"))
    p.add_argument("--arms", default="A,A+L,B,B+L")
    p.add_argument("--out-tag", default="",
                   help="suffix for the output CSV so a new panel does not "
                        "overwrite an earlier one")
    a = p.parse_args()
    main(a.datasets.split(","), list(range(a.seeds)), a.lam,
         a.feature_version, a.mode, a.arms.split(","), a.out_tag)
