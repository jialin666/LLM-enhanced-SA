"""Corrected resampled t-tests and multiple-comparison adjustment for the per-split results.

Reads the per-split test results (``results/per_split_results.csv`` in the
release asset; one row per experiment, cohort, model and split) and re-analyses
them; no model is re-fitted. For every paired comparison it reports

  * the naive paired t interval and Wilcoxon p-value,
  * the Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio 2003,
    Machine Learning 52:239-281; Bouckaert & Frank 2004, PAKDD), which inflates
    the variance of the mean of J split-wise differences from s^2/J to
    (1/J + n_test/n_train) s^2 to account for the overlap between the
    training sets of different random splits,
  * Holm (family-wise) and Benjamini-Hochberg (FDR) adjusted corrected p-values,
    within pre-defined families and over all tests jointly,

and, across cohorts, the exact Wilcoxon signed-rank test over the seven
independent cohorts with one mean difference per cohort (Demsar 2006) and a
DerSimonian-Laird random-effects pooling of the per-cohort differences.

Usage:
    python analysis/corrected_stats.py --results results/per_split_results.csv --out <output dir>
"""
from __future__ import annotations

import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# 70 / 10 / 20 train / validation / test split (llmsa/data.py)
N_TEST_OVER_N_TRAIN = 0.2 / 0.7
J_EXPECTED = 30

COHORTS = ["metabric", "gbsg", "support", "flchain", "tcga", "tcgafull", "heartfailure", "wpbc"]
COHORT_LABEL = {"metabric": "METABRIC", "gbsg": "GBSG", "support": "SUPPORT", "flchain": "FLCHAIN",
                "tcga": "TCGA (n=2,000)", "tcgafull": "TCGA (n=10,986)",
                "heartfailure": "Heart Failure", "wpbc": "WPBC"}
# TCGA (n=2,000) is drawn from TCGA (n=10,986), so tests across cohorts use the other seven
INDEPENDENT = [c for c in COHORTS if c != "tcga"]
BASELINES = ["coxph", "rsf", "deepsurv", "deephit", "coxtime"]
TRAIN_SIZES = [300, 500, 1000, 1400, 2000, 4000]
FULL_TRAIN_N = 7690  # TCGA (n=10,986) training fold

# ---------------------------------------------------------------------------
# Per-split results
# ---------------------------------------------------------------------------
_RES: pd.DataFrame | None = None


def load(path: str) -> pd.DataFrame:
    global _RES
    _RES = pd.read_csv(path)
    return _RES


def get(model, cohort, experiment="benchmark", train_n=None, draw=None, expected=J_EXPECTED) -> dict:
    """{split: (cindex, ibs)} for one experiment / cohort / model (/ train_n / draw)."""
    g = _RES[(_RES.experiment == experiment) & (_RES.cohort == cohort) & (_RES.model == model)]
    if train_n is not None:
        g = g[g.train_n == train_n]
    if draw is not None:
        g = g[g.draw == draw]
    if g.empty:
        raise KeyError(f"no results: {experiment} {cohort} {model} train_n={train_n} draw={draw}")
    if expected is not None and len(g) != expected:
        raise ValueError(f"{experiment} {cohort} {model}: {len(g)} splits, expected {expected}")
    return {int(r.split): (float(r.cindex), float(r.ibs)) for r in g.itertuples()}


def mean_of(blk: dict, k: int) -> float:
    return float(np.mean([v[k] for v in blk.values()]))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def paired_diff(new: dict, ref: dict, metric: str) -> np.ndarray:
    """Per-split improvement of `new` over `ref` (positive = new is better)."""
    seeds = sorted(new)
    if seeds != sorted(ref):
        raise ValueError("seed sets differ")
    k = 0 if metric == "C" else 1
    d = np.array([new[s][k] - ref[s][k] for s in seeds])
    return d if metric == "C" else -d  # IBS: lower is better


def test_diff(d: np.ndarray, rho: float = N_TEST_OVER_N_TRAIN) -> dict:
    J = len(d)
    m, s2 = d.mean(), d.var(ddof=1)
    tq = stats.t.ppf(0.975, J - 1)
    out = {"J": J, "mean_diff": m, "sd_diff": np.sqrt(s2),
           "wins": int((d > 0).sum()), "ties": int((d == 0).sum())}
    for tag, var in (("naive", s2 / J), ("nb", (1.0 / J + rho) * s2)):
        se = np.sqrt(var)
        if se == 0:
            p = 1.0 if m == 0 else 0.0
        else:
            p = 2 * stats.t.sf(abs(m / se), J - 1)
        out[f"{tag}_lo"], out[f"{tag}_hi"], out[f"{tag}_p"] = m - tq * se, m + tq * se, p
        out[f"{tag}_se"] = se
    # break-even overlap factor: the largest r in (1/J + r) s^2 for which the
    # corrected test is still significant at 0.05 (the NB value used is 0.286)
    out["breakeven_r"] = (m / (tq * np.sqrt(s2))) ** 2 - 1.0 / J if s2 > 0 else np.inf
    try:
        out["wilcoxon_p_naive"] = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        out["wilcoxon_p_naive"] = np.nan
    return out


def add_adjustments(df: pd.DataFrame, pcol: str = "nb_p", prefix: str = "") -> pd.DataFrame:
    df = df.copy()
    df[f"{prefix}holm_p"] = multipletests(df[pcol].values, method="holm")[1]
    df[f"{prefix}bh_q"] = multipletests(df[pcol].values, method="fdr_bh")[1]
    return df


def verdict(row) -> str:
    if row["nb_p"] >= 0.05:
        return "n.s. (was sig. naive)" if row["naive_p"] < 0.05 else "n.s. (naive also n.s.)"
    side = "favours new" if row["mean_diff"] > 0 else "FAVOURS REFERENCE"
    if row["holm_p"] < 0.05:
        return f"sig. after Holm, {side}"
    if row["bh_q"] < 0.05:
        return f"sig. after BH only, {side}"
    return f"sig. unadjusted only, {side}"


def random_effects(y: np.ndarray, v: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects and fixed-effect pooling."""
    w = 1 / v
    fe = np.sum(w * y) / w.sum()
    Q = np.sum(w * (y - fe) ** 2)
    k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    wr = 1 / (v + tau2)
    re_m, re_se, fe_se = np.sum(wr * y) / wr.sum(), np.sqrt(1 / wr.sum()), np.sqrt(1 / w.sum())
    return dict(k=k, n_positive=int((y > 0).sum()),
                fixed_mean=fe, fixed_lo=fe - 1.96 * fe_se, fixed_hi=fe + 1.96 * fe_se,
                fixed_p=2 * stats.norm.sf(abs(fe / fe_se)),
                random_mean=re_m, random_lo=re_m - 1.96 * re_se, random_hi=re_m + 1.96 * re_se,
                random_p=2 * stats.norm.sf(abs(re_m / re_se)),
                tau2=tau2, Q=Q, I2=max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full LLM-SA (D) against each tuned baseline and against the best one per cohort-metric."""
    rows = []
    for c in COHORTS:
        D = get("D", c)
        for b in BASELINES:
            ref = get(b, c)
            for metric in ("C", "IBS"):
                r = test_diff(paired_diff(D, ref, metric))
                k = 0 if metric == "C" else 1
                r.update(cohort=c, metric=metric, comparison=f"D vs {b}", baseline=b,
                         new_mean=mean_of(D, k), ref_mean=mean_of(ref, k))
                rows.append(r)
    allb = pd.DataFrame(rows)
    # best tuned baseline per cohort-metric, chosen by mean
    best = []
    for (c, metric), g in allb.groupby(["cohort", "metric"], sort=False):
        idx = g["ref_mean"].idxmax() if metric == "C" else g["ref_mean"].idxmin()
        best.append(allb.loc[idx])
    best = pd.DataFrame(best).reset_index(drop=True)
    best["comparison"] = "D vs best baseline (" + best["baseline"] + ")"
    return allb, best


def across_cohorts(allb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact Wilcoxon signed-rank test over the seven independent cohorts (one mean
    difference per cohort), against each tuned baseline and the best one, and the
    random-effects pooled difference against the best baseline (corrected SEs)."""
    acr, pooled = [], []
    g = allb[allb.cohort.isin(INDEPENDENT)]
    for ref in BASELINES + ["best"]:
        for metric in ("C", "IBS"):
            m = g[g.metric == metric]
            if ref == "best":
                m = m.loc[m.groupby("cohort")["ref_mean"].idxmax() if metric == "C"
                          else m.groupby("cohort")["ref_mean"].idxmin()]
            else:
                m = m[m.baseline == ref]
            y = m["mean_diff"].values
            p = stats.wilcoxon(y, alternative="two-sided", method="exact").pvalue
            acr.append(dict(reference=ref, metric=metric, k=len(y), n_positive=int((y > 0).sum()),
                            median_diff=np.median(y), wilcoxon_exact_p=p))
            if ref == "best":
                re = random_effects(y, m["nb_se"].values ** 2)
                pooled.append(dict(reference="best baseline", metric=metric, **re))
    return pd.DataFrame(acr), pd.DataFrame(pooled)


def ablation() -> pd.DataFrame:
    rows = []
    pairs = [("D", "A", "primary"), ("B", "A", "secondary"), ("C", "A", "secondary"), ("D", "B", "secondary")]
    for c in COHORTS:
        cells = {x: get(x, c) for x in "ABCD"}
        for new, ref, fam in pairs:
            for metric in ("C", "IBS"):
                r = test_diff(paired_diff(cells[new], cells[ref], metric))
                r.update(cohort=c, metric=metric, comparison=f"{new}-{ref}", role=fam)
                rows.append(r)
    return pd.DataFrame(rows)


def learning_curve() -> tuple[pd.DataFrame, pd.DataFrame]:
    """TCGA learning curve: B-A, D-A and D-B by training-set size (test fold fixed),
    Holm adjustment over the seven sizes, and a trend test (per-split OLS slope of
    the gain on log2(train_n), corrected t-test of the 30 slopes)."""
    ns = TRAIN_SIZES + [FULL_TRAIN_N]

    def cells(n):
        if n == FULL_TRAIN_N:  # full training fold, from the benchmark runs on the same splits
            return {x: get(x, "tcgafull") for x in "ABD"}
        return {x: get(x, "tcgafull", "learning_curve", train_n=n) for x in "ABD"}

    rows, gains = [], {}
    for n in ns:
        c = cells(n)
        for comp in ("B-A", "D-A", "D-B"):
            new, ref = comp.split("-")
            for metric in ("C", "IBS"):
                d = paired_diff(c[new], c[ref], metric)
                gains[(comp, n, metric)] = d
                r = test_diff(d)
                r.update(train_n=n, metric=metric, comparison=comp)
                rows.append(r)
    lc = pd.DataFrame(rows)
    lc["holm_p_over_sizes"] = np.nan
    for _, g in lc.groupby(["comparison", "metric"]):
        lc.loc[g.index, "holm_p_over_sizes"] = multipletests(g.nb_p, method="holm")[1]
    trend, x = [], np.log2(ns)
    for comp in ("B-A", "D-A", "D-B"):
        for metric in ("C", "IBS"):
            Y = np.vstack([gains[(comp, n, metric)] for n in ns])  # (sizes, J)
            slopes = np.array([np.polyfit(x, Y[:, j], 1)[0] for j in range(Y.shape[1])])
            r = test_diff(slopes)
            r.update(metric=metric, comparison=f"slope of ({comp}) per doubling of train_n")
            trend.append(r)
    return lc, pd.DataFrame(trend)


def cohort_draws() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for draw in range(5):
        cells = {x: get(x, "tcgafull", "random_cohorts", draw=draw) for x in "ABD"}
        for new, ref in (("D", "A"), ("B", "A")):
            for metric in ("C", "IBS"):
                r = test_diff(paired_diff(cells[new], cells[ref], metric))
                r.update(draw=draw, metric=metric, comparison=f"{new}-{ref}")
                rows.append(r)
    per = pd.DataFrame(rows)
    pooled = []
    for (comp, metric), g in per.groupby(["comparison", "metric"], sort=False):
        means = g["mean_diff"].values
        # (i) between-draw t-test: draws are (nearly) independent 2,000-patient cohorts
        t = stats.ttest_1samp(means, 0.0)
        tq = stats.t.ppf(0.975, len(means) - 1)
        se_b = means.std(ddof=1) / np.sqrt(len(means))
        # (ii) inverse-variance pooling of the corrected within-draw estimates
        w = 1.0 / g["nb_se"].values ** 2
        m_iv = np.sum(w * means) / w.sum()
        se_iv = np.sqrt(1.0 / w.sum())
        pooled.append(dict(comparison=comp, metric=metric, n_draws=len(means),
                           draws_positive=int((means > 0).sum()),
                           between_draw_mean=means.mean(),
                           between_draw_lo=means.mean() - tq * se_b, between_draw_hi=means.mean() + tq * se_b,
                           between_draw_p=t.pvalue,
                           ivw_nb_mean=m_iv, ivw_nb_lo=m_iv - 1.96 * se_iv, ivw_nb_hi=m_iv + 1.96 * se_iv,
                           ivw_nb_p=2 * stats.norm.sf(abs(m_iv / se_iv))))
    return per, pd.DataFrame(pooled)


def briefing() -> pd.DataFrame:
    """Configuration B with and without the cohort briefing (all LLM inputs regenerated)."""
    rows = []
    for c in sorted(_RES.loc[_RES.experiment == "no_briefing", "cohort"].unique()):
        with_b, no_b = get("B", c, "briefing"), get("B", c, "no_briefing")
        for metric in ("C", "IBS"):
            # positive = removing the briefing improves performance
            r = test_diff(paired_diff(no_b, with_b, metric))
            k = 0 if metric == "C" else 1
            r.update(cohort=c, metric=metric, comparison="B(no briefing) - B(briefing)",
                     with_briefing_mean=mean_of(with_b, k), no_briefing_mean=mean_of(no_b, k))
            rows.append(r)
    return pd.DataFrame(rows)


def llmsa_tuned() -> pd.DataFrame:
    """Full LLM-SA with validation-tuned PCA dimension and batch size vs the reported setting."""
    rows = []
    for c in COHORTS:
        T, D = get("D", c, "llmsa_tuned"), get("D", c)
        for metric in ("C", "IBS"):
            r = test_diff(paired_diff(T, D, metric))
            k = 0 if metric == "C" else 1
            r.update(cohort=c, metric=metric, comparison="D(tuned) - D(reported)",
                     tuned_mean=mean_of(T, k), reported_mean=mean_of(D, k))
            rows.append(r)
    return add_adjustments(pd.DataFrame(rows))


def development() -> pd.DataFrame:
    """Development cohorts: full LLM-SA (D) against the covariates-only ensemble (A)
    and against the best tuned baseline of each cohort-metric."""
    rows = []
    dev = _RES[_RES.experiment == "development"]
    for c in dev.cohort.unique():
        n = int(dev[(dev.cohort == c) & (dev.model == "D")].shape[0])
        D, A = get("D", c, "development", expected=n), get("A", c, "development", expected=n)
        base = {b: get(b, c, "development", expected=n) for b in BASELINES}
        for metric in ("C", "IBS"):
            k = 0 if metric == "C" else 1
            means = {b: mean_of(v, k) for b, v in base.items()}
            best = max(means, key=means.get) if metric == "C" else min(means, key=means.get)
            for ref_name, ref in (("A", A), (f"best baseline ({best})", base[best])):
                r = test_diff(paired_diff(D, ref, metric))
                r.update(cohort=c, metric=metric, comparison=f"D vs {ref_name}",
                         D_mean=mean_of(D, k), ref_mean=mean_of(ref, k))
                rows.append(r)
    return pd.DataFrame(rows)


def meta_analysis(abl: pd.DataFrame) -> pd.DataFrame:
    """Random-effects pooling of the per-cohort ablation effects (corrected SEs)."""
    out = []
    for comp in ("D-A", "B-A", "D-B"):
        for metric in ("C", "IBS"):
            for label, excl in (("7 cohorts (excl. TCGA n=2,000)", ["tcga"]),
                                ("7 cohorts (excl. TCGA n=10,986)", ["tcgafull"]),
                                ("6 cohorts (excl. TCGA n=2,000 and SUPPORT)", ["tcga", "support"])):
                g = abl[(abl.comparison == comp) & (abl.metric == metric) & (~abl.cohort.isin(excl))]
                re = random_effects(g["mean_diff"].values, g["nb_se"].values ** 2)
                out.append(dict(comparison=comp, metric=metric, pool=label, **re))
    return pd.DataFrame(out)


def table2() -> pd.DataFrame:
    """Mean C-index and IBS of every Table 2 entry."""
    rows = []
    for c in COHORTS:
        for m in BASELINES + list("ABCD"):
            blk = get(m, c)
            rows.append(dict(cohort=c, model=m if m in BASELINES else f"LLM-SA {m}",
                             C_mean=mean_of(blk, 0), IBS_mean=mean_of(blk, 1)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results/per_split_results.csv")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    load(a.results)

    allb, best = benchmark()
    acr, pooled_best = across_cohorts(allb)
    abl = ablation()
    lc, trend = learning_curve()
    draws, pooled = cohort_draws()
    brief = briefing()

    # ---- multiple-testing families (corrected p-values) ----
    fam = OrderedDict()
    fam["F1: full LLM-SA vs best tuned baseline"] = best
    fam["F2: full LLM-SA vs each tuned baseline"] = allb
    fam["F3: ablation, primary (D-A)"] = abl[abl.role == "primary"]
    fam["F4: ablation, secondary (B-A, C-A, D-B)"] = abl[abl.role == "secondary"]
    fam["F5: learning curve TCGA (by train_n)"] = lc
    fam["F6: cohort briefing (B, no-brief vs brief)"] = brief
    adj = {}
    for name, df in fam.items():
        df = add_adjustments(df.reset_index(drop=True))
        df["family"] = name
        df["verdict"] = df.apply(verdict, axis=1)
        adj[name] = df
    # global adjustment over every distinct test (F2 contains F1, so F1 is not counted twice)
    keys = [k for k in adj if not k.startswith("F1")]
    g = add_adjustments(pd.concat([adj[k] for k in keys], ignore_index=True), prefix="global_")
    ofs = 0
    for k in keys:
        n = len(adj[k])
        adj[k] = adj[k].assign(global_holm_p=g["global_holm_p"].values[ofs:ofs + n],
                               global_bh_q=g["global_bh_q"].values[ofs:ofs + n])
        ofs += n
    f2 = adj["F2: full LLM-SA vs each tuned baseline"].set_index(["cohort", "metric", "baseline"])
    f1 = adj["F1: full LLM-SA vs best tuned baseline"]
    f1["global_holm_p"] = [f2.loc[(r.cohort, r.metric, r.baseline), "global_holm_p"] for r in f1.itertuples()]
    f1["global_bh_q"] = [f2.loc[(r.cohort, r.metric, r.baseline), "global_bh_q"] for r in f1.itertuples()]

    meta = meta_analysis(pd.concat([adj["F3: ablation, primary (D-A)"],
                                    adj["F4: ablation, secondary (B-A, C-A, D-B)"]], ignore_index=True))

    files = {
        "F1: full LLM-SA vs best tuned baseline": "F1_full_vs_best_baseline.csv",
        "F2: full LLM-SA vs each tuned baseline": "F2_full_vs_each_baseline.csv",
        "F3: ablation, primary (D-A)": "F3_ablation_primary_D_minus_A.csv",
        "F4: ablation, secondary (B-A, C-A, D-B)": "F4_ablation_secondary.csv",
        "F5: learning curve TCGA (by train_n)": "F5_learning_curve_tcga.csv",
        "F6: cohort briefing (B, no-brief vs brief)": "F6_briefing_noBrief_vs_brief.csv",
    }
    fmt = dict(index=False, float_format="%.6g")
    for k, f in files.items():
        adj[k].to_csv(os.path.join(a.out, f), **fmt)
    trend.to_csv(os.path.join(a.out, "F5b_learning_curve_trend.csv"), **fmt)
    draws.to_csv(os.path.join(a.out, "F7_cohort2000_per_draw.csv"), **fmt)
    pooled.to_csv(os.path.join(a.out, "F7b_cohort2000_pooled.csv"), **fmt)
    meta.to_csv(os.path.join(a.out, "F8_meta_analysis_across_cohorts.csv"), **fmt)
    acr.to_csv(os.path.join(a.out, "F9_across_cohort_wilcoxon.csv"), **fmt)
    pooled_best.to_csv(os.path.join(a.out, "F9b_pooled_full_vs_best_baseline.csv"), **fmt)
    llmsa_tuned().to_csv(os.path.join(a.out, "F10_llmsa_tuned_vs_reported.csv"), **fmt)
    if (_RES.experiment == "development").any():
        development().to_csv(os.path.join(a.out, "F11_development_cohorts.csv"), **fmt)
    table2().to_csv(os.path.join(a.out, "table2_means.csv"), index=False, float_format="%.4f")
    print("done; families:", {k: len(v) for k, v in adj.items()}, "global tests:", len(g))


if __name__ == "__main__":
    main()
