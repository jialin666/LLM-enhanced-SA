"""Figure B.8: construction of the ordinal reference used for prompt selection.

For every cohort, the Kaplan-Meier curve of the fixed development hold-out
(``llmsa.data.dev_holdout_positions``) is divided into three parts along the
time axis at the tertiles of the observed event times, exactly as
``text_generation.select_prompt._km_categories`` does: events are cut at the
0.33 / 0.66 quantiles of 1 - S_hat(T_i), which is the same as cutting at the
tertiles of the event times, and the drawn cut-points t1 / t2 are the latest
event time of the low and intermediate class. Censored subjects receive no
label.

Outputs (PNG 300 dpi, EPS, TIF) into ``--outdir``:
  ground_truth_label_<COHORT>_corrected.*   one panel per cohort
  ground_truth_label_appendix.*             all cohorts in one grid (Appendix B); pass
                                            --main <cohort> to leave one out for the main text

Usage (from SA-Transformer/final, with the project venv):
    LLMSA_REPO_ROOT=.. python -m text_generation.plot_label_construction \
        --outdir ../new_implementation/Journal-of-Biomedical-Informatics/figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

from llmsa.data import dev_holdout_positions
from text_generation.cohorts import COHORTS

# cohort key -> (display name, raw time unit, plotted unit, divisor, re-selected?)
COHORT_SPECS = [
    ("gbsg", "GBSG", "d", "years", 365.25, True),
    ("metabric", "METABRIC", "mo", "years", 12.0, True),
    ("support", "SUPPORT", "d", "days", 1.0, True),
    ("flchain", "FLCHAIN", "d", "years", 365.25, True),
    ("tcga", "TCGA", "d", "years", 365.25, True),
    ("heartfailure", "Heart Failure", "d", "days", 1.0, False),
    ("wpbc", "WPBC", "mo", "months", 1.0, False),
]

ABBR = {"years": "y", "months": "mo", "days": "d"}
# x-range cap in raw units for the single-panel figures only (GBSG: six-year
# window as in the Sep 14 figure; the single event at 6.5 years that takes
# S_hat to 0 is outside it). The appendix grid shows the full follow-up.
XMAX = {"gbsg": 6 * 365.25}
COLORS = {"low": "#9dc3ee", "intermediate": "#3a8de0", "high": "#0d2d6b"}
GREY = "#7f7f7f"


def label_construction(df):
    """Replicates select_prompt._km_categories (corrected labels) and returns
    the fitted KM, the labelled event rows and the two time cut-points."""
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], event_observed=df["event"])
    df = df.copy()
    df["surv_prob"] = kmf.survival_function_at_times(df["time"]).values
    df["outcome_pct"] = 1.0 - df["surv_prob"]
    pool = df[df["event"] == 1].copy()
    low_cut = pool["outcome_pct"].quantile(0.33)
    high_cut = pool["outcome_pct"].quantile(0.66)

    def categorize(p):
        if p <= low_cut:
            return "low"
        if p <= high_cut:
            return "intermediate"
        return "high"

    pool["category"] = pool["outcome_pct"].apply(categorize)
    t1 = float(pool.loc[pool["category"] == "low", "time"].max())
    t2 = float(pool.loc[pool["category"] == "intermediate", "time"].max())
    return kmf, df, pool, t1, t2


def fmt_time(x, unit):
    return f"{x:,.0f} {unit}"


def draw_panel(ax, key, display, raw_unit, plot_unit, div, reselected,
               annotate=False, title_size=11, legend=False):
    cohort = COHORTS[key]
    full = cohort.load_full()
    dev = full.iloc[dev_holdout_positions(full)].reset_index(drop=True)
    kmf, dev, pool, t1, t2 = label_construction(dev)

    # step curve on a fine grid so each part can be coloured separately
    tmax = float(dev["time"].max())
    grid = np.linspace(0, tmax, 2000)
    surv = kmf.survival_function_at_times(grid).values
    parts = [(0.0, t1, "low"), (t1, t2, "intermediate"), (t2, tmax, "high")]
    for lo, hi, cat in parts:
        m = (grid >= lo) & (grid <= hi)
        ax.step(grid[m] / div, surv[m], where="post", color=COLORS[cat], lw=3.2,
                solid_capstyle="butt", zorder=2)

    cens = dev[dev["event"] == 0]
    ax.scatter(cens["time"] / div, cens["surv_prob"], s=34, facecolors="white",
               edgecolors=GREY, linewidths=1.2, zorder=3)
    for cat in ("low", "intermediate", "high"):
        sub = pool[pool["category"] == cat]
        ax.scatter(sub["time"] / div, sub["surv_prob"], s=34, color=COLORS[cat],
                   edgecolors="white", linewidths=0.6, zorder=4)

    for t, lab in ((t1, "$t_1$"), (t2, "$t_2$")):
        ax.axvline(t / div, color="#444444", ls="--", lw=1.3, zorder=1)
        ax.text(t / div, 1.055, lab, ha="center", va="bottom", fontsize=title_size)

    if annotate and len(pool) > 0:
        # pick an intermediate-to-high event near the median as the worked example
        hi = pool[pool["category"] == "high"].sort_values("time")
        ex = hi.iloc[len(hi) // 4] if len(hi) else pool.iloc[len(pool) // 2]
        ax.annotate(
            f"event at $T_i$ = {ex['time'] / div:.1f} {ABBR[plot_unit]}:  "
            f"$1-\\hat S(T_i)$ = {ex['outcome_pct']:.2f}\n"
            "(fraction of the cohort with an event by $T_i$)",
            xy=(ex["time"] / div, ex["surv_prob"]),
            xytext=(0.52, 0.86), textcoords="axes fraction",
            fontsize=9.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=GREY, lw=1.0),
        )

    n_class = pool["category"].value_counts()
    sizes = "/".join(str(int(n_class.get(c, 0))) for c in ("low", "intermediate", "high"))
    dagger = "" if reselected else "$^\\dagger$"
    ax.set_title(
        f"{display}{dagger}: {len(dev)} development subjects, {len(pool)} events; "
        f"$t_1$ = {fmt_time(t1, raw_unit)}, $t_2$ = {fmt_time(t2, raw_unit)}; "
        f"classes {sizes}",
        fontsize=title_size, loc="left", pad=16,
    )
    xmax = XMAX.get(key, tmax) if annotate else tmax
    ax.set_xlim(0, xmax / div * 1.02)
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel(f"follow-up time ({plot_unit})")
    ax.set_ylabel("$\\hat S(t)$ on the development subset")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    if legend:
        add_legend(ax)
    return dict(cohort=display, n_dev=len(dev), n_events=len(pool), t1=t1, t2=t2,
                unit=raw_unit, sizes=sizes)


def legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], color=COLORS["low"], lw=4,
               label="part 1: low survival (earliest third of events)"),
        Line2D([0], [0], color=COLORS["intermediate"], lw=4,
               label="part 2: intermediate (middle third)"),
        Line2D([0], [0], color=COLORS["high"], lw=4,
               label="part 3: high survival (latest third of events)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=GREY, markeredgewidth=1.2, markersize=8,
               label="censored subject (no reference label)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
               markersize=8,
               label="subject with an event, labelled by the part containing $T_i$"),
    ]


def add_legend(ax, loc="lower right", fontsize=9.5):
    ax.legend(handles=legend_handles(), loc=loc, frameon=False, fontsize=fontsize)


def save(fig, outdir, stem):
    for ext, kw in (("png", dict(dpi=300)), ("eps", {}), ("tif", dict(dpi=300))):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"), bbox_inches="tight", **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--cohorts", default=",".join(k for k, *_ in COHORT_SPECS))
    p.add_argument("--main", default="none",
                   help="cohort shown in the main text and left out of the appendix grid "
                        "(default none: all cohorts in the grid; author decision Sep 15)")
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    wanted = set(args.cohorts.split(","))
    specs = [s for s in COHORT_SPECS if s[0] in wanted]

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11})
    rows = []

    # one figure per cohort
    for key, display, raw_unit, plot_unit, div, resel in specs:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        rows.append(draw_panel(ax, key, display, raw_unit, plot_unit, div, resel,
                               annotate=True, legend=True))
        fig.tight_layout()
        save(fig, args.outdir, f"ground_truth_label_{display.replace(' ', '')}_corrected")
        plt.close(fig)

    # appendix grid (all cohorts except the main-text one): 2 columns, legend in the spare slot
    app = [s for s in specs if s[0] != args.main]
    n = len(app)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    spare = nrow * ncol - n
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.9 * nrow + (0 if spare else 0.9)))
    axes = axes.ravel()
    for ax, (key, display, raw_unit, plot_unit, div, resel) in zip(axes, app):
        draw_panel(ax, key, display, raw_unit, plot_unit, div, resel,
                   annotate=False, title_size=9.5)
        ax.set_ylabel("$\\hat S(t)$")
    for ax in axes[n:]:
        ax.axis("off")
    if spare:
        axes[n].legend(handles=legend_handles(), loc="center", frameon=False, fontsize=10.5)
        fig.tight_layout(h_pad=2.2, w_pad=1.5)
    else:
        fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False,
                   fontsize=10, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(h_pad=2.2, w_pad=1.5, rect=(0, 0.06, 1, 1))
    save(fig, args.outdir, "ground_truth_label_appendix")
    plt.close(fig)

    print("cohort | dev | events | t1 | t2 | class sizes")
    for r in rows:
        print(f"{r['cohort']} | {r['n_dev']} | {r['n_events']} | "
              f"{fmt_time(r['t1'], r['unit'])} | {fmt_time(r['t2'], r['unit'])} | {r['sizes']}")


if __name__ == "__main__":
    main()
