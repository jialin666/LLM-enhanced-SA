"""Sensitivity of LLM-SA to the meta-learner prior strength, per stratum.

The baselines were tuned for this revision (R2-M5) but LLM-SA never was: it has
carried ``meta_prior_lam = 1.0`` since the pooled-cohort runs. That prior pulls
the simplex weights toward the uniform vector [1/3, 1/3, 1/3]. On a pooled
cohort the validation fold is large enough for the data term to dominate and the
prior is harmless. On a single cancer type the validation fold is a few dozen
patients, the prior dominates, and the ensemble collapses toward the *average*
of its three base learners -- which loses to the best of them whenever the three
disagree.

lambda only enters ``_simplex_pgd``; the base learners do not depend on it. So
the base curves are fitted once per (stratum, seed, cell) and the meta-weights
are re-solved for every lambda, making the sweep nearly free.

Reported per stratum and lambda: test C-index and IBS, the mean meta-weights,
and -- for reference -- the best single base learner (an oracle, not a usable
model).

Usage:
    python -m llmsa.run_lambda_sweep --dataset tcgafull --stratum-col cancer_type_raw \
        --seeds 30 --min-n 100 --feature-version v5 --lams 0,0.1,0.3,1.0
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import load_dataset_v4, _csv_path, _read_cohort_csv
from .metrics import ipcw_cindex, ipcw_ibs
from .model import LLMSAStacking, TrainConfig, _simplex_pgd


def _val_targets(bundle):
    """Censoring-masked per (sample, grid) survival labels on the validation fold."""
    grid = bundle.time_grid
    n_val, G = len(bundle.val), len(grid)
    y = np.ones((n_val, G), dtype=np.float32)
    for i in range(n_val):
        t, e = bundle.val.time[i], bundle.val.event[i]
        if e == 1:
            y[i, grid >= t] = 0.0
        else:
            y[i, grid > t] = np.nan
    return y


def _refit_weights(model, bundle, lam):
    """Re-solve the meta-weights at a new prior strength, reusing fitted bases."""
    S_val = np.stack([b.surv(bundle, "val") for b in model.bases], axis=0)
    K = S_val.shape[0]
    S_flat = S_val.transpose(1, 2, 0).reshape(-1, K)
    y_flat = _val_targets(bundle).reshape(-1)
    m = ~np.isnan(y_flat)
    w_prior = np.ones(K, dtype=np.float64) / K
    return _simplex_pgd(S_flat[m], y_flat[m], w_prior,
                        lam, model.train_cfg.pgd_lr, model.train_cfg.pgd_steps)


def main(dataset, stratum_col, seeds, min_n, lams, feature_version, cells):
    raw = _read_cohort_csv(dataset)
    whole = stratum_col in (None, "", "none")
    if whole:
        # treat the cohort as a single stratum
        counts = pd.Series({dataset: len(raw)})
        strata = [dataset]
        print(f"whole cohort {dataset} (n={len(raw)}); "
              f"lambdas = {lams}; cells = {cells}\n", flush=True)
    else:
        counts = raw[stratum_col].value_counts()
        strata = [s for s, c in counts.items() if c >= min_n]
        print(f"{len(strata)} strata with >= {min_n} patients; "
              f"lambdas = {lams}; cells = {cells}\n", flush=True)

    CELLS = {"A": (False, False), "B": (True, False),
             "C": (False, True), "D": (True, True)}
    rows = []
    for st in sorted(strata, key=lambda s: counts[s]):
        acc = {}          # (cell, lam) -> list of (cindex, ibs)
        wacc = {}         # (cell, lam) -> list of weight vectors
        oracle = []       # best single base learner, per seed
        for seed in seeds:
            try:
                for cell in cells:
                    use_num, use_text = CELLS[cell]
                    b = load_dataset_v4(dataset, seed=seed,
                                        feature_version=feature_version,
                                        use_numerics=use_num,
                                        subset=None if whole else (stratum_col, st))
                    dpca = int(min(32, max(4, len(b.train.time) // 10)))
                    cfg = TrainConfig(text_pca_dim=dpca, batch_size=32,
                                      meta_prior_lam=lams[0], use_text=use_text)
                    m = LLMSAStacking(cfg, seed=seed)
                    m.fit(b)                       # bases fitted once
                    for lam in lams:
                        w = _refit_weights(m, b, lam)
                        m.weights_ = w.astype(np.float32)
                        risk = np.asarray(m.predict_risk(b, "test"), dtype=float)
                        surv = np.asarray(m.predict_surv(b, "test"), dtype=float)
                        c = ipcw_cindex(b.train.time, b.train.event,
                                        b.test.time, b.test.event, risk)
                        i_ = ipcw_ibs(b.train.time, b.train.event,
                                      b.test.time, b.test.event, surv, b.time_grid)
                        acc.setdefault((cell, lam), []).append((c, i_))
                        wacc.setdefault((cell, lam), []).append(w)
                    if cell == cells[0]:
                        # oracle: best of the three bases on this split
                        cs = []
                        for base in m.bases:
                            S = np.asarray(base.surv(b, "test"), dtype=float)
                            cs.append(ipcw_cindex(b.train.time, b.train.event,
                                                  b.test.time, b.test.event,
                                                  -S.sum(axis=1)))
                        oracle.append(max(cs))
            except Exception:
                continue
        if not acc:
            print(f"{st:<8} (no successful seeds)", flush=True)
            continue
        n = min(len(v) for v in acc.values())
        if n < 10:
            print(f"{st:<8} (too few successful seeds: {n})", flush=True)
            continue
        ev = int(raw["event"].sum()) if whole else \
            int(raw.loc[raw[stratum_col] == st, "event"].sum())
        parts = []
        for cell in cells:
            for lam in lams:
                v = np.array(acc[(cell, lam)][:n])
                w = np.stack(wacc[(cell, lam)][:n]).mean(axis=0)
                rows.append(dict(stratum=st, n=int(counts[st]), events=ev, seeds=n,
                                 cell=cell, lam=lam,
                                 cindex=float(v[:, 0].mean()),
                                 ibs=float(v[:, 1].mean()),
                                 w_coxph=float(w[0]), w_rsf=float(w[1]),
                                 w_deepsurv=float(w[2]),
                                 oracle_best_base=float(np.mean(oracle[:n])) if oracle else np.nan))
                if cell == cells[0]:
                    parts.append(f"l={lam}:{v[:, 0].mean():.4f}")
        print(f"{st:<8} n={counts[st]:<5} ev={ev:<4} " + "  ".join(parts) +
              (f"   [oracle {np.mean(oracle[:n]):.4f}]" if oracle else ""), flush=True)

    df = pd.DataFrame(rows)
    suffix = "_whole" if whole else ""
    out = _csv_path("data", f"lambda_sweep_{dataset}{suffix}.csv")
    df.to_csv(out, index=False)
    print(f"\nwritten -> {out}")
    if not df.empty:
        print("\nmean over strata, cell A:")
        a = df[df.cell == cells[0]]
        for lam in lams:
            s = a[a.lam == lam]
            print(f"  lambda={lam:<5} C={s.cindex.mean():.4f}  IBS={s.ibs.mean():.4f}  "
                  f"weights=[{s.w_coxph.mean():.2f}, {s.w_rsf.mean():.2f}, "
                  f"{s.w_deepsurv.mean():.2f}]")
    print("LAMBDA_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tcgafull")
    p.add_argument("--stratum-col", default="cancer_type_raw")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--min-n", type=int, default=100)
    p.add_argument("--lams", default="0,0.1,0.3,1.0")
    p.add_argument("--feature-version", default="v5")
    p.add_argument("--cells", default="A,B")
    args = p.parse_args()
    main(args.dataset, args.stratum_col, list(range(args.seeds)), args.min_n,
         [float(x) for x in args.lams.split(",")], args.feature_version,
         args.cells.split(","))
