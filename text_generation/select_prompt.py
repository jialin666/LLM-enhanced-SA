"""Optimal target-prompt selection (manuscript Sec 3.2.2).

For one cohort:

  1. Query the meta prompt to obtain a candidate set of target prompts and keep
     the well-formed ones (exactly one literal ``{FEATURES}`` placeholder).
  2. Restrict to the fixed development hold-out (``llmsa.data.dev_holdout_positions``);
     these rows are pinned to the training fold of every per-seed protocol
     split, so prompt selection never sees validation or test outcomes.
  3. Build ordinal reference labels y_i in {low, intermediate, high} from a
     Kaplan-Meier fit on the development rows. S_hat(T_i) is the *population*
     survival at patient i's own observed time, so 1 - S_hat(T_i) is the
     fraction of the cohort that had already died by the time patient i died:
     an outcome percentile. Among patients with an observed event the labels
     are the empirical tertiles (33rd/66th percentiles) of that percentile --
     early deaths = low survival, late deaths = high survival. Censored
     patients receive no reference label (their percentile is only a lower
     bound). ``--legacy-labels`` reproduces the as-submitted construction,
     which cut S_hat(T_i) itself on all development patients and therefore
     labelled early observed times "high survival" and long follow-up "low"
     (the reverse of the prognosis ordering for events; see the revision
     response to Reviewer 1, Comment 4).
  4. On a random evaluation subset of development patients, generate the
     narrative + numeric estimates with each candidate prompt (one call, v5
     unified design, {BRIEFING} filled per --briefing), pass the narrative
     back to the LLM for a predicted label, and score agreement with phi
     (+1/0/-1). The numeric 5-year estimates give a free secondary ordinal
     score, reported alongside.
  5. The prompt score S(P_m) is the summed agreement over the subset. The
     selected prompt P* = argmax_m S(P_m) is used downstream.

Outputs (under prompts/):
    <dataset>_target_prompts.json   candidate prompts keyed by id
    <dataset>_templates_grades.json summed grade per candidate id (per-run;
                                    not tracked in git -- LLM outputs vary by run)
    <dataset>_numeric_grades.json   secondary numeric-agreement score per id

Usage:
    python -m text_generation.select_prompt --dataset gbsg --reuse-candidates
    python -m text_generation.select_prompt --dataset gbsg --legacy-labels     # as-submitted labels
    python -m text_generation.select_prompt --dataset gbsg --legacy-full-cohort
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from lifelines import KaplanMeierFitter

from llmsa.data import dev_holdout_positions

from .cohorts import COHORTS
from .meta_prompt import (
    BRIEFING_EMPTY, generate_target_prompts, render_features_block, fill_template,
    generate_narrative_and_numerics,
)
from .score import score

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

_ALLOWED_PLACEHOLDERS = {"{FEATURES}", "{BRIEFING}"}


def _valid_templates(all_templates):
    """Keep templates carrying both literal placeholders and no others.

    Only uppercase brace tokens are treated as placeholders, so the JSON
    braces in the v5 numeric-output contract do not trip the filter.
    """
    valid = {}
    key = 0
    for template in all_templates:
        text = template["template"]
        for tok in ("FEATURES", "BRIEFING"):
            if "{{%s}}" % tok in text:
                text = text.replace("{{%s}}" % tok, "{%s}" % tok)
        template["template"] = text
        placeholders = set(re.findall(r"\{[A-Z_]+\}", text))
        if placeholders != _ALLOWED_PLACEHOLDERS:
            print(f"  dropping template with placeholders: {sorted(placeholders)}")
            continue
        if "estimated_5yr_survival_prob" not in text:
            print("  dropping template without the numeric-output contract")
            continue
        valid[key] = template
        key += 1
    return valid


def _select_briefing(cohort, briefing_mode: str) -> str:
    if briefing_mode == "sanitized":
        return cohort.briefing
    if briefing_mode == "legacy":
        return cohort.briefing_legacy
    if briefing_mode == "none":
        return BRIEFING_EMPTY
    raise ValueError(f"unknown briefing mode: {briefing_mode!r}")


def _numeric_ordinal_grade(probs, cats):
    """Ordinal agreement (summed phi) of tertile-ranked numeric 5y estimates
    with the KM reference labels. Returns 0 for an empty list."""
    if not probs:
        return 0
    import numpy as np
    from .score import phi
    arr = np.asarray(probs, dtype=float)
    lo, hi = np.quantile(arr, 0.33), np.quantile(arr, 0.66)
    total = 0
    for p, c in zip(arr, cats):
        pred = "low" if p <= lo else ("intermediate" if p <= hi else "high")
        total += phi(pred, c)
    return int(total)


def _km_categories(df, legacy: bool = False):
    """Assign ordinal KM reference categories.

    Corrected construction (default): the KM curve is fitted on all rows
    (events and censored, as KM requires); each patient's outcome percentile
    is ``1 - S_hat(T_i)``; the labelled pool is restricted to observed events
    and cut at the empirical tertiles of the percentile, so the earliest third
    of deaths is "low", the latest third "high". Returns only the labelled
    (event) rows.

    ``legacy=True`` reproduces the as-submitted construction: tertiles of
    S_hat(T_i) on all rows with low S_hat -> "low". Because S_hat decreases in
    time this labels early observed times "high survival" and long follow-up
    "low survival", i.e. the reverse of the prognosis ordering for events and
    no outcome information for censored patients. Kept for reproducibility
    of the submitted numbers only.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], event_observed=df["event"])
    df = df.copy()
    df["surv_prob"] = kmf.survival_function_at_times(df["time"]).values
    if legacy:
        score = df["surv_prob"]
        pool = df
    else:
        df["outcome_pct"] = 1.0 - df["surv_prob"]
        pool = df[df["event"] == 1]
        score = pool["outcome_pct"]
    low_cut = score.quantile(0.33)
    high_cut = score.quantile(0.66)

    def categorize(p):
        if p <= low_cut:
            return "low"
        if p <= high_cut:
            return "intermediate"
        return "high"

    pool = pool.copy()
    pool["category"] = (pool["surv_prob"] if legacy else pool["outcome_pct"]).apply(categorize)
    return pool.reset_index(drop=True)


def select_for_dataset(dataset: str, seed: int = 0,
                       narrative_model: str = "gpt-4o", grader_model: str = "gpt-4o",
                       dev_only: bool = True, events_only: bool = True,
                       legacy_labels: bool = False,
                       briefing_mode: str = "sanitized",
                       workers: int = 8, reuse_candidates: bool = False):
    cohort = COHORTS[dataset]
    briefing = _select_briefing(cohort, briefing_mode)
    os.makedirs(PROMPTS_DIR, exist_ok=True)

    df = cohort.load_full()
    if dev_only:
        # Corrected protocol: KM fit, tertile cut-points and the grading subset
        # all come from the fixed development hold-out, which the evaluation
        # loaders pin to the training fold of every split.
        df = df.iloc[dev_holdout_positions(df)].reset_index(drop=True)
        print(f"[{dataset}] prompt selection restricted to the {len(df)}-patient "
              f"development hold-out", flush=True)
    if legacy_labels:
        pool = _km_categories(df, legacy=True)
        if events_only:
            pool = pool[pool["event"] == 1]
        print(f"[{dataset}] LEGACY (as-submitted, inverted) labels: grading pool has "
              f"{len(pool)} patients", flush=True)
    else:
        pool = _km_categories(df)
        med = pool.groupby("category")["time"].median().to_dict()
        print(f"[{dataset}] corrected labels (events only, tertiles of 1-S_hat(T_i)): "
              f"grading pool has {len(pool)} events; median time low/int/high = "
              f"{med.get('low', float('nan')):.0f}/{med.get('intermediate', float('nan')):.0f}/"
              f"{med.get('high', float('nan')):.0f}", flush=True)
    df_eval = pool.sample(min(cohort.n_eval_samples, len(pool)), random_state=seed).copy()

    # 1) candidate target prompts
    tp_path = os.path.join(PROMPTS_DIR, f"{dataset}_target_prompts.json")
    if reuse_candidates and os.path.exists(tp_path):
        with open(tp_path, encoding="utf-8") as f:
            valid = json.load(f)
        print(f"[{dataset}] reusing {len(valid)} candidate prompts from {tp_path}", flush=True)
    else:
        print(f"[{dataset}] generating {cohort.n_candidates} candidate prompts ...", flush=True)
        all_templates = generate_target_prompts(
            cohort.description, cohort.feature_list, n=cohort.n_candidates, model=narrative_model)
        valid = _valid_templates(all_templates)
        with open(tp_path, "w", encoding="utf-8") as f:
            json.dump(valid, f, indent=2, ensure_ascii=False)
        print(f"[{dataset}] kept {len(valid)} valid prompts -> {tp_path}", flush=True)

    # 2) grade each candidate on the evaluation subset.
    # Primary criterion (manuscript Sec 3.2.2): grader label vs KM tertile label (phi).
    # Secondary, at no extra API cost in v5: ordinal agreement of the numeric
    # 5-year survival estimate (tertile-ranked within the graded subset) with
    # the same KM labels. Reported alongside; selection uses the primary.
    def _grade_one(text_template, row):
        features_block = render_features_block(cohort.sample_dict(row), fmt="bullet")
        filled = fill_template(text_template, features_block, briefing=briefing)
        narrative, numerics = generate_narrative_and_numerics(filled, model=narrative_model)
        s = score(narrative, row["category"], model=grader_model)
        prob = numerics["estimated_5yr_survival_prob"] if numerics is not None else None
        return s, prob, row["category"]

    rows = [row for _, row in df_eval.iterrows()]
    grades, numeric_grades = {}, {}
    # Resumability: grades are checkpointed after every candidate so an
    # interrupted run (the machine's memory watchdog, a network blip) resumes
    # from the last graded candidate instead of re-spending API calls.
    ckpt_path = os.path.join(PROMPTS_DIR, f"{dataset}_grades.partial.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            ck = json.load(f)
        grades.update({k: int(v) for k, v in ck.get("grades", {}).items()})
        numeric_grades.update({k: int(v) for k, v in ck.get("numeric_grades", {}).items()})
        print(f"[{dataset}] resuming: {len(grades)} candidates already graded "
              f"(checkpoint {ckpt_path})", flush=True)
    for tid, template in valid.items():
        if tid in grades:
            continue
        text_template = template["template"]
        total = 0
        probs, cats = [], []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_grade_one, text_template, row) for row in rows]
            for fut in as_completed(futures):
                s, prob, cat = fut.result()
                total += s
                if prob is not None:
                    probs.append(prob)
                    cats.append(cat)
        grades[tid] = total
        numeric_grades[tid] = _numeric_ordinal_grade(probs, cats)
        print(f"[{dataset}] prompt {tid}: grade={total} numeric_grade={numeric_grades[tid]}",
              flush=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({"grades": grades, "numeric_grades": numeric_grades}, f, indent=2)

    tg_path = os.path.join(PROMPTS_DIR, f"{dataset}_templates_grades.json")
    with open(tg_path, "w", encoding="utf-8") as f:
        json.dump(grades, f, indent=2, ensure_ascii=False)
    ng_path = os.path.join(PROMPTS_DIR, f"{dataset}_numeric_grades.json")
    with open(ng_path, "w", encoding="utf-8") as f:
        json.dump(numeric_grades, f, indent=2, ensure_ascii=False)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    best_id = max(grades, key=grades.get) if grades else None
    print(f"[{dataset}] grades saved -> {tg_path} (numeric secondary -> {ng_path})")
    if best_id is not None:
        print(f"[{dataset}] selected prompt P* = id {best_id} (grade {grades[best_id]}), "
              f"role={valid[best_id]['role']}, structure={valid[best_id]['structure']}")
    return valid, grades


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(COHORTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--narrative-model", default="gpt-4o")
    p.add_argument("--grader-model", default="gpt-4o")
    p.add_argument("--events-only", action="store_true", default=True,
                   help="(default, kept for compatibility) restrict the graded subset to "
                        "observed events; only meaningful together with --legacy-labels")
    p.add_argument("--legacy-labels", action="store_true",
                   help="reproduce the as-submitted reference labels (tertiles of S_hat(T_i) "
                        "on all development patients, which are inverted for events); "
                        "for comparison only")
    p.add_argument("--legacy-full-cohort", action="store_true",
                   help="reproduce the as-submitted behaviour (KM and grading "
                        "subset drawn from the full cohort); leaks test outcomes "
                        "into prompt selection -- for comparison only")
    p.add_argument("--briefing", choices=["sanitized", "legacy", "none"], default="sanitized",
                   help="cohort briefing filled into {BRIEFING} during grading: "
                        "'sanitized' (default; no cohort-matched outcome rates), "
                        "'legacy' (v4 text incl. published cohort figures), "
                        "'none' (empty slot)")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent grading calls per candidate (thread pool)")
    p.add_argument("--reuse-candidates", action="store_true",
                   help="reuse an existing <dataset>_target_prompts.json instead of "
                        "regenerating candidates")
    args = p.parse_args()
    select_for_dataset(args.dataset, seed=args.seed,
                       narrative_model=args.narrative_model, grader_model=args.grader_model,
                       dev_only=not args.legacy_full_cohort, events_only=args.events_only,
                       legacy_labels=args.legacy_labels,
                       briefing_mode=args.briefing, workers=args.workers,
                       reuse_candidates=args.reuse_candidates)
