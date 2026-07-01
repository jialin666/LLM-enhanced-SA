"""Optimal target-prompt selection (manuscript Sec 2.2.2).

For one cohort:

  1. Query the meta prompt to obtain a candidate set of target prompts and keep
     the well-formed ones (exactly one literal ``{FEATURES}`` placeholder).
  2. Build ordinal reference labels y_i in {low, intermediate, high} from a
     Kaplan-Meier fit: evaluate S_hat(T_i) at each observed time and split the
     range into three tertiles.
  3. On a random evaluation subset of patients, generate a narrative with each
     candidate prompt, pass it back to the LLM for a predicted label, and score
     the agreement with phi (+1/0/-1).
  4. The prompt score S(P_m) is the summed agreement over the subset. The
     selected prompt P* = argmax_m S(P_m) is used downstream.

Outputs (under prompts/):
    <dataset>_target_prompts.json   candidate prompts keyed by id
    <dataset>_templates_grades.json summed grade per candidate id (per-run;
                                    not tracked in git -- LLM outputs vary by run)

Usage:
    python -m text_generation.select_prompt --dataset gbsg
"""

from __future__ import annotations

import argparse
import json
import os
import re

from lifelines import KaplanMeierFitter

from .cohorts import COHORTS
from .meta_prompt import generate_target_prompts, render_features_block, fill_template, generate_narrative
from .score import score

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _valid_templates(all_templates):
    """Keep templates whose only placeholder is the literal {FEATURES}."""
    valid = {}
    key = 0
    for template in all_templates:
        text = template["template"]
        if "{{FEATURES}}" in text:
            text = text.replace("{{FEATURES}}", "{FEATURES}")
            template["template"] = text
        placeholders = re.findall(r"\{[^}]+\}", text)
        if all(p == "{FEATURES}" for p in placeholders):
            valid[key] = template
            key += 1
        else:
            print(f"  dropping template with invalid placeholders: {placeholders}")
    return valid


def _km_categories(df):
    """Assign each patient an ordinal KM category from S_hat(T_i) tertiles."""
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], event_observed=df["event"])
    df = df.copy()
    df["surv_prob"] = kmf.survival_function_at_times(df["time"]).values
    low_cut = df["surv_prob"].quantile(0.33)
    high_cut = df["surv_prob"].quantile(0.66)

    def categorize(p):
        if p <= low_cut:
            return "low"
        if p <= high_cut:
            return "intermediate"
        return "high"

    df["category"] = df["surv_prob"].apply(categorize)
    return df


def select_for_dataset(dataset: str, seed: int = 0,
                       narrative_model: str = "gpt-4o", grader_model: str = "gpt-4o"):
    cohort = COHORTS[dataset]
    os.makedirs(PROMPTS_DIR, exist_ok=True)

    df = cohort.load_full()
    df = _km_categories(df)
    df_eval = df.sample(min(cohort.n_eval_samples, len(df)), random_state=seed).copy()

    # 1) candidate target prompts
    print(f"[{dataset}] generating {cohort.n_candidates} candidate prompts ...", flush=True)
    all_templates = generate_target_prompts(
        cohort.description, cohort.feature_list, n=cohort.n_candidates, model=narrative_model)
    valid = _valid_templates(all_templates)
    tp_path = os.path.join(PROMPTS_DIR, f"{dataset}_target_prompts.json")
    with open(tp_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2, ensure_ascii=False)
    print(f"[{dataset}] kept {len(valid)} valid prompts -> {tp_path}", flush=True)

    # 2) grade each candidate on the evaluation subset
    grades = {}
    for tid, template in valid.items():
        text_template = template["template"]
        total = 0
        for _, row in df_eval.iterrows():
            features_block = render_features_block(cohort.sample_dict(row), fmt="bullet")
            filled = fill_template(text_template, features_block)
            narrative = generate_narrative(filled, model=narrative_model)
            total += score(narrative, row["category"], model=grader_model)
        grades[tid] = total
        print(f"[{dataset}] prompt {tid}: grade={total}", flush=True)

    tg_path = os.path.join(PROMPTS_DIR, f"{dataset}_templates_grades.json")
    with open(tg_path, "w", encoding="utf-8") as f:
        json.dump(grades, f, indent=2, ensure_ascii=False)

    best_id = max(grades, key=grades.get) if grades else None
    print(f"[{dataset}] grades saved -> {tg_path}")
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
    args = p.parse_args()
    select_for_dataset(args.dataset, seed=args.seed,
                       narrative_model=args.narrative_model, grader_model=args.grader_model)
