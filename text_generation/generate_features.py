"""Final feature generation with the selected prompt (manuscript Sec 2.1-2.2).

Using the selected target prompt P* -- either the highest-graded candidate from
a local ``select_prompt.py`` run, or a specific one chosen with ``--prompt-id``
from ``prompts/<dataset>_target_prompts.json`` -- generate, for every patient in
a cohort:

  * the free-text clinical narrative  Z_i = LLM(X_i | P*),
  * the numeric prognostic estimates  N_i = (p_5y, r_2y, c)   (a constrained
    JSON object grounded in the cohort knowledge briefing), and
  * the dense narrative embedding      E_i = EMB(Z_i) in R^1536.

Outputs (under data/, consumed by llmsa.data.load_dataset_v4):
    data/v4_structured_<dataset>.csv    patient_idx + the three N_i fields
    data/embeddings_v4_<dataset>.npy    (n, 1536) narrative embeddings E_i
    data/v4/narratives/<dataset>/       one .txt narrative per patient (cache)

Usage:
    python -m text_generation.generate_features --dataset gbsg
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from openai import OpenAI

from llmsa.data import (
    _csv_path, _read_cohort_csv, _support_subsample, SUPPORT_SUBSAMPLE_SIZE,
)
from .cohorts import COHORTS
from .meta_prompt import render_features_block, fill_template, generate_narrative

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


# ---------------------------------------------------------------------------
# Numeric prognostic estimates N_i (constrained JSON)
# ---------------------------------------------------------------------------

JSON_SYSTEM = """\
You are a senior clinician with deep knowledge of published cohort literature \
for {disease}. You produce per-patient quantitative prognostic estimates that \
DRAW ON published cohort data while remaining specific to the individual patient.

{briefing}

Use the above as background. Your estimates must be PATIENT-SPECIFIC, not just \
averages of the cohort context. Output STRICT JSON only.
"""

JSON_USER = """\
Patient features:
{feature_block}

Produce a STRICT JSON object with these fields:
{{
  "estimated_5yr_survival_prob": <float 0.0-1.0>,
  "estimated_2yr_event_risk": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>
}}

Constraints:
- Estimates must be DIFFERENT across patients with different features; do NOT
  default to cohort averages.
- Do not mention identifiers or observed outcomes.
"""


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _gen_numeric(client, cohort, features_block, model="gpt-4o", retries=2):
    system = JSON_SYSTEM.format(disease=cohort.name, briefing=cohort.briefing)
    user = JSON_USER.format(feature_block=features_block)
    for attempt in range(retries + 1):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            return json.loads(rsp.choices[0].message.content)
        except Exception:
            if attempt == retries:
                return {}


def _embed(client, texts, model="text-embedding-3-small"):
    out = []
    BATCH = 64
    for k in range(0, len(texts), BATCH):
        chunk = [t or " " for t in texts[k:k + BATCH]]
        rsp = client.embeddings.create(model=model, input=chunk)
        out.extend(np.asarray(d.embedding, dtype=np.float32) for d in rsp.data)
    return np.stack(out, axis=0)


def _load_selected_prompt(dataset: str, prompt_id=None) -> str:
    """Return the target-prompt template to use for generation.

    ``prompt_id`` selects a specific candidate from ``<dataset>_target_prompts.json``.
    If not given, the selection defers to a local grades file
    ``<dataset>_templates_grades.json`` (written by ``select_prompt.py``) and the
    highest-graded candidate P* is used. Grades are a per-run artefact and are not
    shipped with the repository.
    """
    with open(os.path.join(PROMPTS_DIR, f"{dataset}_target_prompts.json")) as f:
        prompts = json.load(f)

    if prompt_id is not None:
        key = str(prompt_id)
        if key not in prompts:
            raise KeyError(f"prompt id {key!r} not in {dataset}_target_prompts.json "
                           f"(available: {list(prompts)[:5]}...)")
        return prompts[key]["template"]

    grades_path = os.path.join(PROMPTS_DIR, f"{dataset}_templates_grades.json")
    if not os.path.exists(grades_path):
        raise FileNotFoundError(
            f"No selection found for {dataset}. Either run "
            f"`python -m text_generation.select_prompt --dataset {dataset}` to grade "
            f"the candidates and select P*, or pass --prompt-id to choose one directly."
        )
    with open(grades_path) as f:
        grades = json.load(f)
    best_id = max(grades, key=grades.get)
    return prompts[best_id]["template"]


def _indexed_cohort(dataset: str):
    """Full cohort dataframe with a stable ``_patient_idx`` matching load_dataset_v4."""
    cohort = COHORTS[dataset]
    df = _read_cohort_csv(dataset)
    df = df[cohort._cat + cohort._num + ["time", "event"]]
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int).clip(0, 1)
    df["_patient_idx"] = df.index.copy()
    if dataset == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE:
        df = _support_subsample(df)
    return cohort, df


def generate_for_dataset(dataset: str, narrative_model="gpt-4o",
                         numeric_model="gpt-4o", embedding_model="text-embedding-3-small",
                         max_patients=None, prompt_id=None):
    cohort, df = _indexed_cohort(dataset)
    if max_patients is not None:
        df = df.iloc[:max_patients]
    template = _load_selected_prompt(dataset, prompt_id=prompt_id)
    client = _client()

    narr_dir = _csv_path("data", "v4", "narratives", dataset)
    os.makedirs(narr_dir, exist_ok=True)

    rows, narratives = [], []
    n = len(df)
    for k, (_, row) in enumerate(df.iterrows()):
        pi = int(row["_patient_idx"])
        features_block = render_features_block(cohort.sample_dict(row), fmt="bullet")

        # Narrative Z_i via the selected prompt P* (cached to disk).
        npath = os.path.join(narr_dir, f"patient_{pi:05d}.txt")
        if os.path.exists(npath):
            with open(npath) as f:
                narrative = f.read()
        else:
            narrative = generate_narrative(fill_template(template, features_block), model=narrative_model)
            with open(npath, "w") as f:
                f.write(narrative)
        narratives.append(narrative)

        # Numeric estimates N_i.
        j = _gen_numeric(client, cohort, features_block, model=numeric_model)
        rows.append({
            "patient_idx": pi,
            "llm_5yr_surv_v4": float(j.get("estimated_5yr_survival_prob", 0.5)),
            "llm_2yr_event_risk_v4": float(j.get("estimated_2yr_event_risk", 0.5)),
            "llm_confidence_v4": float(j.get("confidence", 0.5)),
        })
        if (k + 1) % 25 == 0 or k + 1 == n:
            print(f"  [{dataset}] {k + 1}/{n}", flush=True)

    struct_out = _csv_path("data", f"v4_structured_{dataset}.csv")
    pd.DataFrame(rows).to_csv(struct_out, index=False)
    print(f"[{dataset}] structured N_i ({len(rows)} rows) -> {struct_out}")

    emb = _embed(client, narratives, model=embedding_model)
    emb_out = _csv_path("data", f"embeddings_v4_{dataset}.npy")
    np.save(emb_out, emb)
    print(f"[{dataset}] embeddings E_i {emb.shape} -> {emb_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(COHORTS))
    p.add_argument("--narrative-model", default="gpt-4o")
    p.add_argument("--numeric-model", default="gpt-4o")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--max-patients", type=int, default=None)
    p.add_argument("--prompt-id", default=None,
                   help="candidate id from <dataset>_target_prompts.json to use; "
                        "if omitted, the highest-graded prompt from a local "
                        "select_prompt run (P*) is used")
    args = p.parse_args()
    generate_for_dataset(
        args.dataset, narrative_model=args.narrative_model,
        numeric_model=args.numeric_model, embedding_model=args.embedding_model,
        max_patients=args.max_patients, prompt_id=args.prompt_id,
    )
