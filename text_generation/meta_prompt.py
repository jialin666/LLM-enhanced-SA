"""Meta prompt and candidate target-prompt generation (manuscript Sec 2.2.1).

A single meta prompt ``P_meta`` defines the desired space of target prompts and,
queried through several independent sampling runs, produces a diverse set of
candidate target prompts ``C = {P_1, ..., P_M}``. Each candidate is a full
instruction template that contains the literal ``{FEATURES}`` placeholder and,
when filled with a patient's covariates, instructs the LLM to write a clinical
narrative for that patient.

Also provides the helpers to render a patient's features into the ``{FEATURES}``
block, fill a template, and generate a narrative from a filled template.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Literal

from openai import OpenAI


# ---------------------------------------------------------------------------
# Few-shot example target prompts (Type A and Type B)
# ---------------------------------------------------------------------------

FEWSHOT_EXAMPLES: List[Dict] = [
    {
        "role": "Oncologist specializing in survival prediction",
        "structure": "A",
        "template": (
            "Clinical Features:\n{{FEATURES}}\n\n"
            "You are an oncologist specializing in survival prediction and patient prognosis.\n\n"
            "Write a comprehensive, fact-based narrative of 600-800 words that will be used as input to a survival analysis model.\n"
            "Your audience consists of medical researchers and machine learning experts, so maintain a professional and evidence-informed tone.\n\n"
            "Your narrative must:\n"
            "- Explain each clinical feature in its medical context (include what each feature means and its typical clinical interpretation).\n"
            "- Provide a synthesized risk profile (low, intermediate, or high survival probability) based on the features.\n"
            "- Include clear qualitative declarations about survival probability (e.g., 'The presence of multiple positive nodes suggests a significantly lower survival probability').\n"
            "- Discuss how each feature contributes to the prognosis, referencing standard clinical thresholds and guidelines (e.g., staging cutoffs, ER/PR thresholds where applicable).\n"
            "- Ensure all reasoning is evidence-based and avoids unverifiable claims.\n\n"
            "Constraints:\n"
            "- Do NOT mention patient ID, raw survival time, or actual observed outcome.\n"
            "- Do NOT invent values not present in {{FEATURES}}.\n\n"
            "Format the output as a coherent, well-structured report, with clear sections or paragraphs that could be used directly for model training."
        ),
    },
    {
        "role": "Biostatistician preparing annotated training text for a survival model",
        "structure": "B",
        "template": (
            "You are a biostatistician who writes structured training narratives for survival models.\n\n"
            "Your task is to produce a detailed, 600-800 word, fact-based report that uses the patient information provided below. "
            "The report should be suitable for machine learning training and for review by medical experts.\n\n"
            "Your report must:\n"
            "- Explain every feature, describing its domain meaning, measurement units, and prognostic implications.\n"
            "- Provide a reasoned survival probability assessment (low, intermediate, or high) and justify it explicitly.\n"
            "- Include qualitative statements about prognosis (e.g., 'The combination of favorable biomarkers and small tumor size suggests a higher survival probability').\n"
            "- Connect your reasoning to survival analysis principles, such as risk stratification and proportional-risk intuition.\n"
            "- Incorporate medically/scientifically recognized knowledge beyond raw values, such as thresholds and typical treatment effects.\n"
            "- Remain strictly fact-based and professional.\n\n"
            "Constraints:\n"
            "- Forbid mention of patient identifiers, observed survival times, or actual outcomes.\n"
            "- Do not speculate beyond established evidence.\n\n"
            "Clinical Features:\n{{FEATURES}}"
        ),
    },
]


# ---------------------------------------------------------------------------
# Meta prompt (defines the target-prompt space; nine required components)
# ---------------------------------------------------------------------------

META_PROMPT_TEMPLATE = """\
You are an advanced prompt designer.

GOAL
Generate a wide variety of TARGET PROMPTS. Each TARGET PROMPT must itself be a detailed instruction template
of approximately 600-1200 words, containing all required components, and must include the literal placeholder {{FEATURES}}
for per-sample feature insertion. These TARGET PROMPTS will later be used to generate comprehensive, fact-based narratives
for survival analysis.

INPUTS
Dataset Description:
{dataset_description}

Feature List:
{feature_list}

REQUIREMENTS (All Nine Components Must Appear in Each TARGET PROMPT; order/layout is flexible):
1) Role Assignment - specify a professional role (e.g., clinician, oncologist, survival analysis expert, biostatistician, educator, guideline author).
2) Narrative Instruction - explicitly require a 600-800 word narrative; coherent, structured, clinically realistic.
3) Content Requirements - require:
- explanation of each feature (name/units/meaning),
- a synthesized survival risk profile (low/intermediate/high),
- qualitative declarations about survival probability,
- discussion of how features influence survival probability,
- linkage to survival concepts (risk stratification, proportional-risk intuition, staging, etc.),
- Inject additional medically and scientifically supported knowledge as much as possible - bring in thresholds, clinical guidelines, known prognostic factors,
    and plausible interactions beyond the raw features. These additional insights are expected to enhance the informativeness of the text
    and improve the performance of survival analysis models.
- State explicitly that all reasoning must be fact-based and verifiable.
4) Constraints - forbid IDs, raw survival times, true observed outcomes; forbid speculation and off-topic content.
5) Feature Placeholder - include the literal token {{FEATURES}} and instruct to use ONLY that block for per-sample values.
6) Structure Variation - support Type A ({{FEATURES}} before), Type B ({{FEATURES}} after), Type C (brief integrated case + {{FEATURES}}),
   Type D (alternate formats like JSON/table/vignette).
7) Tone & Audience - professional, evidence-based, for clinical experts and/or ML models.
8) Output Format (recommended) - return a JSON object with fields: "role", "structure", "template" (template contains {{FEATURES}}).

FEW-SHOT EXAMPLES (good TARGET PROMPTS):
{fewshot_json}

OUTPUT FORMAT (strict JSON):
{{
"prompts": [
    {{
    "role": "string",
    "structure": "A|B|C|D",
    "template": "FULL TARGET PROMPT TEXT that contains the literal {{FEATURES}}"
    }}
]
}}

TASK
Generate N diverse TARGET PROMPTS that satisfy all nine components, vary roles and structures (A/B/C/D),
and keep {{FEATURES}} exactly literal. JSON only, no extra commentary.
"""


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_target_prompts(dataset_description: str, feature_list: str,
                            n: int = 50, model: str = "gpt-4o") -> List[Dict]:
    """Query the LLM with the meta prompt to obtain ``n`` candidate target prompts.

    The LLM returns 5 prompts per call, so ``ceil(n / 5)`` calls are made. Only
    prompts that contain the literal ``{FEATURES}`` placeholder are kept.
    """
    fewshot_json = json.dumps(FEWSHOT_EXAMPLES, ensure_ascii=False, indent=2)
    meta_prompt = META_PROMPT_TEMPLATE.format(
        dataset_description=dataset_description.strip(),
        feature_list=feature_list.strip(),
        fewshot_json=fewshot_json,
    )
    client = _client()
    user_payload = f"Generate 5 TARGET PROMPTS.\n\nMETA PROMPT:\n---\n{meta_prompt}\n---"
    target_prompts: List[Dict] = []
    iters = n // 5 if n % 5 == 0 else n // 5 + 1
    for _ in range(iters):
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You produce high-quality TARGET PROMPTS that strictly follow the meta prompt."},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )
        data = json.loads(rsp.choices[0].message.content)
        for p in data.get("prompts", []):
            if isinstance(p, dict) and "{FEATURES}" in p.get("template", ""):
                target_prompts.append({
                    "role": p.get("role", ""),
                    "structure": p.get("structure", ""),
                    "template": p.get("template", ""),
                })
    return target_prompts


# ---------------------------------------------------------------------------
# Rendering a patient's features and generating a narrative from a template
# ---------------------------------------------------------------------------

def render_features_block(
    features: Dict[str, object],
    fmt: Literal["bullet", "table", "json", "narrative"] = "bullet",
) -> str:
    if fmt == "bullet":
        lines = ["Clinical Features:"]
        for k, v in features.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
    if fmt == "table":
        header = "Clinical Features (Table):\nFeature\tValue"
        rows = [f"{k}\t{v}" for k, v in features.items()]
        return "\n".join([header] + rows)
    if fmt == "json":
        return "Clinical Features (JSON):\n" + json.dumps(features, indent=2, ensure_ascii=False)
    if fmt == "narrative":
        parts = [f"{k}={v}" for k, v in features.items()]
        return "Clinical Features (Narrative): " + ", ".join(parts) + "."
    raise ValueError("Unknown format")


def fill_template(template_text: str, features_block: str) -> str:
    if "{FEATURES}" not in template_text:
        raise ValueError("Template must contain the literal {FEATURES} placeholder.")
    return template_text.format(FEATURES=features_block)


def generate_narrative(filled_prompt: str, model: str = "gpt-4o") -> str:
    client = _client()
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You generate fact-based clinical narratives suitable for ML training. Keep 600-800 words and avoid unverifiable claims."},
            {"role": "user", "content": filled_prompt},
        ],
        temperature=0.4,
    )
    return rsp.choices[0].message.content.strip()
