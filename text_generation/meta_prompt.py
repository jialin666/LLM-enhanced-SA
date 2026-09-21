"""Meta prompt and candidate target-prompt generation (manuscript Sec 3.2.1).

A single meta prompt ``P_meta`` defines the desired space of target prompts and,
queried through several independent sampling runs, produces a diverse set of
candidate target prompts ``C = {P_1, ..., P_M}``. Each candidate is a full
instruction template that contains two literal placeholders -- ``{BRIEFING}``
(cohort background knowledge) and ``{FEATURES}`` (one patient's covariates) --
and, when filled, instructs the LLM to produce BOTH a clinical narrative Z_i
and a JSON block with the numeric prognostic estimates N_i in a single call
(v5 unified design).

Also provides the helpers to render a patient's features into the ``{FEATURES}``
block, fill a template, and generate the narrative + numerics from a filled
template.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Literal, Optional, Tuple

from openai import OpenAI


# Text inserted into {BRIEFING} when generation runs briefing-free
# (the R2-M3 no-briefing ablation arm).
BRIEFING_EMPTY = "(no additional background provided)"

# The three numeric fields consumed downstream (N_i).
NUMERIC_FIELDS = ("estimated_5yr_survival_prob", "estimated_2yr_event_risk", "confidence")

#: Cohorts whose follow-up is far shorter than the default 5-year / 2-year
#: horizons, so that both anchors would fall outside the observable window.
#: Values are ``(short_label, long_label)`` used to rename the two JSON keys.
#: Heart Failure: max follow-up 285 days (median event time 44 days), so the
#: default prompt asks for estimates the cohort can never observe. 90d/180d
#: bracket 69 and 89 of its 96 events respectively.
COHORT_HORIZONS = {
    # follow-up 285 d, median event 44 d; 90d/180d bracket 69 and 89 of 96 events
    "heartfailure": ("90d", "180d"),
    # follow-up 364 d, median event 91 d; 90d/270d bracket 46 and 94 of 96 events
    "aids": ("90d", "270d"),
}


def horizon_fields(short_label: str, long_label: str):
    """JSON key names for a cohort-matched pair of horizons.

    Returned in the same order as ``NUMERIC_FIELDS`` (long-horizon survival,
    short-horizon event risk, confidence) so the storage schema is unchanged.
    """
    return (f"estimated_{long_label}_survival_prob",
            f"estimated_{short_label}_event_risk",
            "confidence")


def retarget_horizons(template: str, short_label: str, long_label: str) -> str:
    """Rewrite a target prompt's numeric block to a different horizon pair."""
    new = horizon_fields(short_label, long_label)
    for old, nw in zip(NUMERIC_FIELDS, new):
        template = template.replace(old, nw)
    return template


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
            "Background knowledge for this patient population (use as context to inform your interpretation; "
            "your assessment must remain specific to this patient and must not simply restate these background figures):\n"
            "{{BRIEFING}}\n\n"
            "Write a comprehensive, fact-based narrative of 600-800 words that will be used as input to a survival analysis model.\n"
            "Your audience consists of medical researchers and machine learning experts, so maintain a professional and evidence-informed tone.\n\n"
            "Your narrative must:\n"
            "- Explain each clinical feature in its medical context (include what each feature means and its typical clinical interpretation).\n"
            "- Reason explicitly about how this patient's specific combination of features - including their interactions - shapes the prognosis, "
            "referencing standard clinical thresholds and guidelines (e.g., staging cutoffs, ER/PR thresholds where applicable).\n"
            "- Provide a synthesized risk profile (low, intermediate, or high survival probability) with clear qualitative declarations about survival probability.\n"
            "- Be patient-specific: a patient with different feature values must receive a visibly different narrative and different numerical estimates.\n"
            "- Ensure all reasoning is evidence-based and avoids unverifiable claims.\n\n"
            "After the narrative, end your response with exactly one JSON code block:\n\n"
            "```json\n"
            "{{\n"
            '  "estimated_5yr_survival_prob": <float 0.0-1.0>,\n'
            '  "estimated_2yr_event_risk": <float 0.0-1.0>,\n'
            '  "confidence": <float 0.0-1.0>\n'
            "}}\n"
            "```\n\n"
            "The estimates must be consistent with your narrative and specific to this patient; do not default to cohort averages. "
            "Nothing may follow the JSON block.\n\n"
            "Constraints:\n"
            "- Do NOT mention patient ID, raw survival time, or actual observed outcome.\n"
            "- Do NOT invent values not present in {{FEATURES}}.\n\n"
            "Format the narrative as a coherent, well-structured report, with clear sections or paragraphs."
        ),
    },
    {
        "role": "Biostatistician preparing annotated training text for a survival model",
        "structure": "B",
        "template": (
            "You are a biostatistician who writes structured training narratives for survival models.\n\n"
            "Your task is to produce a detailed, 600-800 word, fact-based report that uses the patient information provided below, "
            "followed by numerical prognostic estimates. The report should be suitable for machine learning training and for review by medical experts.\n\n"
            "Background knowledge for this patient population (context only; your report and estimates must be patient-specific "
            "and must not simply repeat these background statements):\n"
            "{{BRIEFING}}\n\n"
            "Your report must:\n"
            "- Explain every feature, describing its domain meaning, measurement units, and prognostic implications.\n"
            "- Reason explicitly about the joint effect of this patient's feature values, including interactions among features, "
            "connecting them to risk stratification and proportional-risk intuition.\n"
            "- Provide a reasoned survival probability assessment (low, intermediate, or high) and justify it explicitly.\n"
            "- Incorporate medically/scientifically recognized knowledge beyond raw values, such as thresholds and typical treatment effects.\n"
            "- Be patient-specific: different feature values must lead to visibly different reports and different estimates.\n"
            "- Remain strictly fact-based and professional.\n\n"
            "End your response with exactly one JSON code block containing your numerical estimates, consistent with the report:\n\n"
            "```json\n"
            "{{\n"
            '  "estimated_5yr_survival_prob": <float 0.0-1.0>,\n'
            '  "estimated_2yr_event_risk": <float 0.0-1.0>,\n'
            '  "confidence": <float 0.0-1.0>\n'
            "}}\n"
            "```\n\n"
            "Nothing may follow the JSON block.\n\n"
            "Constraints:\n"
            "- Forbid mention of patient identifiers, observed survival times, or actual outcomes.\n"
            "- Do not speculate beyond established evidence.\n\n"
            "Clinical Features:\n{{FEATURES}}"
        ),
    },
]


# ---------------------------------------------------------------------------
# Meta prompt (defines the target-prompt space; eleven required components)
# ---------------------------------------------------------------------------

META_PROMPT_TEMPLATE = """\
You are an advanced prompt designer for clinical machine-learning applications.

GOAL
Generate a wide variety of TARGET PROMPTS. Each TARGET PROMPT must itself be a detailed instruction template
of approximately 600-1200 words, containing all ELEVEN required components below, and must include two literal
placeholders: {{BRIEFING}} for cohort background knowledge and {{FEATURES}} for per-sample feature insertion.
When later filled with a cohort knowledge briefing and one patient's structured covariates, a TARGET PROMPT
instructs an LLM to produce BOTH (a) a comprehensive, fact-based clinical narrative and (b) a structured block
of numerical prognostic estimates. These outputs will be used as inputs to survival analysis models.

INPUTS
Dataset Description:
{dataset_description}

Feature List:
{feature_list}

REQUIREMENTS (All Eleven Components Must Appear in Each TARGET PROMPT; order/layout is flexible):
1) Role Assignment - specify a professional role with both clinical and analytical expertise
   (e.g., oncologist specializing in prognosis, intensivist, biostatistician developing clinical models,
   clinical research coordinator, guideline author). Prefer roles that emphasize evidence-based clinical
   reasoning and population-level prognosis.
2) Narrative Instruction - explicitly require a 600-800 word narrative; coherent, structured, clinically realistic.
3) Content Requirements - require:
- explanation of each feature (name/units/clinical meaning),
- explicit prognostic REASONING that combines features: how this specific combination of values, including
  interactions among features, maps to risk - referencing recognized thresholds, staging conventions,
  clinical guidelines, and established prognostic factors from the published literature,
- a synthesized survival risk profile (low/intermediate/high survival probability) with clear qualitative
  declarations about survival probability,
- linkage to survival-analysis concepts (risk stratification, proportional-risk intuition,
  short-term versus long-term hazard),
- patient-specificity: two patients with different features must yield visibly different narratives and
  different numerical estimates; forbid generic cohort-average summaries,
- all reasoning fact-based and verifiable; no speculation.
4) Numerical Prognostic Output - require the response to END with exactly ONE JSON code block of the form:
```json
{{
  "estimated_5yr_survival_prob": <float between 0.0 and 1.0>,
  "estimated_2yr_event_risk": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>
}}
```
   The estimates must be consistent with the narrative's reasoning and specific to this patient's feature
   combination - they must NOT default to cohort averages. The confidence value expresses how well this
   feature combination is covered by established clinical knowledge (lower for unusual or conflicting
   feature combinations).
5) Cohort Briefing Placeholder - include the literal token {{BRIEFING}}, introduced as cohort background
   knowledge, together with an explicit usage instruction: the briefing is BACKGROUND context to enrich
   interpretation, not a substitute for patient-specific reasoning; the narrative and estimates must not
   simply restate briefing-level averages.
6) Constraints - forbid patient identifiers, raw survival times, true observed outcomes; forbid inventing
   feature values not present in {{FEATURES}}; forbid off-topic content.
7) Feature Placeholder - include the literal token {{FEATURES}} and instruct to use ONLY that block for
   per-sample values.
8) Structure Variation - support Type A ({{FEATURES}} before), Type B ({{FEATURES}} after),
   Type C (brief integrated case + {{FEATURES}}), Type D (alternate formats like JSON/table/vignette).
9) Tone & Audience - professional, evidence-based, for clinical experts and/or ML models.
10) Output Contract - the narrative comes FIRST, the single JSON code block comes LAST, and nothing may
    follow the JSON block.
11) Target-Prompt Format - return a JSON object with fields: "role", "structure" (A|B|C|D),
    "template" (the full text, containing the literal {{BRIEFING}} and {{FEATURES}}).

FEW-SHOT EXAMPLES (good TARGET PROMPTS):
{fewshot_json}

OUTPUT FORMAT (strict JSON):
{{
"prompts": [
    {{
    "role": "string",
    "structure": "A|B|C|D",
    "template": "FULL TARGET PROMPT TEXT that contains the literal {{BRIEFING}} and {{FEATURES}}"
    }}
]
}}

TASK
Generate N diverse TARGET PROMPTS that satisfy all eleven components, vary roles and structures (A/B/C/D),
and keep {{BRIEFING}} and {{FEATURES}} exactly literal. JSON only, no extra commentary.
"""


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_target_prompts(dataset_description: str, feature_list: str,
                            n: int = 50, model: str = "gpt-4o") -> List[Dict]:
    """Query the LLM with the meta prompt to obtain ``n`` candidate target prompts.

    The LLM returns 5 prompts per call, so ``ceil(n / 5)`` calls are made. Only
    prompts that contain both literal placeholders are kept.
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
            tpl = p.get("template", "") if isinstance(p, dict) else ""
            if "{FEATURES}" in tpl and "{BRIEFING}" in tpl:
                target_prompts.append({
                    "role": p.get("role", ""),
                    "structure": p.get("structure", ""),
                    "template": tpl,
                })
    return target_prompts


# ---------------------------------------------------------------------------
# Rendering a patient's features and generating narrative + numerics
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


def fill_template(template_text: str, features_block: str,
                  briefing: str = BRIEFING_EMPTY) -> str:
    """Fill both literal placeholders.

    Uses ``str.replace`` (not ``str.format``): v5 templates legitimately
    contain JSON braces in the numeric-output contract, which ``format``
    would choke on.
    """
    if "{FEATURES}" not in template_text:
        raise ValueError("Template must contain the literal {FEATURES} placeholder.")
    filled = template_text.replace("{FEATURES}", features_block)
    # Older (v1) templates carry no briefing slot; fill it when present.
    if "{BRIEFING}" in filled:
        filled = filled.replace("{BRIEFING}", briefing or BRIEFING_EMPTY)
    return filled


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\}\s*$", re.S)


def parse_narrative_response(text: str, fields=None) -> Tuple[str, Optional[Dict[str, float]]]:
    """Split a v5 response into (narrative, numerics dict or None).

    The numerics are taken from the LAST fenced JSON block; if none is found,
    a trailing bare JSON object is tried. Values are clamped to [0, 1]. The
    narrative is the response with the numeric block removed.
    """
    m = None
    for m in _JSON_FENCE_RE.finditer(text):
        pass  # keep the last fenced block
    raw = None
    if m is not None:
        raw = m.group(1)
        narrative = (text[:m.start()] + text[m.end():]).strip()
    else:
        tail = _BARE_JSON_RE.search(text)
        if tail is not None:
            raw = tail.group(0)
            narrative = text[:tail.start()].strip()
        else:
            narrative = text.strip()
    if raw is None:
        return narrative, None
    try:
        obj = json.loads(raw)
    except Exception:
        return narrative, None
    out: Dict[str, float] = {}
    # ``fields`` lets a cohort ask for different horizons; results are always
    # keyed back to the canonical NUMERIC_FIELDS slots so every downstream
    # consumer (column names, loaders, ablations) is unaffected.
    read = tuple(fields) if fields else NUMERIC_FIELDS
    for f, src in zip(NUMERIC_FIELDS, read):
        v = obj.get(src)
        if v is None:
            return narrative, None
        try:
            out[f] = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            return narrative, None
    return narrative, out


def generate_narrative_and_numerics(
    filled_prompt: str, model: str = "gpt-4o", temperature: float = 0.4,
    retries: int = 1, api_retries: int = 4, fields=None,
) -> Tuple[str, Optional[Dict[str, float]]]:
    """One call producing the narrative and the numeric JSON block (v5).

    Retries once if the numeric block cannot be parsed; on repeated failure
    returns the narrative with ``None`` numerics (callers substitute the 0.5
    defaults and log the patient). Transient API errors (rate limits,
    timeouts) are retried with exponential backoff.
    """
    import random
    import time as _t

    client = _client()
    last_narrative = ""
    for _ in range(retries + 1):
        rsp = None
        for attempt in range(api_retries + 1):
            try:
                rsp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You generate fact-based clinical narratives with numerical prognostic "
                                    "estimates, suitable for ML training. Follow the prompt's output contract "
                                    "exactly: narrative first, then a single JSON code block, nothing after it. "
                                    "Avoid unverifiable claims."},
                        {"role": "user", "content": filled_prompt},
                    ],
                    temperature=temperature,
                )
                break
            except Exception:
                if attempt == api_retries:
                    raise
                _t.sleep(2.0 * (2 ** attempt) + random.uniform(0, 1))
        narrative, numerics = parse_narrative_response(
            rsp.choices[0].message.content.strip(), fields=fields)
        last_narrative = narrative
        if numerics is not None:
            return narrative, numerics
    return last_narrative, None


def generate_narrative(filled_prompt: str, model: str = "gpt-4o") -> str:
    """Narrative-only helper kept for backward compatibility (v1 pipeline)."""
    narrative, _ = generate_narrative_and_numerics(filled_prompt, model=model, temperature=0.4)
    return narrative
