# -*- coding: utf-8 -*-
"""
Generate high-quality TARGET PROMPT TEMPLATES that include a literal {{FEATURES}} placeholder,
using a meta prompt + two few-shot examples; then (optionally) fill a template with per-sample
features and generate a 600–800 word narrative.

Prereqs:
  pip install --upgrade openai pandas
  export OPENAI_API_KEY=...
"""

import os, json, textwrap
from typing import Dict, List, Literal
from openai import OpenAI


# ---------------------------
# 0) Few-shot example TARGET PROMPTS (Type A & Type B)
# ---------------------------
FEWSHOT_EXAMPLES: List[Dict] = [
    {
        "role": "Oncologist specializing in survival prediction",
        "structure": "A",
        "template": (
            "Clinical Features:\n{{FEATURES}}\n\n"
            "You are an oncologist specializing in survival prediction and patient prognosis.\n\n"
            "Write a comprehensive, fact-based narrative of 600–800 words that will be used as input to a survival analysis model.\n"
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
        )
    },
    {
        "role": "Biostatistician preparing annotated training text for a survival model",
        "structure": "B",
        "template": (
            "You are a biostatistician who writes structured training narratives for survival models.\n\n"
            "Your task is to produce a detailed, 600–800 word, fact-based report that uses the patient information provided below. "
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
        )
    }
]


# ---------------------------
# 1) Meta prompt template (requires 9 components but allows flexible assembly)
# ---------------------------
META_PROMPT_TEMPLATE = """\
You are an advanced prompt designer.

GOAL
Generate a wide variety of TARGET PROMPTS. Each TARGET PROMPT must itself be a detailed instruction template
of approximately 600–1200 words, containing all required components, and must include the literal placeholder {{FEATURES}}
for per-sample feature insertion. These TARGET PROMPTS will later be used to generate comprehensive, fact-based narratives
for survival analysis.

INPUTS
Dataset Description:
{dataset_description}

Feature List:
{feature_list}

REQUIREMENTS (All Nine Components Must Appear in Each TARGET PROMPT; order/layout is flexible):
1) Role Assignment — specify a professional role (e.g., clinician, oncologist, survival analysis expert, biostatistician, educator, guideline author).
2) Narrative Instruction — explicitly require a 600–800 word narrative; coherent, structured, clinically realistic.
3) Content Requirements — require:
• explanation of each feature (name/units/meaning),
• a synthesized survival risk profile (low/intermediate/high),
• qualitative declarations about survival probability,
• discussion of how features influence survival probability,
• linkage to survival concepts (risk stratification, proportional-risk intuition, staging, etc.),
• **Inject additional medically and scientifically supported knowledge as much as possible** — bring in thresholds, clinical guidelines, known prognostic factors,
    and plausible interactions beyond the raw features. These additional insights are expected to enhance the informativeness of the text
    and improve the performance of survival analysis models.
• State explicitly that all reasoning must be fact-based and verifiable.
4) Constraints — forbid IDs, raw survival times, true observed outcomes; forbid speculation and off-topic content.
5) Feature Placeholder — include the literal token {{FEATURES}} and instruct to use ONLY that block for per-sample values.
6) Structure Variation — support Type A ({{FEATURES}} before), Type B ({{FEATURES}} after), Type C (brief integrated case + {{FEATURES}}),
   Type D (alternate formats like JSON/table/vignette).
7) Tone & Audience — professional, evidence-based, for clinical experts and/or ML models.
8) Output Format (recommended) — return a JSON object with fields: "role", "structure", "template" (template contains {{FEATURES}}).

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


# ---------------------------
# 2) OpenAI helpers
# ---------------------------
def openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_target_prompts(dataset_description: str, feature_list: str, n: int = 5, model: str = "gpt-4o",) -> List[Dict]:
    """Call the OpenAI Responses API to generate JSON TARGET PROMPTS using the meta prompt + few-shot examples."""
    fewshot_json = json.dumps(FEWSHOT_EXAMPLES, ensure_ascii=False, indent=2)
    meta_prompt = META_PROMPT_TEMPLATE.format(
        dataset_description=dataset_description.strip(),
        feature_list=feature_list.strip(),
        fewshot_json=fewshot_json
    )

    client = openai_client()
    user_payload = f"Generate 5 TARGET PROMPTS.\n\nMETA PROMPT:\n---\n{meta_prompt}\n---"
    print(user_payload)
    target_prompts = []
    iters = n//5 if n%5 == 0 else n//5 + 1
    for i in range(iters):
        rsp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You produce high-quality TARGET PROMPTS that strictly follow the meta prompt."},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.5,
            # max_tokens=50000,
            response_format={"type": "json_object"},
        )
        text = rsp.choices[0].message.content
        data = json.loads(text)
        prompts = data.get("prompts", [])
        # Quick checks
        for p in prompts:
            if isinstance(p, dict) and "{FEATURES}" in p.get("template", ""):
                target_prompts.append({
                    "role": p.get("role", ""),
                    "structure": p.get("structure", ""),
                    "template": p.get("template", "")
                })
    return target_prompts


# ---------------------------
# 3) Filling a template with per-sample features & generating a narrative
# ---------------------------
def render_features_block(
    features: Dict[str, str | int | float],
    fmt: Literal["bullet", "table", "json", "narrative"] = "bullet"
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
    # return template_text.replace("{{FEATURES}}", features_block)
    return template_text.format(FEATURES=features_block)


def generate_narrative(filled_prompt: str, model: str = "gpt-4o") -> str:
    client = openai_client()
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You generate fact-based clinical narratives suitable for ML training. Keep 600–800 words and avoid unverifiable claims."},
            {"role": "user", "content": filled_prompt},
        ],
        temperature=0.4,
        # max_tokens=50000,
    )
    return rsp.choices[0].message.content.strip()


# ---------------------------
# 4) Minimal demo
# ---------------------------
if __name__ == "__main__":

    NUM_TEMPLATES = 6

    # 1) Describe your dataset & features (generic)
    # data description:
    DATASET_DESCRIPTION = """The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC) 
    is a large-scale international study of breast cancer patients. 
    It contains information on 1,980 patients with breast cancer, including their clinical attributes, 
    molecular data, and survival information. METABRIC uses gene and protein expression profiles to determine new
    breast cancer subgroups in order to help personalize treatment decisions. The dataset consists of gene expression
    data and clinical attributes. 57.72 percent have an observed death due to breast cancer whith a median survival time
    of 116 months."""
    # feature list:
    FEATURE_LIST = """
        MKI67:	MKI67 gene expression level
        EGFR:	EGF receptor gene expression level
        PGR: progesterone receptors (fmol/l)
        ERBB2:	ERBB2 gene expression level
        hormone_treatment: hormonal therapy, 0= no, 1= yes
        radiotherapy: radiotherapy, 0= no, 1= yes
        chemotherapy: chemotherapy, 0= no, 1= yes
        ER_positive: estrogen receptor positive, 0= no, 1= yes
        age:	age, years
        """

    # 2) Generate templates (JSON objects with role/structure/template)
    
    templates = generate_target_prompts(DATASET_DESCRIPTION, FEATURE_LIST, n=NUM_TEMPLATES, model="gpt-4o")
    if templates:
        with open("target_prompts.json", "w", encoding="utf-8") as f:
            json.dump(templates, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(templates)} templates to target_prompts.json")
        for template in templates:
            print(template["template"])
            print("="*100)

    # 3) Fill one template with a sample's FEATURES and generate a narrative
    if templates:
        sample_features = {
            "age_years": 61,
            "sex": "female",
            "bmi_kg_per_m2": 26.8,
            "tumor_size_mm": 28,
            "grade": "II",
            "positive_lymph_nodes": 3,
            "er_fmol_per_l": 180,
            "pgr_fmol_per_l": 95,
            "therapy_indicator": "hormonal_yes",
            "time": 900,   # do not restate as observed survival time
            "status": 0    # do not restate as actual outcome
        }
        feats_block = render_features_block(sample_features, fmt="bullet")
        filled = fill_template(templates[0]["template"], feats_block)
        print(filled)
        print("?"*100)
        narrative = generate_narrative(filled, model="gpt-4o")
        print("\n--- Sample Narrative (truncated) ---\n")
        print(narrative)
        # print(narrative[:1200] + ("..." if len(narrative) > 1200 else ""))
