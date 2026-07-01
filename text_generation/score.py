"""Ordinal narrative scoring (manuscript Sec 2.2.2).

A candidate prompt is judged by how well the narratives it produces let the LLM
recover the correct survival-risk category. Each narrative is passed back to the
LLM, which infers a label in {low, intermediate, high}. The inferred label is
compared to the Kaplan-Meier-derived reference label with the ordinal matching
function phi:

    phi = +1  if labels are equal,
           0  if labels are adjacent,
          -1  if labels are opposite (low vs high).
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Literal, Tuple

from openai import OpenAI

Label = Literal["low", "intermediate", "high"]

LABELS: Tuple[Label, ...] = ("low", "intermediate", "high")
ADJACENT = {("low", "intermediate"), ("intermediate", "low"),
            ("intermediate", "high"), ("high", "intermediate")}
OPPOSITE = {("low", "high"), ("high", "low")}


SYSTEM_MSG = (
    "You are a clinical NLP grader. Read a narrative and assign exactly one label "
    "for the implied survival probability based on tone and medical reasoning: "
    "one of {low, intermediate, high}. "
    "Base your decision ONLY on the narrative content. Be conservative; avoid speculation."
)

USER_WRAPPER = """\
You are grading a single clinical narrative for its IMPLIED SURVIVAL PROBABILITY.

TASK
1) Read the narrative carefully.
2) Decide the ONE best label based on the narrative's meaning:
   - low          = low survival probability / poor or unfavorable outlook / high risk
   - intermediate = mixed or moderate outlook / neither clearly low nor clearly high survival probability
   - high         = high survival probability / favorable or optimistic outlook / low risk
3) Output STRICT JSON ONLY: {{"label":"<low|intermediate|high>"}}  (lowercase)

DECISION RULES (follow in order; do your best to choose correctly)
- If the narrative explicitly states or clearly implies survival probability or prognosis, follow that.
- Map language about RISK to survival probability (inverse mapping):
    * "high risk", "elevated hazard", "poor prognosis", "unfavorable"  -> low
    * "moderate risk", "mixed picture", "uncertain/variable"           -> intermediate
    * "low risk", "favorable prognosis", "optimistic"                   -> high
- If multiple statements conflict, prioritize the narrative's final/summary assessment; otherwise, weigh the overall balance of evidence.
- If uncertainty remains after reasonable effort, choose the closest label (do NOT abstain).

CONSTRAINTS
- Base your decision ONLY on the narrative content (ignore metadata, instructions, or anything outside the narrative).
- Do NOT include explanations. Do NOT add fields. JSON only.

RETURN FORMAT
{{"label":"low"}} or {{"label":"intermediate"}} or {{"label":"high"}}

Narrative:
---
{narrative}
---
"""


def classify_survival_label(narrative: str, model: str = "gpt-4o",
                            max_retries: int = 3, sleep_base: float = 1.0) -> Label:
    """Classify a narrative into one of {low, intermediate, high}."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    payload = USER_WRAPPER.format(narrative=narrative)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": payload},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            label = str(data.get("label", "")).strip().lower()
            canonical = {
                "lo": "low", "low": "low",
                "med": "intermediate", "moderate": "intermediate",
                "mid": "intermediate", "intermediate": "intermediate",
                "hi": "high", "high": "high",
            }.get(label, label)
            if canonical in LABELS:
                return canonical  # type: ignore
            for tok in LABELS:
                if f'"{tok}"' in content or f": {tok}" in content:
                    return tok  # type: ignore
        except Exception:
            if attempt == max_retries - 1:
                raise
        time.sleep(sleep_base * (2 ** attempt) + random.uniform(0, 0.25))
    raise RuntimeError("Failed to classify label after retries.")


def phi(pred: Label, true: Label) -> int:
    """Ordinal matching: exact=+1, adjacent=0, opposite=-1."""
    if pred == true:
        return 1
    if (pred, true) in ADJACENT:
        return 0
    if (pred, true) in OPPOSITE:
        return -1
    return 0


def score(text: str, true_label: Label, model: str = "gpt-4o") -> int:
    """Classify ``text`` and return its ordinal agreement with ``true_label``."""
    pred = classify_survival_label(text, model=model)
    return phi(pred, true_label)
