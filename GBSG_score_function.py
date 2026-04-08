# scorer.py
import os, json, time, math, random
from typing import Dict, List, Literal, Tuple
import pandas as pd
from openai import OpenAI

Label = Literal["low", "intermediate", "high"]

LABELS: Tuple[Label, ...] = ("low", "intermediate", "high")
ADJACENT = {("low","intermediate"), ("intermediate","low"),
            ("intermediate","high"), ("high","intermediate")}
OPPOSITE = {("low","high"), ("high","low")}

# -----------------------------
# 1) OpenAI classifier
# -----------------------------
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
2) Decide the ONE best label based on the narrative’s meaning:
   - low          = low survival probability / poor or unfavorable outlook / high risk
   - intermediate = mixed or moderate outlook / neither clearly low nor clearly high survival probability
   - high         = high survival probability / favorable or optimistic outlook / low risk
3) Output STRICT JSON ONLY: {{"label":"<low|intermediate|high>"}}  (lowercase)

DECISION RULES (follow in order; do your best to choose correctly)
- If the narrative explicitly states or clearly implies survival probability or prognosis, follow that.
- Map language about RISK to survival probability (inverse mapping):
    * “high risk”, “elevated hazard”, “poor prognosis”, “unfavorable”  -> low
    * “moderate risk”, “mixed picture”, “uncertain/variable”           -> intermediate
    * “low risk”, “favorable prognosis”, “optimistic”                   -> high
- If multiple statements conflict, prioritize the narrative’s final/summary assessment; otherwise, weigh the overall balance of evidence.
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

def classify_survival_label(narrative: str,
                            model: str = "gpt-4o",
                            max_retries: int = 3,
                            sleep_base: float = 1.0) -> Label:
    """
    Call OpenAI to classify the narrative into one of {low, intermediate, high}.
    Returns a validated label; raises on repeated failure.
    """
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
            label = str(data.get("label","")).strip().lower()
            # normalize a few common variants
            canonical = {
                "lo":"low", "low":"low",
                "med":"intermediate", "moderate":"intermediate", "mid":"intermediate", "intermediate":"intermediate",
                "hi":"high", "high":"high"
            }.get(label, label)
            if canonical in LABELS:
                return canonical  # type: ignore
            # If the model tried to explain, try a light parse:
            for tok in ["low", "intermediate", "high"]:
                if f"\"{tok}\"" in content or f": {tok}" in content:
                    return tok  # type: ignore
        except Exception as e:
            if attempt == max_retries - 1:
                raise
        # backoff
        time.sleep(sleep_base * (2 ** attempt) + random.uniform(0, 0.25))
    raise RuntimeError("Failed to classify label after retries.")

# -----------------------------
# 2) Scoring function
# -----------------------------
def score_pair(pred: Label, true: Label) -> int:
    """
    Exact = 1, Adjacent = 0, Opposite = -1.
    """
    if pred == true:
        return 1
    if (pred, true) in ADJACENT:
        return 0
    if (pred, true) in OPPOSITE:
        return -1
    # Shouldn't happen, but stay safe:
    return 0

def score(text:str, true_label:Label) -> int:
    pred = classify_survival_label(text, model="gpt-4o")
    return score_pair(pred, true_label)



# # -----------------------------
# # 3) Batch evaluation helpers
# # -----------------------------
# def evaluate_dataframe(df: pd.DataFrame,
#                        text_col: str = "generated_text",
#                        label_col: str = "true_label",
#                        model: str = "gpt-4o-mini") -> pd.DataFrame:
#     """
#     df must contain:
#       - text_col: the generated narrative
#       - label_col: the categorical label ("low"|"intermediate"|"high")
#     Returns a copy with columns: pred_label, score.
#     """
#     out = df.copy()
#     preds: List[Label] = []
#     scores: List[int] = []
#     for i, row in out.iterrows():
#         txt = str(row[text_col])
#         true_label = str(row[label_col]).strip().lower()
#         if true_label not in LABELS:
#             raise ValueError(f"Row {i}: invalid true label '{true_label}'. Must be one of {LABELS}.")
#         pred = classify_survival_label(txt, model=model)
#         s = score_pair(pred, true_label)  # type: ignore
#         preds.append(pred)
#         scores.append(s)
#     out["pred_label"] = preds
#     out["score"] = scores
#     return out

# def summarize_results(df_scored: pd.DataFrame) -> pd.DataFrame:
#     """
#     Produce a compact summary: counts by (true, pred), mean score overall and by true label.
#     """
#     cm = pd.crosstab(df_scored["true_label"], df_scored["pred_label"], dropna=False)
#     overall = pd.DataFrame({
#         "metric": ["mean_score"],
#         "value": [df_scored["score"].mean() if len(df_scored) else float("nan")]
#     })
#     by_true = df_scored.groupby("true_label")["score"].mean().reset_index().rename(columns={"score":"mean_score"})
#     print("\nConfusion matrix (rows=true, cols=pred):\n", cm)
#     print("\nOverall mean score:", overall["value"].iloc[0])
#     print("\nMean score by true label:\n", by_true)
#     return cm

# # -----------------------------
# # 4) Minimal demo
# # -----------------------------
# if __name__ == "__main__":
#     # Example data: replace with your real dataset
#     data = [
#         {
#             "generated_text": "The narrative emphasizes multiple positive lymph nodes, large tumor burden, and poor grade, "
#                               "noting a substantially unfavorable outlook with limited likelihood of long-term survival.",
#             "true_label": "low",
#         },
#         {
#             "generated_text": "Mixed factors: moderate tumor size, some comorbidities, but receptor positivity may support response "
#                               "to therapy. Overall, a middling probability of survival is suggested.",
#             "true_label": "intermediate",
#         },
#         {
#             "generated_text": "Small lesion, favorable biomarkers, limited nodal involvement, and good performance status, "
#                               "supporting an optimistic estimate of survival probability.",
#             "true_label": "high",
#         },
#     ]
#     df = pd.DataFrame(data)
#     scored = evaluate_dataframe(df, text_col="generated_text", label_col="true_label", model="gpt-4o-mini")
#     print("\nScored rows:\n", scored)
#     _ = summarize_results(scored)
