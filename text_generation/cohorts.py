"""Per-cohort configuration for the automatic text-representation pipeline.

Each cohort provides the dataset description and feature list injected into the
meta prompt, the ordered feature columns rendered into the per-patient
``{FEATURES}`` block, and a short knowledge briefing used when producing the
numeric prognostic estimates ``N_i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from llmsa.data import (
    GBSG_CAT, GBSG_NUM, METABRIC_CAT, METABRIC_NUM, SUPPORT_CAT, SUPPORT_NUM,
    _read_cohort_csv, _support_subsample, SUPPORT_SUBSAMPLE_SIZE,
)


@dataclass
class Cohort:
    name: str
    description: str
    feature_list: str
    feature_cols: List[str]
    briefing: str
    # candidate-prompt search settings (manuscript defaults)
    n_candidates: int = 50
    n_eval_samples: int = 100
    _cat: List[str] = field(default_factory=list)
    _num: List[str] = field(default_factory=list)

    def load_full(self) -> pd.DataFrame:
        """Return the full cohort dataframe (covariates + time + event).

        For SUPPORT the fixed 1500-patient event-stratified subsample is used so
        the same cohort is seen everywhere.
        """
        df = _read_cohort_csv(self.name)
        df = df[self._cat + self._num + ["time", "event"]]
        df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
        df["time"] = df["time"].astype(float)
        df["event"] = df["event"].astype(int).clip(0, 1)
        if self.name == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE:
            df = _support_subsample(df)
        return df

    def sample_dict(self, row) -> dict:
        """Extract the ordered feature values rendered into the FEATURES block."""
        return {c: row[c] for c in self.feature_cols}


# -----------------------------------------------------------------------------
# Knowledge briefings (background priors for the numeric estimates N_i)
# -----------------------------------------------------------------------------

_BRIEFINGS = {
    "gbsg": (
        "Cohort context (node-positive breast cancer, GBSG-style cohorts, "
        "use as background knowledge):\n"
        "- EBCTCG meta-analyses: each additional positive lymph node adds "
        "  HR ~1.05; tumour size >20mm HR ~1.2; grade 3 vs grade 1 HR ~1.8.\n"
        "- Adjuvant tamoxifen halves the annual recurrence rate in ER+ disease.\n"
        "- Median follow-up in GBSG ~5 years; 5-year recurrence-free survival "
        "  ~60-70% overall, varying strongly by nodal burden and grade.\n"
        "- PGR >50 fmol/L is a favourable prognostic marker independent of ER."
    ),
    "metabric": (
        "Cohort context (METABRIC invasive breast cancer, use as background knowledge):\n"
        "- PAM50 intrinsic subtypes: Luminal A (best prognosis, ~85-90% 10-yr BCSS); "
        "  Luminal B (moderate, ~70% 10-yr BCSS); HER2-enriched and Basal-like "
        "  (poorer, ~50-60% 10-yr BCSS).\n"
        "- MKI67 proliferation index distinguishes Luminal A (low) from Luminal B (high).\n"
        "- ERBB2 amplification predicts response to HER2-targeted therapy with "
        "  ~60% response rate when combined with chemotherapy.\n"
        "- ER-positivity confers responsiveness to endocrine therapy (~50-70% benefit)."
    ),
    "support": (
        "Cohort context (SUPPORT study of critically ill adults, use as background knowledge):\n"
        "- Baseline mortality varies sharply by disease class: ARF/MOSF ~50% 6-mo "
        "  mortality; COPD/CHF/Cirrhosis ~40%; Cancer ~70%; Coma ~80%.\n"
        "- APACHE-II / SOFA components: elevated creatinine, bilirubin, low mean "
        "  arterial pressure, low PaO2/FiO2 ratio are strongly prognostic.\n"
        "- Number of comorbidities adds linearly to mortality risk.\n"
        "- Serum albumin <2.5 g/dL is a marker of advanced systemic illness."
    ),
}


# -----------------------------------------------------------------------------
# Dataset descriptions and feature lists (injected into the meta prompt)
# -----------------------------------------------------------------------------

_GBSG_DESC = """Source: German Breast Cancer Study Group (GBSG).
Description:
    The dataset contains patient records from a 1984-1989 trial conducted by the
    German Breast Cancer Study Group of 720 patients with node-positive breast
    cancer; it retains the 686 patients with complete data for the prognostic
    variables. These data are used in the paper by Royston and Altman (2013).
"""

_GBSG_FEATS = """
    age: age, years
    meno: menopausal status (0= premenopausal, 1= postmenopausal)
    size: tumor size, mm
    grade: tumor grade
    nodes: number of positive lymph nodes
    pgr: progesterone receptors (fmol/l)
    er: estrogen receptors (fmol/l)
    hormon: hormonal therapy, 0= no, 1= yes
    """

_METABRIC_DESC = """The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC)
    is a large-scale international study of breast cancer patients.
    It contains information on 1,980 patients with breast cancer, including their clinical attributes,
    molecular data, and survival information. METABRIC uses gene and protein expression profiles to determine new
    breast cancer subgroups in order to help personalize treatment decisions. The dataset consists of gene expression
    data and clinical attributes. 57.72 percent have an observed death due to breast cancer with a median survival time
    of 116 months."""

_METABRIC_FEATS = """
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

_SUPPORT_DESC = """This dataset comprises 9105 individual critically ill patients across 5 United States medical centers,
accessioned throughout 1989-1991 and 1992-1994. Each row concerns hospitalized patient records who met the inclusion and exclusion
criteria for nine disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure,
liver disease, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis.
The goal is to determine these patients' 2- and 6-month survival rates based on several physiologic, demographics, and disease severity information."""

_SUPPORT_FEATS = """
    sex: Gender of the patient. Listed values are {male, female}.
    dzclass: The patient's disease category amongst "ARF/MOSF", "COPD/CHF/Cirrhosis", "Cancer", "Coma".
    age:	Age of the patients in years
    num.co: The number of simultaneous diseases (or comorbidities) exhibited by the patient. Higher values indicate worse condition.
    meanbp: mean arterial blood pressure of the patient, measured at day 3.
    wblc: counts of white blood cells (in thousands) measured at day 3.
    hrt: heart rate of the patient measured at day 3.
    resp: respiration rate of the patient measured at day 3.
    temp: temperature in Celsius degrees measured at day 3.
    pafi: PaO2/FiO2 ratio measured at day 3, a widely used clinical indicator of hypoxaemia.
    alb: serum albumin levels measured at day 3.
    bili: bilirubin levels measured at day 3.
    crea: serum creatinine levels measured at day 3.
    sod: serum sodium concentration measured at day 3.
    """


COHORTS = {
    "gbsg": Cohort(
        name="gbsg", description=_GBSG_DESC, feature_list=_GBSG_FEATS,
        feature_cols=["age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon"],
        briefing=_BRIEFINGS["gbsg"], _cat=list(GBSG_CAT), _num=list(GBSG_NUM),
    ),
    "metabric": Cohort(
        name="metabric", description=_METABRIC_DESC, feature_list=_METABRIC_FEATS,
        feature_cols=["MKI67", "EGFR", "PGR", "ERBB2", "hormone_treatment",
                      "radiotherapy", "chemotherapy", "ER_positive", "age"],
        briefing=_BRIEFINGS["metabric"], _cat=list(METABRIC_CAT), _num=list(METABRIC_NUM),
    ),
    "support": Cohort(
        name="support", description=_SUPPORT_DESC, feature_list=_SUPPORT_FEATS,
        feature_cols=["sex", "dzclass", "age", "num.co", "meanbp", "wblc", "hrt",
                      "resp", "temp", "pafi", "alb", "bili", "crea", "sod"],
        briefing=_BRIEFINGS["support"], _cat=list(SUPPORT_CAT), _num=list(SUPPORT_NUM),
    ),
}
