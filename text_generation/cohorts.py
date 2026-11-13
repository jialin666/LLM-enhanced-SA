"""Per-cohort configuration for the automatic text-representation pipeline.

Each cohort provides the dataset description and feature list injected into the
meta prompt, the ordered feature columns rendered into the per-patient
``{FEATURES}`` block, and a short knowledge briefing used when producing the
numeric prognostic estimates ``N_i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

import pandas as pd

from llmsa.data import (
    GBSG_CAT, GBSG_NUM, METABRIC_CAT, METABRIC_NUM, SUPPORT_CAT, SUPPORT_NUM,
    FLCHAIN_CAT, FLCHAIN_NUM, TCGA_CAT, TCGA_NUM,
    _read_cohort_csv, _support_subsample, SUPPORT_SUBSAMPLE_SIZE,
)


@dataclass
class Cohort:
    name: str
    description: str
    feature_list: str
    feature_cols: List[str]
    briefing: str                 # fixed cohort briefing without outcome statistics
    # candidate-prompt search settings (manuscript defaults)
    n_candidates: int = 50
    n_eval_samples: int = 100
    # display names for feature_cols entries that are stored under a different
    # column name (e.g. raw label columns kept alongside the encoded ones)
    feature_labels: dict = field(default_factory=dict)
    # round float values to this many decimals in the prompt only (None = as
    # stored); the model inputs are never rounded
    display_decimals: Optional[int] = None
    _cat: List[str] = field(default_factory=list)
    _num: List[str] = field(default_factory=list)

    def load_full(self) -> pd.DataFrame:
        """Return the full cohort dataframe (covariates + time + event).

        For SUPPORT the fixed 1500-patient event-stratified subsample is used so
        the same cohort is seen everywhere.
        """
        df = _read_cohort_csv(self.name)
        extra = [c for c in self.feature_cols
                 if c not in self._cat + self._num and c in df.columns]
        df = df[self._cat + self._num + extra + ["time", "event"]]
        df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
        df["time"] = df["time"].astype(float)
        df["event"] = df["event"].astype(int).clip(0, 1)
        if self.name == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE:
            df = _support_subsample(df)
        return df

    def sample_dict(self, row) -> dict:
        """Extract the ordered feature values rendered into the FEATURES block."""
        out = {}
        for c in self.feature_cols:
            v = row[c]
            if self.display_decimals is not None and isinstance(v, (float, np.floating)):
                v = round(float(v), self.display_decimals)
                if v.is_integer():
                    v = int(v)
            out[self.feature_labels.get(c, c)] = v
        return out


# -----------------------------------------------------------------------------
# Knowledge briefings
#
# _BRIEFINGS: a fixed description of each cohort's clinical setting and
# covariates -- established risk factors, guideline thresholds and the
# direction of known effects -- WITHOUT any survival percentage, hazard ratio,
# median follow-up or other outcome statistic of the cohort itself. The
# criterion is deployability: each briefing could be written for a newly
# opened cohort before any outcome has accrued (manuscript Appendix E).
# -----------------------------------------------------------------------------

_BRIEFINGS = {
    "gbsg": (
        "Cohort context (node-positive breast cancer, use as background knowledge):\n"
        "- The number of positive axillary lymph nodes is the dominant adverse prognostic factor, "
        "  with risk rising steadily as nodal burden increases.\n"
        "- Higher tumour grade (grade 3 vs grade 1-2) and larger tumour size (>20 mm) are "
        "  independently adverse prognostic factors.\n"
        "- Hormone-receptor positivity (ER and/or PGR) is prognostically favourable and predicts "
        "  benefit from endocrine (hormonal) therapy; adjuvant endocrine therapy substantially "
        "  reduces recurrence risk in receptor-positive disease.\n"
        "- Menopausal status modifies treatment considerations but is a weaker prognostic factor "
        "  than nodal burden, grade, and size."
    ),
    "metabric": (
        "Cohort context (invasive breast cancer with molecular markers, use as background knowledge):\n"
        "- Hormone-receptor-positive tumours with low proliferation (low MKI67) generally carry a "
        "  more favourable prognosis than highly proliferative receptor-positive tumours.\n"
        "- ERBB2 (HER2) amplification marks a biologically aggressive phenotype; HER2-targeted "
        "  therapy combined with chemotherapy substantially improves outcomes in such patients.\n"
        "- Triple-negative / basal-like phenotypes (ER-negative, low PGR, no ERBB2 amplification) "
        "  are associated with earlier relapse and less favourable prognosis.\n"
        "- ER positivity predicts benefit from endocrine therapy; proliferation markers such as "
        "  MKI67 help distinguish indolent from aggressive receptor-positive disease.\n"
        "- Age at diagnosis and receipt of adjuvant therapy (endocrine, radio-, chemotherapy) "
        "  modify the risk profile."
    ),
    "flchain": (
        "Cohort context (general older adult population, serum free light chain assay, "
        "use as background knowledge):\n"
        "- Age is the dominant determinant of all-cause mortality in populations aged 50 and over; "
        "  risk rises steeply in the ninth and tenth decades.\n"
        "- Elevated serum free light chains (kappa and lambda) and a high combined FLC burden are "
        "  associated with increased mortality beyond what renal function alone explains; the FLC "
        "  decile group summarises this burden.\n"
        "- Free light chains are cleared renally, so elevated creatinine raises FLC levels; renal "
        "  impairment is itself an adverse prognostic factor and should be interpreted jointly "
        "  with FLC values.\n"
        "- An abnormal kappa/lambda ratio suggests a clonal plasma cell process, whereas a "
        "  proportional elevation of both chains more often reflects renal or inflammatory causes.\n"
        "- MGUS is a premalignant plasma-cell disorder that carries a small annual risk of "
        "  progression to myeloma or related malignancy.\n"
        "- Women have lower age-specific all-cause mortality than men in this age range."
    ),
    "support": (
        "Cohort context (hospitalized critically ill adults, use as background knowledge):\n"
        "- Prognosis varies sharply with the admitting disease category: multi-organ failure and "
        "  coma carry a graver short-term prognosis than exacerbations of single-organ chronic "
        "  disease (COPD, CHF, cirrhosis); metastatic or advanced cancer carries high short-term "
        "  mortality.\n"
        "- Physiological derangement at day 3 is strongly prognostic: hypotension (low mean "
        "  arterial pressure), renal dysfunction (elevated creatinine), hepatic dysfunction "
        "  (elevated bilirubin), and impaired oxygenation (low PaO2/FiO2 ratio) each add risk, "
        "  in line with APACHE/SOFA-style severity scoring.\n"
        "- The number of comorbidities adds approximately linearly to mortality risk.\n"
        "- Hypoalbuminaemia (serum albumin below ~2.5 g/dL) is a marker of advanced systemic "
        "  illness and poor prognosis."
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
    meanbp: mean arterial blood pressure of the patient (mmHg), measured at day 3.
    wblc: counts of white blood cells (x 1000 per microlitre) measured at day 3.
    hrt: heart rate of the patient (beats per minute) measured at day 3.
    resp: respiration rate of the patient (breaths per minute) measured at day 3.
    temp: temperature in Celsius degrees measured at day 3.
    pafi: PaO2/FiO2 ratio (mmHg) measured at day 3, a widely used clinical indicator of hypoxaemia.
    alb: serum albumin (g/dL) measured at day 3.
    bili: serum bilirubin (mg/dL) measured at day 3.
    crea: serum creatinine (mg/dL) measured at day 3.
    sod: serum sodium concentration (mEq/L) measured at day 3.
    """


_FLCHAIN_DESC = """Source: Olmsted County (Mayo Clinic) study of serum free light chain (FLC) assays.
Description:
    A population-based cohort of residents of Olmsted County, Minnesota, aged 50 years and older,
    who provided a serum sample between 1995 and 2003. Serum free light chain assays were performed
    and participants were followed for all-cause mortality. The cohort is a general (non-diseased)
    older population rather than a clinical disease cohort, and the outcome is death from any cause.
"""

_FLCHAIN_FEATS = """
    age: age at serum sample, years
    sex: sex (F = female, M = male)
    sample.yr: calendar year in which the serum sample was obtained
    flc.grp: serum free light chain group, a decile group from 1 (lowest) to 10 (highest)
    mgus: diagnosed with monoclonal gammopathy of undetermined significance (MGUS), no/yes
    kappa: serum free light chain, kappa portion (mg/dL)
    lambda: serum free light chain, lambda portion (mg/dL)
    creatinine: serum creatinine (mg/dL)
    """


COHORTS = {
    "gbsg": Cohort(
        name="gbsg", description=_GBSG_DESC, feature_list=_GBSG_FEATS,
        feature_cols=["age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon"],
        briefing=_BRIEFINGS["gbsg"],
        _cat=list(GBSG_CAT), _num=list(GBSG_NUM),
    ),
    "metabric": Cohort(
        name="metabric", description=_METABRIC_DESC, feature_list=_METABRIC_FEATS,
        feature_cols=["MKI67", "EGFR", "PGR", "ERBB2", "hormone_treatment",
                      "radiotherapy", "chemotherapy", "ER_positive", "age"],
        briefing=_BRIEFINGS["metabric"],
        _cat=list(METABRIC_CAT), _num=list(METABRIC_NUM),
    ),
    "flchain": Cohort(
        name="flchain", description=_FLCHAIN_DESC, feature_list=_FLCHAIN_FEATS,
        feature_cols=["age", "sex_raw", "sample.yr", "flc.grp", "mgus_raw",
                      "kappa", "lambda", "creatinine"],
        feature_labels={"sex_raw": "sex", "mgus_raw": "mgus"},
        briefing=_BRIEFINGS["flchain"],
        _cat=list(FLCHAIN_CAT), _num=list(FLCHAIN_NUM),
    ),
    "support": Cohort(
        name="support", description=_SUPPORT_DESC, feature_list=_SUPPORT_FEATS,
        # raw clinical values (see llmsa.data._read_support_raw); the
        # categoricals are rendered from their original labels
        feature_cols=["sex_raw", "dzclass_raw", "age", "num.co", "meanbp", "wblc", "hrt",
                      "resp", "temp", "pafi", "alb", "bili", "crea", "sod"],
        feature_labels={"sex_raw": "sex", "dzclass_raw": "dzclass"},
        display_decimals=2,
        briefing=_BRIEFINGS["support"],
        _cat=list(SUPPORT_CAT), _num=list(SUPPORT_NUM),
    ),
}


# -----------------------------------------------------------------------------
# TCGA-CDR pan-cancer pilot. Sanitized briefing: general oncology prognostic
# knowledge only, with no outcome figures taken from TCGA itself.
# -----------------------------------------------------------------------------

_TCGA_DESC = """Source: TCGA Pan-Cancer Clinical Data Resource (Liu et al., Cell 2018).
Description:
    A pan-cancer cohort drawn from The Cancer Genome Atlas, spanning 33 cancer types with
    heterogeneous prognosis. Each record describes one patient at the time of initial pathologic
    diagnosis. The endpoint is overall survival. Covariates are registry-style descriptors:
    the cancer type, the histological subtype (many of which are rare), pathologic stage,
    histological grade, and basic demographics. Stage and grade are frequently unrecorded.
"""

_TCGA_FEATS = """
    cancer_type: TCGA study code for the cancer type (e.g. BRCA breast, LUAD lung adenocarcinoma,
        GBM glioblastoma, PAAD pancreatic adenocarcinoma, THCA thyroid, PRAD prostate)
    histology: histological subtype as recorded by the contributing pathologist
    stage: AJCC pathologic tumour stage, where recorded
    grade: histological grade, where recorded
    age: age at initial pathologic diagnosis, years
    gender: patient sex
    race: self-reported race category
    dx_year: calendar year of initial pathologic diagnosis
    """

_BRIEFINGS["tcga"] = (
    "Cohort context (adults with a solid or haematological malignancy, use as background knowledge):\n"
    "- Stage at diagnosis is the dominant prognostic factor for most solid tumours; survival falls "
    "  steeply from localized to regional to metastatic disease.\n"
    "- Cancer type itself carries a large prognostic effect independent of stage: pancreatic "
    "  adenocarcinoma, glioblastoma, mesothelioma and acute myeloid leukaemia carry poor prognosis, "
    "  whereas thyroid carcinoma, prostate acinar adenocarcinoma and testicular germ-cell tumours "
    "  are usually highly curable.\n"
    "- Histological subtype refines prognosis within a cancer type: small-cell and sarcomatoid "
    "  patterns, and undifferentiated or metaplastic variants, behave more aggressively than "
    "  well-differentiated papillary or acinar patterns.\n"
    "- Higher histological grade indicates poorer differentiation and a worse outcome.\n"
    "- Older age at diagnosis is associated with shorter survival, partly through comorbidity and "
    "  reduced treatment tolerance.\n"
    "- Stage or grade recorded as 'not recorded' reflects incomplete registry documentation and "
    "  should not be read as favourable."
)

COHORTS["tcga"] = Cohort(
    name="tcga", description=_TCGA_DESC, feature_list=_TCGA_FEATS,
    feature_cols=["cancer_type_raw", "histology_raw", "stage_raw", "grade_raw",
                  "age", "gender_raw", "race_raw", "dx_year"],
    feature_labels={"cancer_type_raw": "cancer_type", "histology_raw": "histology",
                    "stage_raw": "stage", "grade_raw": "grade",
                    "gender_raw": "gender", "race_raw": "race"},
    briefing=_BRIEFINGS["tcga"],
    _cat=list(TCGA_CAT), _num=list(TCGA_NUM),
)

import copy as _copy
COHORTS["tcgafull"] = _copy.replace(COHORTS["tcga"], name="tcgafull") if hasattr(_copy, "replace") else None
if COHORTS["tcgafull"] is None:
    import dataclasses as _dc
    COHORTS["tcgafull"] = _dc.replace(COHORTS["tcga"], name="tcgafull")


# -----------------------------------------------------------------------------
# Small public survival cohorts (low-data-regime panel)
# -----------------------------------------------------------------------------

from llmsa.data import SMALL_COHORTS as _SMALL

#: Prompt directory of cohorts registered at run time (default: prompts/).
PROMPT_DIRS: dict = {}


def prompts_dir(dataset: str, default: str) -> str:
    return PROMPT_DIRS.get(dataset, default)


# --- UCI Heart Failure Clinical Records ---------------------------------------
_BRIEFINGS["heartfailure"] = (
    "Cohort context (patients with heart failure and reduced ejection fraction, use as "
    "background knowledge):\n"
    "- Left ventricular ejection fraction is the central prognostic measure: values below about "
    "  30% indicate severely impaired systolic function and high mortality.\n"
    "- Elevated serum creatinine reflects renal impairment, which frequently accompanies advanced "
    "  heart failure (cardiorenal syndrome) and markedly worsens prognosis.\n"
    "- Hyponatraemia (low serum sodium, below roughly 135 mmol/L) is a recognised marker of "
    "  neurohormonal activation and advanced disease.\n"
    "- Anaemia and older age are independently associated with worse outcome.\n"
    "- Creatine phosphokinase reflects recent myocardial or skeletal muscle injury and is less "
    "  specific prognostically than ejection fraction or renal function."
)
COHORTS["heartfailure"] = Cohort(
    name="heartfailure",
    description="""Source: UCI Heart Failure Clinical Records (Faisalabad Institute of Cardiology).
Description:
    299 patients with heart failure and left ventricular systolic dysfunction, followed for
    all-cause death during a follow-up period recorded in days.
""",
    feature_list="""
    age: age in years
    ejection_fraction: left ventricular ejection fraction (percentage of blood leaving the heart per contraction)
    serum_creatinine: serum creatinine (mg/dL)
    serum_sodium: serum sodium (mmol/L)
    platelets: platelet count (kiloplatelets/mL)
    cpk: creatine phosphokinase in the blood (mcg/L)
    anaemia: decreased red blood cells or haemoglobin (yes/no)
    diabetes: diabetes mellitus (yes/no)
    hypertension: history of high blood pressure (yes/no)
    smoking: current smoker (yes/no)
    sex: patient sex
    """,
    feature_cols=["age","ejection_fraction","serum_creatinine","serum_sodium","platelets","cpk",
                  "anaemia_raw","diabetes_raw","hypertension_raw","smoking_raw","sex_raw"],
    feature_labels={"anaemia_raw":"anaemia","diabetes_raw":"diabetes",
                    "hypertension_raw":"hypertension","smoking_raw":"smoking","sex_raw":"sex"},
    briefing=_BRIEFINGS["heartfailure"],
    _cat=list(_SMALL["heartfailure"][0]), _num=list(_SMALL["heartfailure"][1]),
)

# --- UCI Wisconsin Prognostic Breast Cancer -----------------------------------
_BRIEFINGS["wpbc"] = (
    "Cohort context (invasive breast cancer after surgery, use as background knowledge):\n"
    "- The number of positive axillary lymph nodes is the dominant predictor of recurrence; "
    "  node-negative disease recurs far less often than node-positive disease.\n"
    "- Larger primary tumour diameter is independently associated with higher recurrence risk.\n"
    "- Cell-nuclear morphometry from fine-needle aspirates describes tumour aggressiveness: larger "
    "  and more irregular nuclei, greater concavity of the nuclear boundary and more concave "
    "  points correspond to poorly differentiated, more aggressive tumours.\n"
    "- These morphometric values are computed image measurements; their absolute scales are "
    "  specific to the imaging protocol and are informative mainly in relative terms."
)
COHORTS["wpbc"] = Cohort(
    name="wpbc",
    description="""Source: UCI Breast Cancer Wisconsin (Prognostic), WPBC.
Description:
    198 patients with invasive breast cancer, each described by the primary tumour size, the number
    of positive axillary lymph nodes, and cell-nuclear features computed from a digitised
    fine-needle aspirate. The endpoint is recurrence; disease-free patients are censored at last
    follow-up. Times are recorded in months.
""",
    feature_list="""
    tumor_size: diameter of the excised tumour (cm)
    lymph_node_status: number of positive axillary lymph nodes observed at surgery
    radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
    compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dim_worst:
        the largest ("worst") value across cell nuclei for each morphometric feature measured on
        the fine-needle aspirate image
    """,
    feature_cols=["tumor_size","lymph_node_status","radius_worst","texture_worst","perimeter_worst",
                  "area_worst","smoothness_worst","compactness_worst","concavity_worst",
                  "concave_points_worst","symmetry_worst","fractal_dim_worst"],
    briefing=_BRIEFINGS["wpbc"],
    _cat=list(_SMALL["wpbc"][0]), _num=list(_SMALL["wpbc"][1]),
)
