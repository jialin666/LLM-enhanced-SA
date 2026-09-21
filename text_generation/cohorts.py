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
    FLCHAIN_CAT, FLCHAIN_NUM, ROTTERDAM_CAT, ROTTERDAM_NUM, TCGA_CAT, TCGA_NUM,
    GBSG3_CAT, GBSG3_NUM, GBSG5_CAT, GBSG5_NUM,
    METABRIC3_CAT, METABRIC3_NUM, METABRIC5_CAT, METABRIC5_NUM,
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
        return {self.feature_labels.get(c, c): row[c] for c in self.feature_cols}


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
        feature_cols=["sex", "dzclass", "age", "num.co", "meanbp", "wblc", "hrt",
                      "resp", "temp", "pafi", "alb", "bili", "crea", "sod"],
        briefing=_BRIEFINGS["support"],
        _cat=list(SUPPORT_CAT), _num=list(SUPPORT_NUM),
    ),
}


# -----------------------------------------------------------------------------
# Reduced-covariate variants (covariate-sparsity experiment). Same cohort and
# outcomes; the LLM and the downstream models both see only the listed subset.
# -----------------------------------------------------------------------------

_GBSG3_FEATS = """
    age: age, years
    grade: tumor grade
    nodes: number of positive lymph nodes
    """
_GBSG5_FEATS = """
    age: age, years
    size: tumor size, mm
    grade: tumor grade
    nodes: number of positive lymph nodes
    hormon: hormonal therapy, 0= no, 1= yes
    """
_METABRIC3_FEATS = """
    age:	age, years
    MKI67:	MKI67 gene expression level
    ER_positive: estrogen receptor positive, 0= no, 1= yes
    """
_METABRIC5_FEATS = """
    age:	age, years
    MKI67:	MKI67 gene expression level
    ERBB2:	ERBB2 gene expression level
    ER_positive: estrogen receptor positive, 0= no, 1= yes
    hormone_treatment: hormonal therapy, 0= no, 1= yes
    """

for _name, _desc, _feats, _cat, _num, _cols_order, _brief in (
    ("gbsg3", _GBSG_DESC, _GBSG3_FEATS, GBSG3_CAT, GBSG3_NUM,
     ["age", "grade", "nodes"], _BRIEFINGS["gbsg"]),
    ("gbsg5", _GBSG_DESC, _GBSG5_FEATS, GBSG5_CAT, GBSG5_NUM,
     ["age", "size", "grade", "nodes", "hormon"], _BRIEFINGS["gbsg"]),
    ("metabric3", _METABRIC_DESC, _METABRIC3_FEATS, METABRIC3_CAT, METABRIC3_NUM,
     ["age", "MKI67", "ER_positive"], _BRIEFINGS["metabric"]),
    ("metabric5", _METABRIC_DESC, _METABRIC5_FEATS, METABRIC5_CAT, METABRIC5_NUM,
     ["age", "MKI67", "ERBB2", "ER_positive", "hormone_treatment"], _BRIEFINGS["metabric"]),
):
    COHORTS[_name] = Cohort(
        name=_name, description=_desc, feature_list=_feats,
        feature_cols=_cols_order, briefing=_brief,
        _cat=list(_cat), _num=list(_num),
    )


# -----------------------------------------------------------------------------
# Rotterdam breast cancer cohort (GBSG's companion cohort; same covariates and
# the same recurrence-free survival endpoint). Uses the GBSG briefing, since it
# is the same disease and the briefing carries no cohort-specific figures.
# -----------------------------------------------------------------------------

_ROTTERDAM_DESC = """Source: Rotterdam tumour bank (Royston and Altman, 2013).
Description:
    Records of 2,982 patients with primary breast cancer treated at the Rotterdam tumour bank,
    with follow-up for recurrence-free survival (the earlier of recurrence or death). The cohort
    is the standard companion to the German Breast Cancer Study Group data and carries the same
    prognostic variables.
"""

_ROTTERDAM_FEATS = """
    age: age at surgery, years
    meno: menopausal status (0= premenopausal, 1= postmenopausal)
    size: tumour size category (<=20 mm, 20-50 mm, >50 mm)
    grade: tumour grade (2 or 3)
    nodes: number of positive lymph nodes
    pgr: progesterone receptors (fmol/l)
    er: estrogen receptors (fmol/l)
    hormon: hormonal therapy, 0= no, 1= yes
    """

COHORTS["rotterdam"] = Cohort(
    name="rotterdam", description=_ROTTERDAM_DESC, feature_list=_ROTTERDAM_FEATS,
    feature_cols=["age", "meno", "size_raw", "grade_raw", "nodes", "pgr", "er", "hormon"],
    feature_labels={"size_raw": "size", "grade_raw": "grade"},
    briefing=_BRIEFINGS["gbsg"],
    _cat=list(ROTTERDAM_CAT), _num=list(ROTTERDAM_NUM),
)


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

_BRIEFINGS['veteran'] = 'Cohort context (advanced inoperable lung cancer, use as background knowledge):\n- Performance status is the dominant prognostic factor in advanced lung cancer; low Karnofsky\n  scores indicate poor functional reserve and short expected survival.\n- Small-cell histology behaves more aggressively than squamous or adenocarcinoma histology.\n- Advanced inoperable disease carries a poor prognosis overall, with chemotherapy offering\n  modest benefit in this setting.\n- Longer time from diagnosis to treatment can reflect more indolent disease.'
COHORTS['veteran'] = Cohort(
    name='veteran', description='Source: US Veterans Administration lung cancer trial (Kalbfleisch and Prentice).\nDescription:\n    A randomised trial in 137 male patients with advanced inoperable lung cancer comparing a\n    standard and a test chemotherapy. The endpoint is death from any cause.\n', feature_list='\n    celltype: histological cell type (squamous, smallcell, adeno, large)\n    karnofsky: Karnofsky performance score (0-100; higher is better functional status)\n    months_from_dx: months from diagnosis to randomisation\n    age: age in years\n    prior_therapy: prior chemotherapy received (yes/no)\n    treatment: assigned treatment arm (standard/test)\n    ',
    feature_cols=[c + "_raw" if c in _SMALL['veteran'][0] else c
                  for c in list(_SMALL['veteran'][1]) + list(_SMALL['veteran'][0])],
    feature_labels={c + "_raw": c for c in _SMALL['veteran'][0]},
    briefing=_BRIEFINGS['veteran'],
    _cat=list(_SMALL['veteran'][0]), _num=list(_SMALL['veteran'][1]),
)

_BRIEFINGS['lung'] = 'Cohort context (advanced lung cancer, use as background knowledge):\n- ECOG and Karnofsky performance status are the strongest clinical predictors of survival in\n  advanced lung cancer; deteriorating performance status implies short expected survival.\n- Significant recent weight loss (cancer cachexia) is an independent adverse prognostic factor.\n- Reduced caloric intake reflects both disease burden and nutritional depletion.\n- Physician- and patient-rated performance scores may disagree; both carry information.'
COHORTS['lung'] = Cohort(
    name='lung', description='Source: North Central Cancer Treatment Group (NCCTG) lung cancer study.\nDescription:\n    Survival of 228 patients with advanced lung cancer, with performance scores rated by both\n    physician and patient, and nutritional indicators. The endpoint is death from any cause.\n', feature_list='\n    age: age in years\n    sex: patient sex\n    ecog: ECOG performance status rated by the physician (0 good, higher is worse)\n    karnofsky_phys: Karnofsky performance score rated by the physician (0-100)\n    karnofsky_pat: Karnofsky performance score rated by the patient (0-100)\n    meal_calories: calories consumed at meals\n    weight_loss: weight loss in the last six months (pounds)\n    ',
    feature_cols=[c + "_raw" if c in _SMALL['lung'][0] else c
                  for c in list(_SMALL['lung'][1]) + list(_SMALL['lung'][0])],
    feature_labels={c + "_raw": c for c in _SMALL['lung'][0]},
    briefing=_BRIEFINGS['lung'],
    _cat=list(_SMALL['lung'][0]), _num=list(_SMALL['lung'][1]),
)

_BRIEFINGS['whas500'] = 'Cohort context (hospitalised acute myocardial infarction, use as background knowledge):\n- Older age is the dominant determinant of mortality after myocardial infarction.\n- Cardiogenic shock and complete heart block are severe complications carrying very high\n  short-term mortality; congestive heart failure complicating infarction is also strongly adverse.\n- Recurrent infarction implies worse prognosis than a first event.\n- Both hypotension and marked tachycardia at admission indicate haemodynamic compromise.\n- Very low body mass index is associated with frailty and worse outcome.'
COHORTS['whas500'] = Cohort(
    name='whas500', description='Source: Worcester Heart Attack Study (WHAS500).\nDescription:\n    A population-based study of 500 patients hospitalised with acute myocardial infarction in\n    Worcester, Massachusetts, followed for all-cause mortality after admission.\n', feature_list='\n    age: age in years\n    bmi: body mass index at admission\n    sysbp: systolic blood pressure at admission (mmHg)\n    diasbp: diastolic blood pressure at admission (mmHg)\n    hr: initial heart rate (beats per minute)\n    gender: patient sex\n    chf: congestive heart failure complications (yes/no)\n    afb: atrial fibrillation (yes/no)\n    av3: complete heart block (yes/no)\n    cvd: cardiovascular disease history (yes/no)\n    miord: MI order (first or recurrent)\n    mitype: MI type (Q-wave or non-Q-wave)\n    sho: cardiogenic shock (yes/no)\n    ',
    feature_cols=[c + "_raw" if c in _SMALL['whas500'][0] else c
                  for c in list(_SMALL['whas500'][1]) + list(_SMALL['whas500'][0])],
    feature_labels={c + "_raw": c for c in _SMALL['whas500'][0]},
    briefing=_BRIEFINGS['whas500'],
    _cat=list(_SMALL['whas500'][0]), _num=list(_SMALL['whas500'][1]),
)

_BRIEFINGS['larynx'] = 'Cohort context (laryngeal carcinoma, use as background knowledge):\n- Stage at diagnosis dominates prognosis: stage I disease is usually curable with local therapy,\n  while stage IV disease carries substantially reduced survival.\n- Older age at diagnosis is associated with worse outcome, partly through comorbidity and\n  reduced tolerance of treatment.'
COHORTS['larynx'] = Cohort(
    name='larynx', description='Source: Larynx cancer cohort (Kardaun; Klein and Moeschberger).\nDescription:\n    Ninety male patients with cancer of the larynx treated at a single institution, followed for\n    death from any cause. Disease stage at diagnosis is the principal covariate.\n', feature_list='\n    age: age at diagnosis, years\n    stage: disease stage at diagnosis (I, II, III or IV)\n    ',
    feature_cols=[c + "_raw" if c in _SMALL['larynx'][0] else c
                  for c in list(_SMALL['larynx'][1]) + list(_SMALL['larynx'][0])],
    feature_labels={c + "_raw": c for c in _SMALL['larynx'][0]},
    briefing=_BRIEFINGS['larynx'],
    _cat=list(_SMALL['larynx'][0]), _num=list(_SMALL['larynx'][1]),
)

_BRIEFINGS['aids'] = 'Cohort context (HIV infection with prior antiretroviral therapy, use as background knowledge):\n- The baseline CD4 count is the dominant prognostic marker: low counts indicate advanced\n  immunosuppression and a high risk of progression to AIDS or death.\n- Poorer Karnofsky performance status indicates greater disease burden.\n- Extensive prior monotherapy is associated with drug resistance and reduced response.\n- Three-drug regimens were substantially more effective than two-drug regimens in this era.'
COHORTS['aids'] = Cohort(
    name='aids', description='Source: AIDS Clinical Trials Group protocol 320 (ACTG 320).\nDescription:\n    A randomised trial in HIV-infected patients with prior antiretroviral therapy comparing\n    two- and three-drug regimens. The endpoint is progression to AIDS or death.\n', feature_list='\n    age: age in years\n    cd4: baseline CD4 T-cell count (cells/mm3)\n    priorzdv: months of prior zidovudine therapy\n    karnof: Karnofsky performance score category\n    sex: patient sex\n    raceth: race/ethnicity category\n    ivdrug: history of intravenous drug use\n    hemophil: haemophilia (yes/no)\n    strat2: CD4 stratum at entry\n    tx: treatment assignment\n    txgrp: treatment group\n    ',
    feature_cols=[c + "_raw" if c in _SMALL['aids'][0] else c
                  for c in list(_SMALL['aids'][1]) + list(_SMALL['aids'][0])],
    feature_labels={c + "_raw": c for c in _SMALL['aids'][0]},
    briefing=_BRIEFINGS['aids'],
    _cat=list(_SMALL['aids'][0]), _num=list(_SMALL['aids'][1]),
)

_BRIEFINGS['brcamicro'] = 'Cohort context (invasive breast cancer with microarray profiling, use as background knowledge):\n- Estrogen-receptor-positive disease generally carries a more favourable prognosis and predicts\n  benefit from endocrine therapy.\n- Higher histological grade indicates poorer differentiation and worse outcome.\n- Microarray probe-set values are platform-specific normalised intensities; their absolute scale\n  has no fixed clinical interpretation.'
COHORTS['brcamicro'] = Cohort(
    name='brcamicro', description='Source: Breast cancer gene-expression cohort (van de Vijver / Desmedt-style microarray data).\nDescription:\n    198 breast cancer patients with clinical grade and estrogen-receptor status together with\n    normalised microarray expression values for individual probe sets. The endpoint is distant\n    metastasis-free survival.\n', feature_list='\n    er: estrogen receptor status\n    grade: histological grade\n    X200726_at, X200965_s_at, X201068_s_at, X201091_s_at, X201288_at, X201368_at,\n    X201663_s_at, X201664_at, X202239_at, X202240_at:\n        normalised microarray expression values for individual Affymetrix probe sets\n    ',
    feature_cols=[c + "_raw" if c in _SMALL['brcamicro'][0] else c
                  for c in list(_SMALL['brcamicro'][1]) + list(_SMALL['brcamicro'][0])],
    feature_labels={c + "_raw": c for c in _SMALL['brcamicro'][0]},
    briefing=_BRIEFINGS['brcamicro'],
    _cat=list(_SMALL['brcamicro'][0]), _num=list(_SMALL['brcamicro'][1]),
)


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
