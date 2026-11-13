"""Seven public survival cohorts evaluated during method development and not
part of the benchmark: Rotterdam, WHAS500, Veteran, Larynx, NCCTG Lung,
ACTG320 (``aids``) and a breast-cancer microarray cohort (``brcamicro``).

Importing this package registers them with the main loaders (covariates, data
and prompt locations, per-cohort settings, time units, horizons and briefings),
so every script in ``llmsa`` and ``text_generation`` can run on them unchanged.
Use the runner:

    python -m development_cohorts.run llmsa.run_ablation --datasets larynx --feature-version v5 --cindex-tau grid
"""
import os

from llmsa import data as _data, model as _model, train_eval as _train_eval
from text_generation import cohorts as _cohorts, meta_prompt as _meta_prompt
from text_generation.cohorts import Cohort, _BRIEFINGS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
PROMPTS_DIR = os.path.join(HERE, "prompts")

COVARIATES = {
    "rotterdam": (["meno", "size", "grade", "hormon"], ["age", "nodes", "pgr", "er"]),
    "veteran": (['celltype', 'prior_therapy', 'treatment'], ['age', 'karnofsky', 'months_from_dx']),
    "lung": (['sex_lbl'], ['age', 'ecog', 'karnofsky_phys', 'karnofsky_pat', 'meal_calories', 'weight_loss']),
    "whas500": (['afb', 'av3', 'chf', 'cvd', 'gender', 'miord', 'mitype', 'sho'], ['age', 'bmi', 'diasbp', 'hr', 'sysbp']),
    "larynx": (['stage'], ['age']),
    "aids": (['hemophil', 'ivdrug', 'karnof', 'raceth', 'sex', 'strat2', 'tx', 'txgrp'], ['age', 'cd4', 'priorzdv']),
    "brcamicro": (['er', 'grade'], ['X200726_at', 'X200965_s_at', 'X201068_s_at', 'X201091_s_at', 'X201288_at', 'X201368_at', 'X201663_s_at', 'X201664_at', 'X202239_at', 'X202240_at']),
}

CSV = {"rotterdam": "6_rotterdam_all.csv",
       **{k: f"8_{k}.csv" for k in ("veteran", "lung", "whas500", "larynx", "aids", "brcamicro")}}

# Settings used for every development run (as for the small benchmark cohorts).
SETTINGS = {"rotterdam": dict(text_pca_dim=64, batch_size=128),
            **{k: dict(text_pca_dim=32, batch_size=32)
               for k in ("veteran", "lung", "whas500", "larynx", "aids", "brcamicro")}}

# years -> native time unit of each cohort
TIME_UNIT_PER_YEAR = {"rotterdam": 365.25, "whas500": 365.25, "brcamicro": 365.25,
                      "aids": 365.25, "lung": 365.25, "veteran": 365.25, "larynx": 1.0}

# ACTG320: follow-up 364 d, median event 91 d; 90d/270d bracket 46 and 94 of 96 events
HORIZONS = {"aids": ("90d", "270d")}
_D = 365.25
ANCHOR_YEARS = {("v5hz", "aids"): (90.0 / _D, 270.0 / _D)}

# Rotterdam uses the GBSG briefing (same disease, same covariates).
_GBSG_BRIEFING = _BRIEFINGS["gbsg"]
BRIEFINGS = {}
COHORTS = {}

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
    briefing=_GBSG_BRIEFING,
    _cat=list(COVARIATES['rotterdam'][0]), _num=list(COVARIATES['rotterdam'][1]),
)

BRIEFINGS['veteran'] = 'Cohort context (advanced inoperable lung cancer, use as background knowledge):\n- Performance status is the dominant prognostic factor in advanced lung cancer; low Karnofsky\n  scores indicate poor functional reserve and short expected survival.\n- Small-cell histology behaves more aggressively than squamous or adenocarcinoma histology.\n- Advanced inoperable disease carries a poor prognosis overall, with chemotherapy offering\n  modest benefit in this setting.\n- Longer time from diagnosis to treatment can reflect more indolent disease.'
COHORTS['veteran'] = Cohort(
    name='veteran', description='Source: US Veterans Administration lung cancer trial (Kalbfleisch and Prentice).\nDescription:\n    A randomised trial in 137 male patients with advanced inoperable lung cancer comparing a\n    standard and a test chemotherapy. The endpoint is death from any cause.\n', feature_list='\n    celltype: histological cell type (squamous, smallcell, adeno, large)\n    karnofsky: Karnofsky performance score (0-100; higher is better functional status)\n    months_from_dx: months from diagnosis to randomisation\n    age: age in years\n    prior_therapy: prior chemotherapy received (yes/no)\n    treatment: assigned treatment arm (standard/test)\n    ',
    feature_cols=[c + "_raw" if c in COVARIATES['veteran'][0] else c
                  for c in list(COVARIATES['veteran'][1]) + list(COVARIATES['veteran'][0])],
    feature_labels={c + "_raw": c for c in COVARIATES['veteran'][0]},
    briefing=BRIEFINGS['veteran'],
    _cat=list(COVARIATES['veteran'][0]), _num=list(COVARIATES['veteran'][1]),
)

BRIEFINGS['lung'] = 'Cohort context (advanced lung cancer, use as background knowledge):\n- ECOG and Karnofsky performance status are the strongest clinical predictors of survival in\n  advanced lung cancer; deteriorating performance status implies short expected survival.\n- Significant recent weight loss (cancer cachexia) is an independent adverse prognostic factor.\n- Reduced caloric intake reflects both disease burden and nutritional depletion.\n- Physician- and patient-rated performance scores may disagree; both carry information.'
COHORTS['lung'] = Cohort(
    name='lung', description='Source: North Central Cancer Treatment Group (NCCTG) lung cancer study.\nDescription:\n    Survival of 228 patients with advanced lung cancer, with performance scores rated by both\n    physician and patient, and nutritional indicators. The endpoint is death from any cause.\n', feature_list='\n    age: age in years\n    sex: patient sex\n    ecog: ECOG performance status rated by the physician (0 good, higher is worse)\n    karnofsky_phys: Karnofsky performance score rated by the physician (0-100)\n    karnofsky_pat: Karnofsky performance score rated by the patient (0-100)\n    meal_calories: calories consumed at meals\n    weight_loss: weight loss in the last six months (pounds)\n    ',
    feature_cols=[c + "_raw" if c in COVARIATES['lung'][0] else c
                  for c in list(COVARIATES['lung'][1]) + list(COVARIATES['lung'][0])],
    feature_labels={c + "_raw": c for c in COVARIATES['lung'][0]},
    briefing=BRIEFINGS['lung'],
    _cat=list(COVARIATES['lung'][0]), _num=list(COVARIATES['lung'][1]),
)

BRIEFINGS['whas500'] = 'Cohort context (hospitalised acute myocardial infarction, use as background knowledge):\n- Older age is the dominant determinant of mortality after myocardial infarction.\n- Cardiogenic shock and complete heart block are severe complications carrying very high\n  short-term mortality; congestive heart failure complicating infarction is also strongly adverse.\n- Recurrent infarction implies worse prognosis than a first event.\n- Both hypotension and marked tachycardia at admission indicate haemodynamic compromise.\n- Very low body mass index is associated with frailty and worse outcome.'
COHORTS['whas500'] = Cohort(
    name='whas500', description='Source: Worcester Heart Attack Study (WHAS500).\nDescription:\n    A population-based study of 500 patients hospitalised with acute myocardial infarction in\n    Worcester, Massachusetts, followed for all-cause mortality after admission.\n', feature_list='\n    age: age in years\n    bmi: body mass index at admission\n    sysbp: systolic blood pressure at admission (mmHg)\n    diasbp: diastolic blood pressure at admission (mmHg)\n    hr: initial heart rate (beats per minute)\n    gender: patient sex\n    chf: congestive heart failure complications (yes/no)\n    afb: atrial fibrillation (yes/no)\n    av3: complete heart block (yes/no)\n    cvd: cardiovascular disease history (yes/no)\n    miord: MI order (first or recurrent)\n    mitype: MI type (Q-wave or non-Q-wave)\n    sho: cardiogenic shock (yes/no)\n    ',
    feature_cols=[c + "_raw" if c in COVARIATES['whas500'][0] else c
                  for c in list(COVARIATES['whas500'][1]) + list(COVARIATES['whas500'][0])],
    feature_labels={c + "_raw": c for c in COVARIATES['whas500'][0]},
    briefing=BRIEFINGS['whas500'],
    _cat=list(COVARIATES['whas500'][0]), _num=list(COVARIATES['whas500'][1]),
)

BRIEFINGS['larynx'] = 'Cohort context (laryngeal carcinoma, use as background knowledge):\n- Stage at diagnosis dominates prognosis: stage I disease is usually curable with local therapy,\n  while stage IV disease carries substantially reduced survival.\n- Older age at diagnosis is associated with worse outcome, partly through comorbidity and\n  reduced tolerance of treatment.'
COHORTS['larynx'] = Cohort(
    name='larynx', description='Source: Larynx cancer cohort (Kardaun; Klein and Moeschberger).\nDescription:\n    Ninety male patients with cancer of the larynx treated at a single institution, followed for\n    death from any cause. Disease stage at diagnosis is the principal covariate.\n', feature_list='\n    age: age at diagnosis, years\n    stage: disease stage at diagnosis (I, II, III or IV)\n    ',
    feature_cols=[c + "_raw" if c in COVARIATES['larynx'][0] else c
                  for c in list(COVARIATES['larynx'][1]) + list(COVARIATES['larynx'][0])],
    feature_labels={c + "_raw": c for c in COVARIATES['larynx'][0]},
    briefing=BRIEFINGS['larynx'],
    _cat=list(COVARIATES['larynx'][0]), _num=list(COVARIATES['larynx'][1]),
)

BRIEFINGS['aids'] = 'Cohort context (HIV infection with prior antiretroviral therapy, use as background knowledge):\n- The baseline CD4 count is the dominant prognostic marker: low counts indicate advanced\n  immunosuppression and a high risk of progression to AIDS or death.\n- Poorer Karnofsky performance status indicates greater disease burden.\n- Extensive prior monotherapy is associated with drug resistance and reduced response.\n- Three-drug regimens were substantially more effective than two-drug regimens in this era.'
COHORTS['aids'] = Cohort(
    name='aids', description='Source: AIDS Clinical Trials Group protocol 320 (ACTG 320).\nDescription:\n    A randomised trial in HIV-infected patients with prior antiretroviral therapy comparing\n    two- and three-drug regimens. The endpoint is progression to AIDS or death.\n', feature_list='\n    age: age in years\n    cd4: baseline CD4 T-cell count (cells/mm3)\n    priorzdv: months of prior zidovudine therapy\n    karnof: Karnofsky performance score category\n    sex: patient sex\n    raceth: race/ethnicity category\n    ivdrug: history of intravenous drug use\n    hemophil: haemophilia (yes/no)\n    strat2: CD4 stratum at entry\n    tx: treatment assignment\n    txgrp: treatment group\n    ',
    feature_cols=[c + "_raw" if c in COVARIATES['aids'][0] else c
                  for c in list(COVARIATES['aids'][1]) + list(COVARIATES['aids'][0])],
    feature_labels={c + "_raw": c for c in COVARIATES['aids'][0]},
    briefing=BRIEFINGS['aids'],
    _cat=list(COVARIATES['aids'][0]), _num=list(COVARIATES['aids'][1]),
)

BRIEFINGS['brcamicro'] = 'Cohort context (invasive breast cancer with microarray profiling, use as background knowledge):\n- Estrogen-receptor-positive disease generally carries a more favourable prognosis and predicts\n  benefit from endocrine therapy.\n- Higher histological grade indicates poorer differentiation and worse outcome.\n- Microarray probe-set values are platform-specific normalised intensities; their absolute scale\n  has no fixed clinical interpretation.'
COHORTS['brcamicro'] = Cohort(
    name='brcamicro', description='Source: Breast cancer gene-expression cohort (van de Vijver / Desmedt-style microarray data).\nDescription:\n    198 breast cancer patients with clinical grade and estrogen-receptor status together with\n    normalised microarray expression values for individual probe sets. The endpoint is distant\n    metastasis-free survival.\n', feature_list='\n    er: estrogen receptor status\n    grade: histological grade\n    X200726_at, X200965_s_at, X201068_s_at, X201091_s_at, X201288_at, X201368_at,\n    X201663_s_at, X201664_at, X202239_at, X202240_at:\n        normalised microarray expression values for individual Affymetrix probe sets\n    ',
    feature_cols=[c + "_raw" if c in COVARIATES['brcamicro'][0] else c
                  for c in list(COVARIATES['brcamicro'][1]) + list(COVARIATES['brcamicro'][0])],
    feature_labels={c + "_raw": c for c in COVARIATES['brcamicro'][0]},
    briefing=BRIEFINGS['brcamicro'],
    _cat=list(COVARIATES['brcamicro'][0]), _num=list(COVARIATES['brcamicro'][1]),
)


def register():
    for name, (cat, num) in COVARIATES.items():
        _data.EXTRA_COHORTS[name] = {"cat": cat, "num": num, "dir": DATA_DIR, "csv": CSV[name]}
        _cohorts.PROMPT_DIRS[name] = PROMPTS_DIR
    _train_eval.EXTRA_CFG.update(SETTINGS)
    _model._TIME_UNIT_PER_YEAR.update(TIME_UNIT_PER_YEAR)
    _model._VERSION_ANCHOR_YEARS.update(ANCHOR_YEARS)
    _meta_prompt.COHORT_HORIZONS.update(HORIZONS)
    _cohorts._BRIEFINGS.update(BRIEFINGS)
    _cohorts.COHORTS.update(COHORTS)


register()
