# LLM-SA: Large Language Model-Enhanced Survival Analysis

Official implementation of **LLM-SA**, a framework that augments structured
clinical covariates with LLM-derived numerical prognostic estimates and
patient-specific clinical narrative embeddings, and integrates them through a
heterogeneous stacking ensemble of survival learners.

> **LLM-Enhanced Survival Analysis**
> [Author Names] · [Venue, Year]
> [[Paper]](#)

---

## Overview

For each subject the framework produces three complementary representations:

- **X_i** - the original structured covariates.
- **N_i = (p_5y, r_2y, c)** - three LLM numeric prognostic fields (5-year
  survival probability, 2-year event risk, and self-reported confidence).
- **E_i ∈ R^1536** - a dense embedding of the LLM's free-text clinical
  narrative **Z_i**.

`(X_i, N_i, E_i)` are consumed by a stacking ensemble of three complementary
base learners, combined by a simplex-constrained meta-learner.

The pipeline has two stages:

1. **Automatic medical textual representation learning** (`text_generation/`).
   A meta prompt generates a diverse set of candidate target prompts; the best
   prompt **P\*** is selected by an ordinal alignment criterion against
   Kaplan-Meier survival-risk labels; **P\*** then generates the final
   per-patient narratives, numeric estimates, and embeddings.
2. **Heterogeneous ensemble survival model** (`llmsa/`). CoxPH, Random Survival
   Forest, and a DeepSurv network with PCA-reduced narrative embedding are
   stacked by a simplex-constrained meta-learner.

Full method details are in [`docs/METHOD.md`](docs/METHOD.md).

---

## Repository structure

```
llmsa/                     # survival method
  data.py                  #   dataset loaders (structured and LLM-augmented)
  metrics.py               #   IPCW C-index and IPCW Integrated Brier Score
  baselines.py             #   CoxPH / RSF / DeepSurv / DeepHit / CoxTime
  model.py                 #   LLM-SA stacking ensemble + simplex meta-learner
  eval.py                  #   multi-seed evaluation harness
  train_eval.py            #   run LLM-SA across cohorts
  run_baselines.py         #   run the structured-covariate baselines (fixed configuration)
  tune_baselines.py        #   per-cohort hyperparameter search for the baselines (Appendix E)
  run_tuned_baselines.py   #   evaluate the selected baseline configurations on all 30 splits
  run_ablation.py          #   component ablation A-D (Table 2, Sec 4.4)
  run_zeroshot.py          #   LLM estimate as a training-free risk score (Sec 4.4)
  run_traintest_gap.py     #   train-fold vs test-fold C-index of the ablation cells (Appendix D)
  run_lambda_sweep.py      #   sensitivity to the meta-learner prior strength
  tune_llmsa.py            #   the baseline tuning protocol applied to LLM-SA (check only)

text_generation/           # automatic textual representation learning (manuscript Sec 3.2)
  cohorts.py               #   per-cohort descriptions, feature lists, briefings
  meta_prompt.py           #   meta prompt -> candidate target prompts
  score.py                 #   ordinal narrative grading (phi: +1 / 0 / -1)
  select_prompt.py         #   KM reference labels -> grade candidates -> P*
  generate_features.py     #   apply P* -> narratives Z_i, numerics N_i, embeddings E_i
  plot_label_construction.py #  Appendix B figure (reference-label construction)

prompts/                   # released candidate target prompts, per cohort
docs/METHOD.md             # method write-up
```

---

## Models

| ID | Model | Type |
|----|-------|------|
| 1 | CoxPH | Cox proportional hazards (baseline) |
| 2 | RSF | Random survival forest (baseline) |
| 3 | DeepSurv | MLP + Cox loss (baseline) |
| 4 | DeepHit | Discrete-time, single event (baseline) |
| 5 | CoxTime | Time-dependent Cox (baseline) |
| - | **LLM-SA** (ours) | Stacking ensemble of CoxPH + RSF + DeepSurv-with-text |

---

## Setup

```bash
git clone https://github.com/jialin666/LLM-enhanced-SA.git
cd LLM-enhanced-SA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # required only for text_generation/
```

The seven cohorts of the paper are expected under `data/` and are not tracked
in git (all are public; see `llmsa/data.py` for the expected file names):
GBSG, METABRIC, SUPPORT (the full 9,105-patient cohort), FLCHAIN, TCGA
(pan-cancer clinical data resource; the 2,000-patient subsample is drawn by the
loader), the UCI Heart Failure Clinical Records, and the Wisconsin Prognostic
Breast Cancer (WPBC) cohort. Set `LLMSA_REPO_ROOT` if `data/` lives elsewhere.

The per-patient LLM outputs used for every reported result (narratives,
numeric estimates and embeddings) are released as a GitHub Release asset; see
**Released LLM outputs** below. With them in place, every number in the paper
can be reproduced without an OpenAI key.

---

## Usage

### The v5 unified pipeline

Each target prompt carries two literal placeholders — `{BRIEFING}` (cohort
background knowledge) and `{FEATURES}` (one patient's covariates) — and
instructs the LLM to return, in **one call**, the clinical narrative followed
by a single JSON block with the numeric prognostic estimates
(`estimated_5yr_survival_prob`, `estimated_2yr_event_risk`, `confidence`).

Prompt selection is leakage-controlled: the Kaplan–Meier fit, the tertile
cut-points, and the graded patient subset are all restricted to a fixed
**development hold-out** (15%, event-stratified, `llmsa.data.dev_holdout_positions`)
that the evaluation loaders pin to the training fold of every per-seed split,
so no validation or test outcome can influence the selected prompt.

The cohort briefing comes in two arms (`--briefing`):

| arm | content | artifact tag |
|---|---|---|
| `sanitized` (default) | a fixed description of the cohort and its covariates; **no cohort-matched outcome statistics** (Appendix E of the paper) | `v5` |
| `none` | empty slot (no-briefing run) | `v5nobrief` |

`--tag-suffix` appends a string to the artifact tag; the reported results use
the tag `v5fixmini` (sanitized briefing, `gpt-4o-mini`) for the six larger
cohorts and `v5` for Heart Failure and WPBC.

**1. Select the optimal target prompt P\*** for a cohort (meta prompt →
candidate generation → KM-ordinal grading on the development hold-out →
selection):

```bash
python -m text_generation.select_prompt --dataset gbsg              # gpt-4o for grading (GBSG, METABRIC, SUPPORT)
python -m text_generation.select_prompt --dataset flchain --narrative-model gpt-4o-mini --grader-model gpt-4o-mini
# variants:
#   --events-only          restrict graded labels to observed events
#   --briefing none        grade with an empty briefing slot
#   --legacy-full-cohort   as-submitted behaviour (leaks; comparison only)
```

This writes the candidate prompts to `prompts/gbsg_target_prompts.json`, their
grades to `prompts/gbsg_templates_grades.json`, and a secondary
numeric-agreement score to `prompts/gbsg_numeric_grades.json`. Grades are a
per-run artefact (LLM outputs vary between runs) and are not tracked in git.

**2. Generate per-patient features** with the selected prompt (one call per
patient: narrative Z_i + numerics N_i; then embeddings E_i):

```bash
# as reported (six larger cohorts; SUPPORT on the full 9,105 patients):
python -m text_generation.generate_features --dataset gbsg --model gpt-4o-mini --tag-suffix fixmini
python -m text_generation.generate_features --dataset support --model gpt-4o-mini --tag-suffix fixmini --full-support
# Heart Failure / WPBC (tag v5):
python -m text_generation.generate_features --dataset heartfailure --model gpt-4o-mini
# no-briefing arm:
python -m text_generation.generate_features --dataset gbsg --model gpt-4o-mini --briefing none
```

Generation uses temperature 0.4 and one call per patient; the outputs behind
the paper were generated on 8-11 September 2026.

By default this uses the highest-graded prompt P\* from step 1. To skip grading
and use a specific candidate directly, pass its id:
`--prompt-id <id>` (ids are the keys in `prompts/gbsg_target_prompts.json`).

This writes `data/<tag>_structured_gbsg.csv`, `data/embeddings_<tag>_gbsg.npy`
and the narratives under `data/<tag>/narratives/gbsg/` (tag = `v5`, `v5fixmini`,
`v5nobrief`, ...). Loaders select a generation via
`load_dataset_v4(..., feature_version=<tag>)`.

**3. Table 2: LLM-SA and the component ablation** (30 splits, seeds 0-29;
cells A = covariates only, B = + numeric estimates, C = + narrative embedding,
D = full framework; `--cindex-tau grid` truncates Uno's C-index at the horizon
of the integrated Brier score, as in the paper):

```bash
python -m llmsa.run_ablation --seeds 30 --datasets metabric,gbsg,support,flchain,tcga,tcgafull \
    --feature-version v5fixmini --full-support --cindex-tau grid
python -m llmsa.run_ablation --seeds 30 --datasets heartfailure,wpbc --feature-version v5 --cindex-tau grid
```

The meta-learner prior strength defaults to `--lam 1.0`, the value used for
every reported result. `llmsa.train_eval` runs the full framework alone with
the same flags.

**Baselines** (tuned per cohort on the validation fold, then evaluated on the
same 30 splits; grids in Appendix E of the paper):

```bash
python -m llmsa.tune_baselines --datasets metabric,gbsg,support,flchain,tcga,tcgafull \
    --models coxph,rsf,deepsurv,deephit,coxtime --tune-seeds 5 --seeds 30 --full-support
python -m llmsa.tune_baselines --datasets heartfailure,wpbc --tune-seeds 20 --seeds 30
python -m llmsa.run_tuned_baselines --logs <tuning log> --datasets metabric,gbsg --seeds 30 --full-support --cindex-tau grid
```

`llmsa.run_baselines` runs the fixed (untuned) configuration of the submitted
version and is kept for comparison only.

**Further analyses** reported in Sec 4.4 and the appendices:

```bash
python -m llmsa.run_zeroshot --datasets gbsg,metabric,flchain,support --feature-version v5fixmini --full-support
python -m llmsa.run_traintest_gap --datasets gbsg,metabric,flchain,support --feature-version v5fixmini --full-support
python -m llmsa.run_ablation --datasets tcgafull --feature-version v5fixmini --cindex-tau grid --train-n 300   # learning curve
python -m llmsa.run_ablation --datasets tcgafull --feature-version v5fixmini --cindex-tau grid --cohort-n 2000 --cohort-seed 1   # random 2,000-patient cohorts
python -m llmsa.run_lambda_sweep --dataset tcgafull --feature-version v5fixmini
python -m text_generation.plot_label_construction --outdir figures
```

Both C-index and IBS are reported under Uno's IPCW estimator so every model is
scored under identical definitions.

### SUPPORT sample size

Every reported experiment uses the full 9,105-patient SUPPORT cohort
(`--full-support`). The loader's default 1,500-patient event-stratified
subsample was a development-time convenience and is not used in the paper.

---

## Released LLM outputs

The per-patient outputs behind every reported number are published as the
release asset `llmsa-llm-outputs.tar.gz` (280 MB) of release **v1.1**:
<https://github.com/jialin666/LLM-enhanced-SA/releases/download/v1.1/llmsa-llm-outputs.tar.gz>
(release page: <https://github.com/jialin666/LLM-enhanced-SA/releases/tag/v1.1>).

```bash
curl -L -o llmsa-llm-outputs.tar.gz \
  https://github.com/jialin666/LLM-enhanced-SA/releases/download/v1.1/llmsa-llm-outputs.tar.gz
tar -xzf llmsa-llm-outputs.tar.gz          # unpacks into data/
```

The archive contains, for each cohort, the
structured numeric estimates (`data/<tag>_structured_<cohort>.csv`), the
narrative embeddings (`data/embeddings_<tag>_<cohort>.npy`) and the raw
narratives with the parsed JSON (`data/<tag>/narratives/<cohort>/`), for the
reported arm (`v5fixmini`, or `v5` for Heart Failure and WPBC) and for the
no-briefing arm (`v5nobrief` / `v5nobriefmini`). The full text of every cohort
briefing is in `text_generation/cohorts.py`.

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt`. A GPU is optional; the models run on CPU.
