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
  train_eval.py            #   run LLM-SA across the three cohorts
  run_baselines.py         #   run the structured-covariate baselines

text_generation/           # automatic textual representation learning (Sec 2.2)
  cohorts.py               #   per-cohort descriptions, feature lists, briefings
  meta_prompt.py           #   meta prompt -> candidate target prompts
  score.py                 #   ordinal narrative grading (phi: +1 / 0 / -1)
  select_prompt.py         #   KM reference labels -> grade candidates -> P*
  generate_features.py     #   apply P* -> narratives Z_i, numerics N_i, embeddings E_i

prompts/                   # released candidate prompts + grades, per cohort
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

Datasets (GBSG, METABRIC, SUPPORT) are expected under `data/` and are not
tracked in git. Set `LLMSA_REPO_ROOT` if `data/` lives elsewhere.

---

## Usage

**1. Select the optimal target prompt P\*** for a cohort (meta prompt →
candidate generation → KM-ordinal grading → selection):

```bash
python -m text_generation.select_prompt --dataset gbsg
```

This writes `prompts/gbsg_target_prompts.json` and
`prompts/gbsg_templates_grades.json`. The released prompts are already in
`prompts/`.

**2. Generate per-patient features** with the selected prompt (narratives Z_i,
numerics N_i, embeddings E_i):

```bash
python -m text_generation.generate_features --dataset gbsg
```

This writes `data/v4_structured_gbsg.csv` and `data/embeddings_v4_gbsg.npy`.

**3. Train and evaluate LLM-SA** across the three cohorts (10 seeds):

```bash
python -m llmsa.train_eval --seeds 10 --datasets metabric,gbsg,support --lam 0.3
```

**Baselines**:

```bash
python -m llmsa.run_baselines --seeds 10 --models coxph,rsf,deepsurv,deephit,coxtime
```

Both C-index and IBS are reported under Uno's IPCW estimator so every model is
scored under identical definitions.

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt`. A GPU is optional; the models run on CPU.
