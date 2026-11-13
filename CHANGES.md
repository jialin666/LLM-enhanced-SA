# Changes since v1.1

This file lists the changes and improvements made since the v1.1 release
(tag `v1.1`, commit `b9a23f5`).

## 1. SUPPORT covariates are passed to the LLM in clinical units

**Problem.** `llmsa/data.py` loaded SUPPORT from
`2_support_{train,test}_new_with_embeddings_preprocessed_DL.csv`. In those files
the numeric covariates were already z-scored and `sex` / `dzclass` were integer
codes. The same values were rendered into the LLM prompt, which describes them in
clinical units. For example, age 40.9 years appeared as `age: -1.39`, and
`sex: 1` (female) was read by the LLM as male.

**Fix.**
- `llmsa/data.py`: the new `_read_support_raw()` loads `2_support_{train,test}_new.csv`
  (raw values, same patients and row order). `sex` / `dzclass` keep the integer
  codes used by the models (`SUPPORT_CODES`, identical to the old coding). The
  original labels are kept in `sex_raw` / `dzclass_raw`. The loader checks that
  the raw and legacy files are row-aligned.
- `text_generation/cohorts.py`: the SUPPORT prompt shows the labels
  (`sex: female`, `dzclass: COPD/CHF/Cirrhosis`). Units are added to the feature
  descriptions. Floats are rounded to 2 decimals **in the prompt only**, through
  the new `Cohort.display_decimals` field.
- `text_generation/meta_prompt.py`: SUPPORT is asked for 60-day event risk and
  180-day survival (`COHORT_HORIZONS["support"] = ("60d", "180d")`) instead of
  2-year / 5-year.

**Effect on structured-only results: none.** Every numeric column of the old file
is an exact affine transform of the raw column. The models standardise numerics
on the training fold, and tree models are invariant to monotone transforms.
After fold-wise scaling, the old and new inputs differ by at most 5e-13. The
categorical codes and the 1,500-patient subsample are identical. Baselines and
ablation cell A therefore need no re-run. Only the LLM-dependent features and
cells (B, C, D, no-briefing, zero-shot) were regenerated (tags `v5rawmini` and
`v5nobriefrawmini`).

**Other cohorts checked.** METABRIC, GBSG, FLCHAIN, TCGA (both), Heart Failure
and WPBC already render raw clinical values, with categoricals either as labels
or with their 0/1 meaning stated in the feature list.

## 2. Statistical analysis

`analysis/corrected_stats.py` recomputes every reported comparison from the
per-split results with the Nadeau–Bengio corrected resampled t-test
(overlapping training sets), applies Holm and Benjamini–Hochberg adjustment
within comparison families and globally, and adds tests across cohorts (exact
Wilcoxon signed-rank test over the seven independent cohorts, random-effects
pooling).

    python analysis/corrected_stats.py --results results/per_split_results.csv --out stats/

## 3. Cohort-matched horizons and the v1.2 release asset

- SUPPORT features are generated from the covariates in clinical units
  (`v5rawmini`), and SUPPORT and Heart Failure ask the LLM for horizons within
  their follow-up (60/180 days and 90/180 days, `v5hz` for Heart Failure).
- The v1.2 release asset adds these features, the with-briefing arm of the
  cohort briefing comparison (`v5`, `v5mini`), the SUPPORT arm without the briefing
  on the clinical-unit inputs (`v5nobriefrawmini`), the development cohorts, and
  `results/per_split_results.csv` with the per-split test results of every
  reported run.

## 4. Development cohorts

Seven further public cohorts evaluated during method development (Rotterdam,
WHAS500, Veteran, Larynx, NCCTG Lung, ACTG320, breast-cancer microarray) are
kept separate from the benchmark in `development_cohorts/`, with their cohort
definitions, candidate prompts and the LLM-as-base-learner experiment
(`run_llm_base.py`). The main loaders define only the benchmark cohorts;
`python -m development_cohorts.run <module> [arguments]` runs any script on the
development cohorts.

## 5. Also added to the repository

- `llmsa/tune_llmsa_pca_batch.py`: validation-fold tuning of the LLM-SA PCA
  dimension and batch size, with the same protocol as the baselines.
