"""Final feature generation with the selected prompt (v5 unified design).

Using the selected target prompt P* -- either the highest-graded candidate from
a local ``select_prompt.py`` run, or a specific one chosen with ``--prompt-id``
from ``prompts/<dataset>_target_prompts.json`` -- generate, for every patient in
a cohort, in ONE LLM call:

  * the free-text clinical narrative  Z_i = LLM(X_i | P*), and
  * the numeric prognostic estimates  N_i = (p_5y, r_2y, c), parsed from the
    JSON block that P* instructs the model to append after the narrative.

The narrative is then embedded:  E_i = EMB(Z_i) in R^1536.

The cohort briefing filled into the template's ``{BRIEFING}`` slot is selected
with ``--briefing``:

    sanitized  (default) the fixed cohort briefing, no
               cohort-matched outcome statistics       -> tag ``v5``
    none       empty slot (no-briefing ablation)       -> tag ``v5nobrief``

Outputs (under data/, consumed by ``llmsa.data.load_dataset_v4(...,
feature_version=<tag>)``):
    data/<tag>_structured_<dataset>.csv   patient_idx + the three N_i fields
                                          (columns ``llm_*_<tag>``)
    data/embeddings_<tag>_<dataset>.npy   (n, 1536) narrative embeddings E_i
    data/<tag>/narratives/<dataset>/      one .txt narrative per patient (cache)

Usage:
    python -m text_generation.generate_features --dataset gbsg
    python -m text_generation.generate_features --dataset gbsg --briefing none
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI

from llmsa.data import (
    _csv_path, _read_cohort_csv, _support_subsample, SUPPORT_SUBSAMPLE_SIZE,
)
from .cohorts import COHORTS
from .meta_prompt import (
    BRIEFING_EMPTY, NUMERIC_FIELDS, COHORT_HORIZONS, horizon_fields,
    retarget_horizons, render_features_block, fill_template,
    generate_narrative_and_numerics,
)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

_BRIEFING_TAGS = {"sanitized": "v5", "none": "v5nobrief"}


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def _embed(client, texts, model="text-embedding-3-small"):
    out = []
    BATCH = 64
    for k in range(0, len(texts), BATCH):
        chunk = [t or " " for t in texts[k:k + BATCH]]
        rsp = client.embeddings.create(model=model, input=chunk)
        out.extend(np.asarray(d.embedding, dtype=np.float32) for d in rsp.data)
    return np.stack(out, axis=0)


def _load_selected_prompt(dataset: str, prompt_id=None) -> str:
    """Return the target-prompt template to use for generation.

    ``prompt_id`` selects a specific candidate from ``<dataset>_target_prompts.json``.
    If not given, the selection defers to a local grades file
    ``<dataset>_templates_grades.json`` (written by ``select_prompt.py``) and the
    highest-graded candidate P* is used. Grades are a per-run artefact and are not
    shipped with the repository.
    """
    with open(os.path.join(PROMPTS_DIR, f"{dataset}_target_prompts.json")) as f:
        prompts = json.load(f)

    if prompt_id is not None:
        key = str(prompt_id)
        if key not in prompts:
            raise KeyError(f"prompt id {key!r} not in {dataset}_target_prompts.json "
                           f"(available: {list(prompts)[:5]}...)")
        return prompts[key]["template"]

    grades_path = os.path.join(PROMPTS_DIR, f"{dataset}_templates_grades.json")
    if not os.path.exists(grades_path):
        raise FileNotFoundError(
            f"No selection found for {dataset}. Either run "
            f"`python -m text_generation.select_prompt --dataset {dataset}` to grade "
            f"the candidates and select P*, or pass --prompt-id to choose one directly."
        )
    with open(grades_path) as f:
        grades = json.load(f)
    best_id = max(grades, key=grades.get)
    return prompts[best_id]["template"]


def _indexed_cohort(dataset: str, full_support: bool = False):
    """Full cohort dataframe with a stable ``_patient_idx`` matching the loaders.

    ``full_support=True`` generates for the full 9105-patient SUPPORT cohort;
    the default keeps the fixed 1500-patient subsample.
    """
    cohort = COHORTS[dataset]
    df = _read_cohort_csv(dataset)
    # keep any display-label columns referenced by feature_cols (e.g. FLCHAIN's
    # raw sex/mgus labels), mirroring Cohort.load_full
    extra = [c for c in cohort.feature_cols
             if c not in cohort._cat + cohort._num and c in df.columns]
    df = df[cohort._cat + cohort._num + extra + ["time", "event"]]
    df = df.dropna(subset=["time", "event"]).reset_index(drop=True)
    df["time"] = df["time"].astype(float)
    df["event"] = df["event"].astype(int).clip(0, 1)
    df["_patient_idx"] = df.index.copy()
    if dataset == "support" and len(df) > SUPPORT_SUBSAMPLE_SIZE and not full_support:
        df = _support_subsample(df)
    return cohort, df


def generate_for_dataset(dataset: str, model="gpt-4o",
                         embedding_model="text-embedding-3-small",
                         temperature: float = 0.4,
                         max_patients=None, prompt_id=None,
                         briefing_mode: str = "sanitized",
                         workers: int = 8, full_support: bool = False,
                         tag_suffix: str = "", horizons=None):
    cohort, df = _indexed_cohort(dataset, full_support=full_support)
    if max_patients is not None:
        df = df.iloc[:max_patients]
    template = _load_selected_prompt(dataset, prompt_id=prompt_id)
    if briefing_mode == "sanitized":
        briefing = cohort.briefing
    elif briefing_mode == "none":
        briefing = BRIEFING_EMPTY
    else:
        raise ValueError(f"unknown briefing mode: {briefing_mode!r}")
    # ``tag_suffix`` isolates artefacts AND the per-patient cache for runs that
    # differ in something other than the briefing (e.g. a different generating
    # model), so a cached run is never silently reused across configurations.
    tag = _BRIEFING_TAGS[briefing_mode] + tag_suffix
    client = _client()

    # Cohort-matched horizons. The default prompt asks for 5-year survival and
    # 2-year event risk; on a cohort whose entire follow-up is shorter than
    # that, both anchors are unobservable. ``--horizons short,long`` (or an
    # entry in COHORT_HORIZONS) retargets the numeric block. The JSON keys the
    # model returns change; the stored column names do not.
    gen_fields = None
    if horizons:
        short_label, long_label = horizons
        template = retarget_horizons(template, short_label, long_label)
        gen_fields = horizon_fields(short_label, long_label)
        print(f"  [{dataset}] horizons retargeted -> survival@{long_label}, "
              f"event risk@{short_label}", flush=True)

    narr_dir = _csv_path("data", tag, "narratives", dataset)
    os.makedirs(narr_dir, exist_ok=True)

    num_cols = {f: f"llm_{short}_{tag}" for f, short in zip(
        NUMERIC_FIELDS, ("5yr_surv", "2yr_event_risk", "confidence"))}

    n = len(df)
    df_rows = list(df.iterrows())

    def _gen_one(k):
        """Generate (or read from cache) one patient.

        Returns ``(k, pi, narrative, numerics, parse_fail)``; on a transport
        error returns ``numerics=None, narrative=None`` so the pool keeps
        going and the caller can retry that patient (a network blip must not
        discard a nearly-complete run).
        """
        try:
            return _gen_one_inner(k)
        except (KeyError, AttributeError, TypeError, ValueError, IndexError):
            # deterministic bug (bad column, bad template, ...) -- retrying
            # 6k patients four times would only waste time; fail immediately.
            raise
        except Exception as exc:  # transport/API error, not a parse failure
            _, row = df_rows[k]
            print(f"  [{dataset}] transient failure on patient "
                  f"{int(row['_patient_idx'])}: {type(exc).__name__}", flush=True)
            return k, int(row["_patient_idx"]), None, None, False

    def _gen_one_inner(k):
        _, row = df_rows[k]
        pi = int(row["_patient_idx"])
        npath = os.path.join(narr_dir, f"patient_{pi:05d}.txt")
        jpath = os.path.join(narr_dir, f"patient_{pi:05d}.json")
        if os.path.exists(npath) and os.path.exists(jpath):
            with open(npath) as f:
                narrative = f.read()
            with open(jpath) as f:
                numerics = json.load(f)
            return k, pi, narrative, numerics, False
        features_block = render_features_block(cohort.sample_dict(row), fmt="bullet")
        filled = fill_template(template, features_block, briefing=briefing)
        narrative, numerics = generate_narrative_and_numerics(
            filled, model=model, temperature=temperature, fields=gen_fields)
        fail = numerics is None
        if fail:
            numerics = {f: 0.5 for f in NUMERIC_FIELDS}
        with open(npath, "w") as f:
            f.write(narrative)
        with open(jpath, "w") as f:
            json.dump(numerics, f)
        return k, pi, narrative, numerics, fail

    narratives = [None] * n
    rows = [None] * n
    n_parse_fail, done = 0, 0

    def _collect(res):
        nonlocal n_parse_fail, done
        k, pi, narrative, numerics, fail = res
        if numerics is None and narrative is None:
            return k  # transient failure -> retry later
        narratives[k] = narrative
        rec = {"patient_idx": pi}
        for f_name, col in num_cols.items():
            rec[col] = float(numerics.get(f_name, 0.5))
        rows[k] = rec
        if fail:
            n_parse_fail += 1
            print(f"  [{dataset}] WARN patient {pi}: numeric block unparseable, "
                  f"using 0.5 defaults", flush=True)
        done += 1
        if done % 25 == 0 or done == n:
            print(f"  [{dataset}] {done}/{n}", flush=True)
        return None

    pending = list(range(n))
    for attempt in range(4):
        if not pending:
            break
        if attempt:
            wait = 30 * attempt
            print(f"  [{dataset}] retry pass {attempt} for {len(pending)} patients "
                  f"in {wait}s ...", flush=True)
            time.sleep(wait)
        failed = []
        # Bounded batches keep the progress counter close to real completion.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for start in range(0, len(pending), 200):
                batch = pending[start:start + 200]
                futures = [ex.submit(_gen_one, k) for k in batch]
                for fut in as_completed(futures):
                    again = _collect(fut.result())
                    if again is not None:
                        failed.append(again)
        pending = failed
    if pending:
        raise RuntimeError(
            f"[{dataset}] {len(pending)} patients still failing after retries "
            f"(indices {pending[:10]}...). Nothing written; re-run to resume from cache.")

    struct_out = _csv_path("data", f"{tag}_structured_{dataset}.csv")
    pd.DataFrame(rows).to_csv(struct_out, index=False)
    print(f"[{dataset}] structured N_i ({len(rows)} rows, {n_parse_fail} parse "
          f"fallbacks) -> {struct_out}")

    emb = _embed(client, narratives, model=embedding_model)
    emb_out = _csv_path("data", f"embeddings_{tag}_{dataset}.npy")
    np.save(emb_out, emb)
    print(f"[{dataset}] embeddings E_i {emb.shape} -> {emb_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(COHORTS))
    p.add_argument("--model", default="gpt-4o",
                   help="chat model for the unified narrative+numerics call")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-patients", type=int, default=None)
    p.add_argument("--prompt-id", default=None,
                   help="candidate id from <dataset>_target_prompts.json to use; "
                        "if omitted, the highest-graded prompt from a local "
                        "select_prompt run (P*) is used")
    p.add_argument("--briefing", choices=list(_BRIEFING_TAGS), default="sanitized",
                   help="briefing filled into {BRIEFING}: sanitized (default, tag v5), "
                        "none (tag v5nobrief)")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent generation calls (thread pool)")
    p.add_argument("--full-support", action="store_true",
                   help="generate for the full 9105-patient SUPPORT cohort "
                        "(default: the fixed 1500-patient subsample)")
    p.add_argument("--horizons", default=None,
                   help="cohort-matched horizon pair 'short,long' (e.g. '90d,180d') "
                        "replacing the default 2-year risk / 5-year survival "
                        "anchors; use where follow-up is shorter than 5 years. "
                        "Defaults to COHORT_HORIZONS[dataset] when defined.")
    p.add_argument("--tag-suffix", default="",
                   help="suffix appended to the artefact tag and cache directory, "
                        "e.g. 'mini' -> v5mini_structured_<ds>.csv (use when varying "
                        "the generating model so caches are not shared)")
    args = p.parse_args()
    generate_for_dataset(
        args.dataset, model=args.model, embedding_model=args.embedding_model,
        temperature=args.temperature, max_patients=args.max_patients,
        prompt_id=args.prompt_id, briefing_mode=args.briefing, workers=args.workers,
        full_support=args.full_support, tag_suffix=args.tag_suffix,
        horizons=(tuple(args.horizons.split(",")) if args.horizons
                  else COHORT_HORIZONS.get(args.dataset)),
    )
