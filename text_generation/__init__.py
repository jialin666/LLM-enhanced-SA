"""Automatic medical textual representation learning (manuscript Sec 2.2).

Pipeline:
    meta_prompt.py       P_meta -> candidate target prompts C
    score.py             ordinal narrative grading (phi: +1/0/-1)
    select_prompt.py     KM reference labels -> grade candidates -> select P*
    generate_features.py apply P* -> narratives Z_i, numerics N_i, embeddings E_i
    cohorts.py           per-cohort descriptions, feature lists, briefings
"""
