"""Run any llmsa / text_generation module with the development cohorts registered.

    python -m development_cohorts.run <module> [arguments ...]

e.g.
    python -m development_cohorts.run llmsa.run_ablation --seeds 30 \
        --datasets rotterdam,aids,whas500,lung,veteran,larynx,brcamicro \
        --feature-version v5 --cindex-tau grid
"""
import runpy
import sys

import development_cohorts  # noqa: F401  (registers the cohorts)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
