# LLM-Enhanced Survival Analysis

Official implementation of the paper:

> **[Paper Title]**
> [Author Names] · [Venue, Year]
> [[Paper]](#) · [[arXiv]](#)

---

## Overview

This repository provides the implementation of an LLM-enhanced survival analysis framework that integrates large language model (LLM) generated clinical narratives and embeddings with deep survival analysis models. We benchmark against a comprehensive suite of classical and modern survival analysis baselines.

### Key Contributions

- A pipeline that uses GPT-4o to generate diverse, high-quality clinical narratives from structured patient features.
- OpenAI embedding vectors (1536-dim) derived from narratives, fused with structured data for improved survival prediction.
- A differentiable **Integrated Brier Score (IBS) loss** with Inverse Probability of Censoring Weighting (IPCW) for end-to-end training.
- A unified evaluation framework comparing 7+ models on C-index and IBS.

---

## Models

| ID | Model | Type |
|----|-------|------|
| 1 | CoxPH | Traditional (Cox Proportional Hazards) |
| 2 | RSF | Traditional (Random Survival Forest) |
| 3 | DeepSurv | Deep Learning (MLP + Cox loss) |
| 4 | CoxTime | Deep Learning (discrete-time Cox) |
| 5 | DeepHit | Deep Learning (competing risks, PMF) |
| 6 | DASA | Deep Learning (LSTM-based) |
| 7 | Transformer | Deep Learning (causal Transformer) |
| — | **LLMSA** (Ours) | LLM-enhanced Transformer |

---

## Repository Structure

```
.
├── model_1_CoxPH.py              # Cox Proportional Hazards baseline
├── model_2_RSF.py                # Random Survival Forest baseline
├── model_3_DeepSurv.py           # DeepSurv
├── model_4_CoxTime.py            # CoxTime (discrete-time)
├── model_5_DeepHit.py            # DeepHit (competing risks)
├── model_6_DASA.py               # DASA (LSTM-based)
├── model_7_Transformer.py        # Transformer-based survival analysis
├── model_LLMSA.py                # Our proposed LLM-enhanced model
│
├── 1_generate_feature_openAI.py  # Step 1: generate narratives & embeddings
├── GBSG_generate_prompts_3.py    # Prompt template generation via meta-prompting
├── GBSG_score_function.py        # LLM-based narrative quality scoring
├── GBSG_test_templates.py        # Template validation and selection pipeline
│
├── dataset_sa.py                 # Dataset loader (structured + embedding inputs)
├── dataloader_drsa.py            # Data loader utilities
├── concordance.py                # C-index computation
├── utils.py                      # Shared utilities (device, time grid, AUC)
├── util_dasa.py                  # DASA-specific utilities
│
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/jialin666/LLM-enhanced-SA.git
cd LLM-enhanced-SA
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set your OpenAI API key** (required for LLM feature generation only)
```bash
export OPENAI_API_KEY=sk-...
```

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- See `requirements.txt` for the full dependency list.

Hardware: GPU (CUDA or Apple MPS) is recommended but not required. The code automatically selects the best available device via `utils.get_best_device()`.

---

