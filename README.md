# medqa_pipeline

Reference implementation of the **data, training and evaluation pipeline** behind a
trilingual (English · French · Moroccan Darija) medical language model and its
knowledge-grounding layer.

This repository is a **curated selection of the core scripts** that produce and evaluate
the corpus and the model. It is published as supporting material for the engineering
thesis chapters it implements — the scripts are the real artifacts referenced there, not
a turnkey application. They read every credential from environment variables; **no API
keys, datasets, or model weights are included.**

---

## What each script does

### `corpus/` — corpus construction & multilingual cleaning *(Ch. 4–5)*
| Script | Role |
|---|---|
| `quality_filter.py` | Applies length, script, NER and deduplication filters to the corpus. |
| `generate_multi_api.py` | Generates synthetic medical Q&A through the provider route. |
| `translate_api_fast.py`, `translate_free.py`, `translate_icliniq.py`, `translate_openc_multi.py` | Translate EN→FR→Darija and normalize multilingual variants. |
| `translate_darija_batch.py` | Batch Darija translation pass. |
| `darija_transformer_scorer.py` | Scores Darija quality through transformer pseudo-perplexity. |

### `shared/` — provider routing & robust parsing
| Script | Role |
|---|---|
| `llm_router.py` | Routes requests across the OpenAI-compatible provider pool (failover chain). |
| `robust_json.py` | Repairs malformed model JSON with layered fallbacks. |

### `validation/` — human validation & agreement
| Script | Role |
|---|---|
| `validation_sampler.py` | Builds human-validation packets for annotators. |
| `kappa_calculator.py` | Computes inter-annotator agreement (κ). |

### `brain_med_cot/` — reasoning corpus *(Ch. 5)*
| Script | Role |
|---|---|
| `build.py` | Constructs the BrainMed-CoT chain-of-thought reasoning corpus. |

### `ckg/` — causal knowledge graph *(Ch. 6)*
| Script | Role |
|---|---|
| `build_ckg_from_dorosz.py` | Extracts the Dorosz causal knowledge graph (drugs, dosages, routes, indications, contraindications). |

### `training/` — fine-tuning & evaluation *(Ch. 8)*
| Script | Role |
|---|---|
| `train_sft.py` | Runs the supervised fine-tuning (QLoRA) stage. |
| `train_sasr.py` | Runs the adaptive SASR / GRPO stage. |
| `sasr_reward.py` | Computes the composite semantic reward (SB-CS) and the Dorosz anti-hallucination penalty. |
| `evaluate.py` | Runs the three-layer evaluation protocol (lexical / semantic / medical-grade). |
| `medical_metrics.py` | Computes entity, hallucination, diagnostic and citation metrics. |

---

## Pipeline at a glance

```
 raw Q&A  ──▶  corpus/quality_filter ──▶  corpus/translate_* ──▶  corpus/darija_transformer_scorer
                                                    │
                       validation/{validation_sampler, kappa_calculator}  (human QC + κ)
                                                    │
   ckg/build_ckg_from_dorosz  ─────────────┐        ▼
                                           ├──▶  brain_med_cot/build  (CoT corpus)
   corpus/generate_multi_api ──────────────┘        │
                                                    ▼
        training/train_sft ──▶ train_sasr (sasr_reward) ──▶ evaluate (medical_metrics)
```

Cross-cutting: `shared/llm_router.py` (provider failover) and `shared/robust_json.py`
(JSON repair) are used by the generation, translation and CoT scripts.

---

## Configuration (environment variables only)

Every credential is read at runtime from the environment — nothing is hardcoded. Set only
the providers you use, e.g. in a local `.env` (which is git-ignored):

```bash
# OpenAI-compatible provider pool (any subset)
CEREBRAS_API_KEY=...
GROQ_API_KEY=...
MISTRAL_API_KEY=...
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...        # Gemini
# optional routing override
BMC_PROVIDER_CHAIN=groq:llama-3.3-70b,cerebras:llama3.3-70b
# data / model hub
HF_TOKEN=...
```

---

## Important notes

- **Reference excerpts, not a runnable package.** Some scripts import sibling modules of
  the full pipeline that are intentionally **not** included here (e.g. `config.py`,
  `data.py`, and helpers under `scripts/datasets/shared/` such as the validators, entity
  extractor and dataset seeds). Treat each file as a documented implementation reference.
- **No data or weights.** Corpora and models live on the Hugging Face Hub (the published
  datasets are referenced inside the scripts); they are not redistributed here.
- **Secrets.** `.env` and key files are git-ignored. Do not commit credentials.

## License

Code is provided for academic/reference use. Released datasets follow their own license
(CC-BY-4.0) on the Hugging Face Hub.
