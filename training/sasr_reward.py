"""
SASR composite reward — semantic (SB_CS) content + format + anti-hallucination.

WHY THIS EXISTS
---------------
The first SASR run plateaued. Root cause: the old `content_reward` used ROUGE-L against
the gold answer. But SFT-Big has a low-ROUGE / high-SB_CS profile — it *paraphrases*
(BLEU 0.079, ROUGE-1 0.38, SB_CS 0.776). A ROUGE-L reward therefore punishes the exact
skill the model learned, so the GRPO signal was adversarial/noisy and the reward never
moved. This module replaces ROUGE-L with **SB_CS (sentence-BERT cosine)** — it rewards
meaning, not word overlap — and adds an **anti-hallucination penalty** that discourages
introducing drugs the gold answer never mentioned (cross-checked against the Dorosz KG).

DESIGN
------
- Pure, configurable, unit-testable. The semantic scorer is pluggable (`configure(content_scorer=...)`)
  so the format + anti-hallucination logic can be tested without downloading the 1 GB
  multilingual sentence model.
- Module-level config set once via `configure(...)` in train_sasr.main(); the hot path
  `composite_reward(prediction, gold, prompt_text)` then needs no extra plumbing through
  the GRPO loop.

reward = W_FORMAT·format + W_CONTENT·sbcs(answer, gold) − hallucination_penalty
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

# Reasoning tags — kept local so this module is importable without config.py (eases testing).
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL)
_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.DOTALL)

# ── Module configuration (set once via configure()) ────────────────────────
_DEFAULT_SENT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
_sent_model = None
_sent_model_name = _DEFAULT_SENT_MODEL
_sent_lock = threading.Lock()

_drug_vocab: frozenset[str] = frozenset()
_content_scorer: Optional[Callable[[str, str], float]] = None  # override for tests

_W_FORMAT = 0.20
_W_CONTENT = 0.80
_W_HALLU = 0.30           # max penalty subtracted from the reward
_HALLU_PER_DRUG = 0.10    # penalty per illegitimate drug introduced


def configure(
    *,
    drug_vocab: Optional[Iterable[str]] = None,
    sent_model_name: Optional[str] = None,
    content_scorer: Optional[Callable[[str, str], float]] = None,
    w_format: Optional[float] = None,
    w_content: Optional[float] = None,
    w_hallu: Optional[float] = None,
) -> None:
    """Configure the reward once before training. All args optional."""
    global _drug_vocab, _sent_model_name, _content_scorer, _W_FORMAT, _W_CONTENT, _W_HALLU
    if drug_vocab is not None:
        _drug_vocab = frozenset(d.lower().strip() for d in drug_vocab if d and len(d.strip()) >= 4)
    if sent_model_name:
        _sent_model_name = sent_model_name
    if content_scorer is not None:
        _content_scorer = content_scorer
    if w_format is not None:
        _W_FORMAT = w_format
    if w_content is not None:
        _W_CONTENT = w_content
    if w_hallu is not None:
        _W_HALLU = w_hallu


def load_drug_vocab_from_kg(kg_path: str | Path) -> set[str]:
    """Extract the set of drug labels from a Dorosz CKG json (nodes with type == 'drug')."""
    p = Path(kg_path)
    if not p.is_file():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    out: set[str] = set()
    for n in data.get("nodes", []):
        if n.get("type") == "drug":
            label = (n.get("label") or "").strip()
            if len(label) >= 4:
                out.add(label)
    return out


# ── Primitives ──────────────────────────────────────────────────────────────

def extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text or "")
    return (m.group(1).strip() if m else (text or "").strip())


def format_reward(text: str) -> float:
    has_think = bool(_THINK_RE.search(text or ""))
    has_answer = bool(_ANSWER_RE.search(text or ""))
    if has_think and has_answer:
        return 1.0
    if has_answer:
        return 0.5
    return 0.0


def _get_sent_model():
    global _sent_model
    if _sent_model is not None:
        return _sent_model
    with _sent_lock:
        if _sent_model is None:
            from sentence_transformers import SentenceTransformer
            _sent_model = SentenceTransformer(_sent_model_name)
    return _sent_model


def sbcs(pred: str, gold: str) -> float:
    """Sentence-BERT cosine similarity in [0, 1] (negative cosines clamped to 0)."""
    if not pred or not gold:
        return 0.0
    if _content_scorer is not None:
        return float(_content_scorer(pred, gold))
    import numpy as np
    model = _get_sent_model()
    embs = model.encode([pred, gold], normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    cos = float(np.dot(embs[0], embs[1]))
    return max(0.0, min(1.0, cos))


def _find_drugs(text: str, vocab: frozenset[str]) -> set[str]:
    """Word-boundary, case-insensitive search for vocab drugs present in `text`."""
    if not text or not vocab:
        return set()
    low = text.lower()
    hits = set()
    for d in vocab:
        if re.search(r"\b" + re.escape(d) + r"\b", low):
            hits.add(d)
    return hits


def anti_hallucination_penalty(pred_answer: str, gold: str) -> float:
    """
    Penalize drugs the prediction introduces that the gold answer never mentioned.

    Proxy for hallucination: a Dorosz drug present in the prediction but absent from the
    gold is "introduced". This is intentionally conservative (only Dorosz-known drugs are
    checked), matching the agent's anti-hallucination logic in services/agent/soap.py.
    """
    if not _drug_vocab:
        return 0.0
    pred_drugs = _find_drugs(pred_answer, _drug_vocab)
    if not pred_drugs:
        return 0.0
    gold_drugs = _find_drugs(gold, _drug_vocab)
    introduced = pred_drugs - gold_drugs
    return min(_W_HALLU, _HALLU_PER_DRUG * len(introduced))


def composite_reward(prediction: str, gold: str, prompt_text: str = "") -> tuple[float, dict]:
    """
    Returns (reward, parts). `prompt_text` is accepted for signature-compatibility with the
    old reward but is no longer used (the old length-heuristic bonus is dropped — SB_CS
    already captures adequacy far better).
    """
    r_format = format_reward(prediction)
    pred_ans = extract_answer(prediction)
    r_content = sbcs(pred_ans, gold)
    r_hallu = anti_hallucination_penalty(pred_ans, gold)

    reward = _W_FORMAT * r_format + _W_CONTENT * r_content - r_hallu
    parts = {
        "r_format": r_format,
        "r_content": r_content,
        "r_hallu": r_hallu,
        "r_total": reward,
    }
    return reward, parts
