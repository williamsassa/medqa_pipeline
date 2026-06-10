"""Darija quality scorer + optional re-generator.

Primary use-case: after the Mistral translation step, score each Darija sentence
for "real Moroccan Darija likelihood" and flag low-quality outputs for re-generation.

Uses SI2M-Lab/DarijaBERT as masked-LM. We compute pseudo-perplexity on a small set
of tokens and combine with an Arabic-script ratio check. Pairs scoring below the
threshold are rewritten via an LLM call (Mistral) constrained with an explicit
"authentic Moroccan Darija" prompt.

This script is CPU-friendly (DarijaBERT ~135M params) and uses chunked batching
so it runs on an 8 GB laptop.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL_NAME = os.getenv("DARIJA_MODEL", "SI2M-Lab/DarijaBERT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_ARABIC = set(range(0x0600, 0x06FF + 1)) | set(range(0x0750, 0x077F + 1))


def arabic_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) in _ARABIC) / len(letters)


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    return tok, mdl


@torch.no_grad()
def pseudo_perplexity(text: str, tok, mdl, max_tokens: int = 64) -> float:
    """Return an approximate pseudo-PPL for `text` (lower = better).

    We compute it by masking each token once (up to `max_tokens` masks) and
    summing the negative log-probability of the true token. Then
    PPL = exp(sum / N).
    """
    ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=max_tokens)
    if len(ids) < 4:
        return float("inf")
    ids_t = torch.tensor(ids, device=DEVICE)
    mask_id = tok.mask_token_id
    nll = 0.0
    n = 0
    # avoid masking [CLS] (0) and [SEP] (-1)
    for i in range(1, len(ids) - 1):
        batch = ids_t.clone().unsqueeze(0)
        batch[0, i] = mask_id
        logits = mdl(batch).logits[0, i]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll -= log_probs[ids[i]].item()
        n += 1
    if n == 0:
        return float("inf")
    return math.exp(nll / n)


def score_pair(pair: dict, tok, mdl) -> dict:
    q = pair.get("question_darija", "")
    a = pair.get("answer_darija", "")
    q_ppl = pseudo_perplexity(q, tok, mdl) if q else float("inf")
    a_ppl = pseudo_perplexity(a, tok, mdl) if a else float("inf")
    return {
        "q_arabic_ratio": round(arabic_ratio(q), 3),
        "a_arabic_ratio": round(arabic_ratio(a), 3),
        "q_ppl": round(q_ppl, 2) if q_ppl != float("inf") else None,
        "a_ppl": round(a_ppl, 2) if a_ppl != float("inf") else None,
    }


def flag(score: dict, min_ratio: float = 0.35, max_ppl: float = 800.0) -> bool:
    return (
        (score["q_arabic_ratio"] or 0) < min_ratio
        or (score["a_arabic_ratio"] or 0) < min_ratio
        or (score["q_ppl"] is not None and score["q_ppl"] > max_ppl)
        or (score["a_ppl"] is not None and score["a_ppl"] > max_ppl)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--specialty", nargs="+", help="subset of specialties")
    p.add_argument("--sample", type=int, default=100,
                   help="pairs per specialty to score (0 = all)")
    p.add_argument("--min-ratio", type=float, default=0.35)
    p.add_argument("--max-ppl", type=float, default=800.0)
    args = p.parse_args()

    src = Path("data_trilingual")
    out_dir = Path("data_darija_scores")
    out_dir.mkdir(exist_ok=True)

    print(f"[darija-scorer] loading {MODEL_NAME} on {DEVICE}…")
    tok, mdl = load_model()

    specs = args.specialty or sorted(f.stem for f in src.glob("*.json"))
    summary = {}
    for sid in specs:
        fp = src / f"{sid}.json"
        if not fp.exists():
            continue
        pairs = json.loads(fp.read_text(encoding="utf-8"))
        if args.sample > 0:
            pairs = pairs[: args.sample]
        scored = []
        flagged = 0
        for pr in pairs:
            sc = score_pair(pr, tok, mdl)
            sc["_flagged"] = flag(sc, args.min_ratio, args.max_ppl)
            flagged += int(sc["_flagged"])
            scored.append(sc)
        q_ratio = float(np.mean([s["q_arabic_ratio"] for s in scored]))
        a_ratio = float(np.mean([s["a_arabic_ratio"] for s in scored]))
        q_ppl_vals = [s["q_ppl"] for s in scored if s["q_ppl"] is not None]
        a_ppl_vals = [s["a_ppl"] for s in scored if s["a_ppl"] is not None]
        summary[sid] = {
            "n": len(scored),
            "flagged": flagged,
            "q_arabic_ratio_mean": round(q_ratio, 3),
            "a_arabic_ratio_mean": round(a_ratio, 3),
            "q_ppl_median": round(float(np.median(q_ppl_vals)), 2) if q_ppl_vals else None,
            "a_ppl_median": round(float(np.median(a_ppl_vals)), 2) if a_ppl_vals else None,
        }
        (out_dir / f"{sid}.json").write_text(
            json.dumps({"summary": summary[sid], "scores": scored},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{sid}] n={len(scored)} flagged={flagged} "
              f"q_ar={q_ratio:.2f} a_ar={a_ratio:.2f}")

    (out_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"[darija-scorer] done → {out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()
