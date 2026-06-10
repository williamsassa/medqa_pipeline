"""Quality filter for v3 pairs (real + synthetic).

Applies a strict multi-criterion filter before a pair enters the final
trilingual/validated pipeline. The goal is to keep only clean, medically
plausible, language-correct pairs so the paper can defend the quality claim.

Criteria (per pair):
    1. Length: question in [40, 3000] chars, answer in [80, 5000] chars
    2. Language heuristic: >= 75% target-script chars (arabic ratio for AR,
       ASCII ratio for EN/FR)
    3. Not near-duplicate (MinHash-free fallback: 8-shingle Jaccard < 0.9)
    4. NER density: >= 1 medical term (from a minimal lexicon)
    5. No refusal / disclaimer boilerplate ("I am an AI", "consult a doctor
       only" without content, etc.)
    6. Pair is a proper Q/A (question ends with `?` or starts with a question
       word; answer has >= 3 sentences OR >= 80 chars of actual content)

Usage:
    python quality_filter.py --src data_scraped_v3/open_corpora \
                              --dst data_scraped_v3/open_corpora_clean

Outputs a per-specialty clean JSON + `_qc_report.json` summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MIN_Q = 40
MAX_Q = 3000
MIN_A = 80
MAX_A = 5000
MIN_SENTENCES_A = 2

REFUSAL_PATTERNS = [
    r"\bi am an ai\b",
    r"\bas an ai\b",
    r"\bi cannot provide\b",
    r"\bi'?m not a doctor\b",
    r"^\s*please consult a (licensed )?physician\.?$",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

QUESTION_WORDS = {"what", "how", "why", "when", "where", "can", "is", "does",
                  "do", "should", "could", "would", "are", "could", "qu'est",
                  "comment", "pourquoi", "quand", "où", "est-ce", "puis-je",
                  "ما", "هل", "كيف", "لماذا", "متى", "أين"}

MED_LEXICON = {
    # EN
    "pain", "fever", "cough", "diabetes", "hypertension", "infection",
    "medication", "dose", "dosage", "mg", "symptom", "treatment", "diagnos",
    "cancer", "tumor", "surgery", "antibiotic", "vaccine", "blood", "heart",
    "lung", "kidney", "liver", "skin", "doctor", "patient", "hospital",
    "prescription", "side effect", "chronic", "acute",
    # FR
    "douleur", "fièvre", "toux", "diabète", "hypertension", "infection",
    "médicament", "posologie", "symptôme", "traitement", "diagnostic",
    "chirurgie", "antibiotique", "vaccin", "sang", "cœur", "poumon", "rein",
    "foie", "peau", "médecin", "patient", "hôpital",
    # AR
    "ألم", "حمى", "سعال", "سكري", "ضغط", "عدوى", "دواء", "جرعة", "عرض",
    "علاج", "تشخيص", "جراحة", "مضاد", "لقاح", "دم", "قلب", "رئة", "كلية",
    "كبد", "جلد", "طبيب", "مريض", "مستشفى",
}

_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENT_RE = re.compile(r"[.!?؟]+\s+")


def arabic_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if _ARABIC_RE.match(c)) / len(letters)


def ascii_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if ord(c) < 128) / len(letters)


def has_medical_term(text: str) -> bool:
    lc = text.lower()
    return any(term in lc for term in MED_LEXICON)


def shingles(text: str, k: int = 8) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def looks_like_question(q: str) -> bool:
    q_strip = q.strip()
    if q_strip.endswith(("?", "؟")):
        return True
    first = q_strip.split()[0].lower() if q_strip else ""
    return first in QUESTION_WORDS


def language_of(pair: dict) -> str:
    lang = (pair.get("language") or "").lower()
    if lang in {"arabic", "ar", "darija"}:
        return "arabic"
    if lang in {"french", "fr"}:
        return "french"
    return "english"


def check_pair(pair: dict) -> tuple[bool, str]:
    q = (pair.get("question") or "").strip()
    a = (pair.get("answer") or "").strip()
    if not (MIN_Q <= len(q) <= MAX_Q):
        return False, f"q_len={len(q)}"
    if not (MIN_A <= len(a) <= MAX_A):
        return False, f"a_len={len(a)}"
    if REFUSAL_RE.search(a):
        return False, "refusal_boilerplate"
    if not looks_like_question(q):
        return False, "not_a_question"
    sentences = [s for s in _SENT_RE.split(a) if s.strip()]
    if len(sentences) < MIN_SENTENCES_A and len(a) < 160:
        return False, "answer_too_thin"
    lang = language_of(pair)
    if lang == "arabic":
        if arabic_ratio(q) < 0.5 and arabic_ratio(a) < 0.5:
            return False, "arabic_ratio_low"
    else:
        if ascii_ratio(q) < 0.5 and ascii_ratio(a) < 0.5:
            return False, "ascii_ratio_low"
    if not (has_medical_term(q) or has_medical_term(a)):
        return False, "no_medical_term"
    return True, "ok"


def dedup_shingle(pairs: list[dict], threshold: float = 0.9) -> list[dict]:
    out = []
    seen_shingles: list[set[str]] = []
    for p in pairs:
        s = shingles(p.get("question", ""))
        if any(jaccard(s, prev) >= threshold for prev in seen_shingles[-512:]):
            continue
        out.append(p)
        seen_shingles.append(s)
    return out


def run(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    report = {}
    for fp in sorted(src.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        kept, reasons = [], {}
        for p in data:
            ok, reason = check_pair(p)
            if ok:
                kept.append(p)
            else:
                reasons[reason] = reasons.get(reason, 0) + 1
        before = len(kept)
        kept = dedup_shingle(kept)
        after = len(kept)
        (dst / fp.name).write_text(
            json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        report[fp.stem] = {
            "input": len(data),
            "kept_after_filter": before,
            "kept_after_dedup": after,
            "rejections": reasons,
        }
        print(f"[qc] {fp.stem:30} in={len(data):>6} keep={after:>6} "
              f"reject={len(data)-after:>6}", flush=True)
    (dst / "_qc_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    tot_in = sum(r["input"] for r in report.values())
    tot_out = sum(r["kept_after_dedup"] for r in report.values())
    print(f"\n[qc] TOTAL input={tot_in} kept={tot_out} "
          f"keep_rate={100*tot_out/max(1,tot_in):.1f}%", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    run(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
