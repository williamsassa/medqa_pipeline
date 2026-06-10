#!/usr/bin/env python3
"""
Trilingualize iCliniq rows.

EN → Darija : local model `ychafiqui/english-to-darija-2` (MarianMT) on GPU.
EN → French : Mistral API (json_mode, batched), free tier.

Source: data_merged/{spec}.json — rows where source contains 'icliniq' or 'scraped'.
Output: data_trilingual_icliniq/{spec}.json — trilingual rows ready for validation.

Resume-friendly: skips rows whose pair_id is already present in the output.
Strips the well-known "Hi/Hello, Welcome to icliniq.com." opener before translating.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
import threading
import queue
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("translate_icliniq.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
SRC = PROJECT_DIR / "data_merged"
DST = PROJECT_DIR / "data_trilingual_icliniq"
DST.mkdir(parents=True, exist_ok=True)

# Reuse the same minimal opener-stripper from auto_rebuild
_ICLINIQ_WELCOME = re.compile(
    r"^\s*(?:hi|hello|hey|dear|sir|madam|madame|greetings|bonjour|salut)?\s*[!,.\s]*"
    r"(?:welcome|bienvenue)\s+(?:to|sur|au|à|a)\s+(?:i[\s-]?cliniq|icliniq|i\s+cliniq)"
    r"(?:\.com)?\s*[.!,]?\s*",
    re.IGNORECASE,
)


def clean_icliniq(t: str) -> str:
    if not t:
        return t
    return _ICLINIQ_WELCOME.sub("", t, count=1).strip()


def mkid(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ─── Darija translator (local) ────────────────────────────────────────────────

class DarijaTranslator:
    """Wraps ychafiqui/english-to-darija-2 (MarianMT EN→ar dialect)."""

    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        self.torch = torch
        log.info("loading ychafiqui/english-to-darija-2 ...")
        self.tok = AutoTokenizer.from_pretrained("ychafiqui/english-to-darija-2")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ychafiqui/english-to-darija-2")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        log.info(f"  loaded on {self.device}")

    @staticmethod
    def _split(text: str, max_chars: int = 350) -> list[str]:
        """Split into ~sentence-sized chunks under MarianMT length cap."""
        if len(text) <= max_chars:
            return [text]
        # First split by sentence terminators
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks, buf = [], ""
        for p in parts:
            if not p:
                continue
            if len(buf) + len(p) + 1 <= max_chars:
                buf = (buf + " " + p).strip()
            else:
                if buf:
                    chunks.append(buf)
                if len(p) > max_chars:
                    # Hard slice over-long chunks
                    for i in range(0, len(p), max_chars):
                        chunks.append(p[i:i + max_chars])
                    buf = ""
                else:
                    buf = p
        if buf:
            chunks.append(buf)
        return chunks

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        # Expand each text into chunks, remember mapping back
        all_chunks = []
        owners = []
        for i, t in enumerate(texts):
            chunks = self._split(t or "")
            for c in chunks:
                all_chunks.append(c)
                owners.append(i)
        if not all_chunks:
            return ["" for _ in texts]
        outs_per_chunk = []
        with self.torch.no_grad():
            for j in range(0, len(all_chunks), 8):
                batch = all_chunks[j:j + 8]
                inputs = self.tok(batch, return_tensors="pt", padding=True,
                                  truncation=True, max_length=512).to(self.device)
                gen = self.model.generate(**inputs, max_length=512, num_beams=4,
                                          length_penalty=0.9, no_repeat_ngram_size=3)
                outs_per_chunk.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        # Reassemble
        merged = ["" for _ in texts]
        for k, owner in enumerate(owners):
            merged[owner] = (merged[owner] + " " + outs_per_chunk[k]).strip()
        return merged


# ─── French translator (Mistral) ──────────────────────────────────────────────

MISTRAL_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"

_rate_lock = threading.Lock()
_last_call = 0.0
_min_interval = 60.0 / 48  # ~48 RPM


def _mistral_request(prompt: str, retries: int = 5) -> str | None:
    global _last_call
    if not MISTRAL_KEY:
        return None
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "max_tokens": 4000,
    }
    headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}
    for attempt in range(retries):
        with _rate_lock:
            wait = _min_interval - (time.time() - _last_call)
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        try:
            r = requests.post(MISTRAL_URL, json=payload, headers=headers, timeout=120)
            if r.status_code == 429:
                ra = int(r.headers.get("Retry-After", "30"))
                log.warning(f"  Mistral 429 → sleep {ra}s")
                time.sleep(ra)
                continue
            if not r.ok:
                log.warning(f"  Mistral {r.status_code}: {r.text[:200]}")
                time.sleep(5)
                continue
            return r.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            log.warning(f"  Mistral err: {e}")
            time.sleep(10)
    return None


_FR_PROMPT = (
    "You are a medical translator. Translate the following English medical "
    "question and answer into natural professional French. Preserve medical "
    "terminology accurately. Do NOT paraphrase or summarize — translate every "
    "sentence faithfully. Return ONLY a JSON object: "
    '{{"q_fr": "<French translation of the question>", '
    '"a_fr": "<French translation of the answer>"}}.\n\n'
    "QUESTION (EN): {q}\n\n"
    "ANSWER (EN): {a}"
)


def translate_french_one(q_en: str, a_en: str) -> tuple[str, str]:
    """Single-pair translation. Reliable (no batch desync)."""
    if not q_en and not a_en:
        return ("", "")
    prompt = _FR_PROMPT.format(q=q_en[:3500], a=a_en[:5000])
    raw = _mistral_request(prompt)
    if not raw:
        return ("", "")
    try:
        obj = json.loads(raw)
        return (obj.get("q_fr", "") or "", obj.get("a_fr", "") or "")
    except json.JSONDecodeError as e:
        log.warning(f"  Mistral JSON parse fail (single): {e}")
        return ("", "")


# ─── Main ─────────────────────────────────────────────────────────────────────

def gather_icliniq() -> list[tuple[str, dict]]:
    rows = []
    for f in sorted(SRC.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "pairs" in data:
            data = data["pairs"]
        for r in data:
            s = str(r.get("source", "")).lower()
            if "icliniq" in s or "scraped" in s:
                rows.append((f.stem, r))
    return rows


def load_existing(spec: str) -> dict[str, dict]:
    out_file = DST / f"{spec}.json"
    if not out_file.exists():
        return {}
    try:
        existing = json.loads(out_file.read_text(encoding="utf-8"))
        if isinstance(existing, list):
            return {r["pair_id"]: r for r in existing if r.get("pair_id")}
    except Exception:
        pass
    return {}


def save_spec(spec: str, rows: dict[str, dict]):
    out_file = DST / f"{spec}.json"
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(rows.values()), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out_file)


def main():
    raw_rows = gather_icliniq()
    log.info(f"iCliniq rows total: {len(raw_rows)}")

    # Group by spec
    by_spec: dict[str, list[dict]] = {}
    for spec, r in raw_rows:
        by_spec.setdefault(spec, []).append(r)

    log.info(f"specs touched: {len(by_spec)}")

    dt = DarijaTranslator()
    BATCH_DARIJA = 8

    total_done = 0
    for spec, rows in sorted(by_spec.items()):
        existing = load_existing(spec)
        # Build to-do list
        todo = []
        prepared = []
        for r in rows:
            q_raw = r.get("question") or r.get("question_en") or ""
            a_raw = r.get("answer") or r.get("answer_en") or ""
            q_en = clean_icliniq(q_raw)
            a_en = clean_icliniq(a_raw)
            if not q_en and not a_en:
                continue
            pid = r.get("pair_id") or mkid(q_en + "|" + a_en + "|" + spec)
            if pid in existing and existing[pid].get("question_darija") and existing[pid].get("question_fr"):
                continue
            prepared.append({
                "pair_id": pid,
                "specialty_id": spec,
                "source": r.get("source") or "scraped_icliniq",
                "question_en": q_en,
                "answer_en": a_en,
            })
        if not prepared:
            log.info(f"  [{spec}] all {len(rows)} already done")
            continue
        log.info(f"  [{spec}] translating {len(prepared)} rows (already had {len(existing)})")

        # Translate French one pair at a time (reliable, no batch desync)
        # Only redo French for pairs missing it
        existing_fr = {existing[p["pair_id"]].get("question_fr"): existing[p["pair_id"]]
                       for p in prepared if p["pair_id"] in existing}
        for j, p in enumerate(prepared):
            prev = existing.get(p["pair_id"])
            if prev and prev.get("question_fr") and prev.get("answer_fr"):
                p["question_fr"] = prev["question_fr"]
                p["answer_fr"] = prev["answer_fr"]
                continue
            qfr, afr = translate_french_one(p["question_en"], p["answer_en"])
            p["question_fr"] = qfr
            p["answer_fr"] = afr
            if (j + 1) % 20 == 0:
                log.info(f"    fr: {j+1}/{len(prepared)}")
                # Persist progress every 20 pairs in case of crash
                snapshot = {pp["pair_id"]: pp for pp in prepared if pp.get("question_fr")}
                snapshot.update({k: v for k, v in existing.items() if k not in snapshot})
                save_spec(spec, snapshot)

        # Translate Darija (only for pairs missing it)
        need_darija = [p for p in prepared
                       if not (existing.get(p["pair_id"]) and
                               existing[p["pair_id"]].get("question_darija"))]
        if need_darija:
            all_q = [p["question_en"] for p in need_darija]
            all_a = [p["answer_en"] for p in need_darija]
            log.info(f"    darija: {len(all_q)} pairs to translate")
            q_da, a_da = [], []
            for i in range(0, len(all_q), BATCH_DARIJA):
                q_da.extend(dt.translate_batch(all_q[i:i + BATCH_DARIJA]))
            for i in range(0, len(all_a), BATCH_DARIJA):
                a_da.extend(dt.translate_batch(all_a[i:i + BATCH_DARIJA]))
            for p, q, a in zip(need_darija, q_da, a_da):
                p["question_darija"] = q
                p["answer_darija"] = a
        # Carry forward existing darija for pairs already done
        for p in prepared:
            prev = existing.get(p["pair_id"])
            if prev and prev.get("question_darija") and not p.get("question_darija"):
                p["question_darija"] = prev["question_darija"]
                p["answer_darija"] = prev["answer_darija"]
            existing[p["pair_id"]] = p

        save_spec(spec, existing)
        total_done += len(prepared)
        log.info(f"  [{spec}] saved {len(existing)} total to disk; cumulative done={total_done}")

    log.info(f"DONE. Total translated this run: {total_done}")


if __name__ == "__main__":
    main()
