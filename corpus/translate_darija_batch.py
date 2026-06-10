#!/usr/bin/env python3
"""
Post-generation batch Darija translation.
Runs AFTER generate_multi_api.py completes.

Reads data_synthetic/*.json (English pairs) and data_backfilled/*.json (FR+Arabic backfilled).
Translates EN → Moroccan Darija using Mistral API (json_mode, batched 10 pairs per call).
Also does EN → FR via Google if FR is missing.

Output: data_trilingual/{specialty}.json — complete trilingual pairs ready for pipeline.

Usage:
    python translate_darija_batch.py              # translate all
    python translate_darija_batch.py --check      # just check what needs translation
"""

import json
import os
import re
import sys
import time
import logging
import threading
import argparse
import queue
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("translate_darija.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

SYNTHETIC_DIR = Path("data_synthetic")
BACKFILL_DIR = Path("data_backfilled")
TRILINGUAL_DIR = Path("data_trilingual")
TRILINGUAL_DIR.mkdir(exist_ok=True)

DARIJA_BATCH = 5    # 5 pairs per call — ~30-40s latency vs 60-90s for 10 pairs
MISTRAL_RPM = 48    # near max now that generation is complete

_rate_lock = threading.Lock()
_last_call = 0


def api_wait():
    global _last_call
    min_interval = 60.0 / MISTRAL_RPM
    while True:
        with _rate_lock:
            elapsed = time.time() - _last_call
            if elapsed >= min_interval:
                _last_call = time.time()
                return
        time.sleep(0.05)


def _parse_translations(content: str) -> list:
    """Robustly parse the translations array from Mistral JSON response."""
    content = content.strip()
    # Remove markdown fences
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    content = content.strip()

    # Try direct parse first
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            arr = obj.get("translations", [])
            if not arr:
                for v in obj.values():
                    if isinstance(v, list):
                        arr = v
                        break
            return arr if arr else []
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Try to find and extract the array
    start = content.find('[')
    end = content.rfind(']')
    if start != -1 and end != -1 and end > start:
        chunk = content[start:end+1]
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            # Fix common issues: trailing commas, truncated last item
            chunk = re.sub(r',\s*([}\]])', r'\1', chunk)  # trailing commas
            # If truncated, remove last incomplete object
            last_complete = chunk.rfind('},')
            if last_complete != -1:
                chunk = chunk[:last_complete+1] + ']'
            try:
                return json.loads(chunk)
            except Exception:
                pass

    return []


def translate_batch_fr_and_darija(pairs: list[dict], api_key: str) -> list[dict]:
    """Translate a batch of EN pairs to BOTH French AND Moroccan Darija via Groq (llama-3.3-70b)."""
    items_json = json.dumps(
        [{"id": i, "q": p["question"][:400], "a": p["answer"][:500]}
         for i, p in enumerate(pairs)],
        ensure_ascii=False
    )

    prompt = f"""Translate these medical Q&A pairs from English to TWO languages:
1. French (medical French)
2. Moroccan Darija Arabic — MUST use ARABIC SCRIPT (حروف عربية), natural Moroccan dialect (دارجة مغربية). NOT Arabizi/Latin letters.

Input: {items_json}

Return ONLY valid JSON (no markdown):
{{"translations": [{{"id": 0, "question_fr": "...", "answer_fr": "...", "question_darija": "...", "answer_darija": "..."}}, ...]}}"""

    for attempt in range(3):
        api_wait()
        try:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                json={
                    "model": "mistral-small-latest",
                    "messages": [
                        {"role": "system", "content": "You are a bilingual medical translator (French + Moroccan Darija). Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 8192,
                    "response_format": {"type": "json_object"},
                },
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=120,
            )

            if resp.status_code == 429:
                wait_s = 30 * (attempt + 1)
                log.warning(f"Mistral 429, waiting {wait_s}s...")
                time.sleep(wait_s)
                continue

            if resp.status_code >= 500:
                log.warning(f"Mistral {resp.status_code}, attempt {attempt+1}")
                time.sleep(10)
                continue

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Robust JSON parsing with repair
            arr = _parse_translations(content)

            if not arr:
                log.warning(f"Empty translation response, attempt {attempt+1}")
                time.sleep(3)
                continue

            lookup = {item.get("id", i): item for i, item in enumerate(arr)
                      if isinstance(item, dict)}
            for i, pair in enumerate(pairs):
                t = lookup.get(i, {})
                pair["question_fr"] = t.get("question_fr", pair.get("question_fr", ""))
                pair["answer_fr"] = t.get("answer_fr", pair.get("answer_fr", ""))
                pair["question_darija"] = t.get("question_darija", "")
                pair["answer_darija"] = t.get("answer_darija", "")
            return pairs

        except Exception as e:
            log.warning(f"Translation attempt {attempt+1} failed: {e}")
            time.sleep(5)

    for pair in pairs:
        pair.setdefault("question_fr", "")
        pair.setdefault("answer_fr", "")
        pair.setdefault("question_darija", "")
        pair.setdefault("answer_darija", "")
    return pairs


def load_specialty_pairs(sid: str) -> list[dict]:
    """Load all pairs for a specialty from synthetic + backfilled sources."""
    pairs = []
    seen = set()

    # 1. Load backfilled (FR+Arabic already done)
    bf = BACKFILL_DIR / f"{sid}.json"
    if bf.exists():
        with open(bf, "r", encoding="utf-8") as f:
            for p in json.load(f):
                q = p.get("question", "")
                if q[:100] not in seen:
                    seen.add(q[:100])
                    pairs.append(p)

    # 2. Load synthetic (may have new pairs not in backfilled)
    sf = SYNTHETIC_DIR / f"{sid}.json"
    if sf.exists():
        with open(sf, "r", encoding="utf-8") as f:
            for p in json.load(f):
                q = p.get("question", "")
                if q[:100] not in seen:
                    seen.add(q[:100])
                    pairs.append(p)

    return pairs


def process_specialty(sid: str, api_key: str) -> int:
    """Translate all pairs for one specialty to trilingual. Return count."""
    out_file = TRILINGUAL_DIR / f"{sid}.json"

    # Load existing output (resume support)
    done_qs = set()
    output_pairs = []
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            output_pairs = json.load(f)
        done_qs = {p.get("question", "")[:100] for p in output_pairs}

    all_pairs = load_specialty_pairs(sid)
    to_translate = [p for p in all_pairs if p.get("question", "")[:100] not in done_qs]

    if not to_translate:
        return len(output_pairs)

    log.info(f"  [{sid}] {len(to_translate)} pairs to translate ({len(output_pairs)} already done)")

    for i in range(0, len(to_translate), DARIJA_BATCH):
        batch = [p.copy() for p in to_translate[i:i + DARIJA_BATCH]]

        # Always set EN fields
        for p in batch:
            p.setdefault("question_en", p.get("question", ""))
            p.setdefault("answer_en", p.get("answer", ""))

        # Translate to FR + Darija in ONE call (both languages together)
        # Only skip if BOTH are already done
        needs_translation = [p for p in batch if not p.get("question_darija")]
        already_done = [p for p in batch if p.get("question_darija")]

        if needs_translation:
            needs_translation = translate_batch_fr_and_darija(needs_translation, api_key)

        batch = already_done + needs_translation
        output_pairs.extend(batch)

        if (i // DARIJA_BATCH + 1) % 5 == 0:
            log.info(f"    [{sid}] {len(output_pairs)}/{len(all_pairs)} done")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(output_pairs, f, ensure_ascii=False, indent=2)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_pairs, f, ensure_ascii=False, indent=2)

    log.info(f"  [{sid}] DONE — {len(output_pairs)} trilingual pairs saved")
    return len(output_pairs)


def worker(q: queue.Queue, api_key: str):
    while True:
        try:
            sid = q.get(timeout=5)
        except queue.Empty:
            return
        try:
            process_specialty(sid, api_key)
        except Exception as e:
            log.error(f"Error processing {sid}: {e}", exc_info=True)
        finally:
            q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Just show status")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        log.error("MISTRAL_API_KEY not set")
        sys.exit(1)

    # Find all specialties
    specialties = sorted(set(
        f.stem for f in list(SYNTHETIC_DIR.glob("*.json")) + list(BACKFILL_DIR.glob("*.json"))
        if not f.name.startswith("_")
    ))

    log.info("=" * 60)
    log.info(f"DARIJA BATCH TRANSLATION (Mistral) — {len(specialties)} specialties")
    log.info(f"Mistral RPM: {MISTRAL_RPM} | Batch: {DARIJA_BATCH} pairs/call")
    log.info(f"Workers: {args.workers}")
    log.info("=" * 60)

    if args.check:
        total_synth = total_tri = 0
        for sid in specialties:
            pairs = load_specialty_pairs(sid)
            out = TRILINGUAL_DIR / f"{sid}.json"
            tri_n = len(json.load(open(out, "r", encoding="utf-8"))) if out.exists() else 0
            status = "DONE" if tri_n >= len(pairs) else f"{tri_n}/{len(pairs)}"
            print(f"  {sid}: synth={len(pairs)}, trilingual={status}")
            total_synth += len(pairs)
            total_tri += tri_n
        print(f"\nTotal: {total_tri}/{total_synth} translated")
        return

    work_q = queue.Queue()
    for sid in specialties:
        work_q.put(sid)

    threads = [
        threading.Thread(target=worker, args=(work_q, api_key), daemon=True)
        for _ in range(args.workers)
    ]
    for t in threads:
        t.start()

    work_q.join()
    log.info("All specialties translated!")


if __name__ == "__main__":
    main()
