#!/usr/bin/env python3
"""
Fast LLM-based translation of synthetic EN QA pairs to FR + Moroccan Darija.
Reads from data_synthetic/, writes completed trilingual pairs to data_translated_synth/.

Translation backends:
  - EN → FR : deep_translator (Google, free, instant)
  - EN → Darija : Mistral API batched (5 pairs/call at 20 RPM)

Runs continuously so it can pick up new pairs as generation progresses.
"""

import json
import os
import sys
import time
import logging
import threading
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
        logging.FileHandler("translate_api_fast.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

SYNTHETIC_DIR = Path("data_synthetic")
OUTPUT_DIR = Path("data_translated_synth")
OUTPUT_DIR.mkdir(exist_ok=True)

DARIJA_BATCH = 5   # pairs per API call
CEREBRAS_RPM = 22  # Cerebras limit is 30 RPM; generation uses 5 → leaves 25
FR_DELAY = 0.15    # seconds between Google Translate calls

FILE_LOCKS = {}
FILE_LOCKS_LOCK = threading.Lock()


def get_file_lock(path: str) -> threading.Lock:
    with FILE_LOCKS_LOCK:
        if path not in FILE_LOCKS:
            FILE_LOCKS[path] = threading.Lock()
        return FILE_LOCKS[path]


# ── Google Translate (EN → FR) ────────────────────────────────────────────────

def translate_fr_batch(questions: list[str], answers: list[str]) -> list[tuple[str, str]]:
    """Translate a list of (question, answer) pairs from EN to FR using Google."""
    from deep_translator import GoogleTranslator

    results = []
    for q, a in zip(questions, answers):
        try:
            q_fr = GoogleTranslator(source="en", target="fr").translate(q[:4500]) if q else ""
            time.sleep(FR_DELAY)
            a_fr = GoogleTranslator(source="en", target="fr").translate(a[:4500]) if a else ""
            time.sleep(FR_DELAY)
            results.append((q_fr or "", a_fr or ""))
        except Exception as e:
            log.warning(f"Google Translate error: {e}")
            results.append(("", ""))
    return results


# ── Mistral API (EN → Darija) ─────────────────────────────────────────────────

class CerebrasTranslator:
    """Uses Cerebras (llama3.1-8b) for EN→Darija translation.
    Cerebras has 30 RPM limit; generation uses ~5 → we can use 22 safely."""

    def __init__(self):
        self.api_key = os.getenv("CEREBRAS_API_KEY", "")
        self.url = "https://api.cerebras.ai/v1/chat/completions"
        self.model = "llama3.1-8b"
        self.min_interval = 60.0 / CEREBRAS_RPM
        self.last_call = 0
        self.lock = threading.Lock()

    def _wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()

    def translate_darija_batch(self, pairs: list[dict]) -> list[dict]:
        """Translate a batch of {question, answer} dicts EN→Moroccan Darija."""
        if not self.api_key:
            log.error("No CEREBRAS_API_KEY")
            return pairs

        items_json = json.dumps(
            [{"id": i, "q": p["question"][:500], "a": p["answer"][:700]}
             for i, p in enumerate(pairs)],
            ensure_ascii=False
        )

        prompt = f"""Translate these medical Q&A pairs from English to Moroccan Darija Arabic (دارجة مغربية).
Use natural Moroccan dialect. Medical terms may stay in French/English as used in Morocco.

Input: {items_json}

Output ONLY a JSON array (no markdown, no explanation):
[{{"id": 0, "question_darija": "...", "answer_darija": "..."}}, ...]"""

        for attempt in range(3):
            self._wait()
            try:
                resp = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a medical translator. Output only valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 3000,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=60,
                )

                if resp.status_code == 429:
                    wait_s = 20 * (attempt + 1)
                    log.warning(f"Cerebras 429, waiting {wait_s}s...")
                    time.sleep(wait_s)
                    continue

                if resp.status_code >= 500:
                    log.warning(f"Cerebras {resp.status_code}, attempt {attempt+1}")
                    time.sleep(10)
                    continue

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]

                # Parse JSON array from response
                text = content.strip()
                import re
                # Remove markdown fences
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                text = text.strip()

                start = text.find("[")
                end = text.rfind("]")
                if start == -1:
                    # Try wrapping in array
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1:
                        text = "[" + text[start:end+1] + "]"
                        start, end = 0, len(text) - 1

                arr = []
                if start != -1 and end != -1:
                    try:
                        arr = json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        try:
                            fixed = re.sub(r",\s*]", "]", text[start:end+1])
                            arr = json.loads(fixed)
                        except Exception:
                            pass

                lookup = {item.get("id", i): item for i, item in enumerate(arr)}
                for i, pair in enumerate(pairs):
                    translated = lookup.get(i, {})
                    pair["question_darija"] = translated.get("question_darija", "")
                    pair["answer_darija"] = translated.get("answer_darija", "")
                return pairs

            except Exception as e:
                log.warning(f"Cerebras translation attempt {attempt+1} failed: {e}")
                time.sleep(5)

        # Fallback: empty strings
        for pair in pairs:
            pair.setdefault("question_darija", "")
            pair.setdefault("answer_darija", "")
        return pairs


# ── Per-specialty translation ─────────────────────────────────────────────────

def translate_specialty(specialty_id: str, translator: CerebrasTranslator):
    """Translate all untranslated pairs in one specialty."""
    synth_file = SYNTHETIC_DIR / f"{specialty_id}.json"
    out_file = OUTPUT_DIR / f"{specialty_id}.json"

    if not synth_file.exists():
        return

    # Load synthetic pairs
    with open(synth_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    if not pairs:
        return

    # Load existing translations
    existing = []
    existing_qs = set()
    if out_file.exists():
        lock = get_file_lock(str(out_file))
        with lock:
            with open(out_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing_qs = {p.get("question", "").lower()[:100] for p in existing}

    # Find untranslated pairs
    to_translate = []
    for p in pairs:
        q = p.get("question", "").strip()
        if q.lower()[:100] not in existing_qs and q:
            to_translate.append(p.copy())

    if not to_translate:
        return

    log.info(f"  [{specialty_id}] Translating {len(to_translate)} pairs...")

    # Process in batches
    newly_translated = []
    for i in range(0, len(to_translate), DARIJA_BATCH):
        batch = to_translate[i:i + DARIJA_BATCH]

        # EN → Darija (Mistral)
        batch = translator.translate_darija_batch(batch)

        # EN → FR (Google)
        qs = [p["question"] for p in batch]
        as_ = [p["answer"] for p in batch]
        fr_results = translate_fr_batch(qs, as_)
        for j, (q_fr, a_fr) in enumerate(fr_results):
            batch[j]["question_fr"] = q_fr
            batch[j]["answer_fr"] = a_fr
            batch[j]["question_en"] = batch[j].get("question", "")
            batch[j]["answer_en"] = batch[j].get("answer", "")

        newly_translated.extend(batch)

        if (i // DARIJA_BATCH + 1) % 5 == 0:
            log.info(f"    [{specialty_id}] {len(newly_translated)}/{len(to_translate)} done")
            # Save progress
            lock = get_file_lock(str(out_file))
            with lock:
                all_translated = existing + newly_translated
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(all_translated, f, ensure_ascii=False, indent=2)

    # Final save
    lock = get_file_lock(str(out_file))
    with lock:
        all_translated = existing + newly_translated
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_translated, f, ensure_ascii=False, indent=2)

    log.info(f"  [{specialty_id}] DONE — {len(all_translated)} total translated pairs saved")


def worker(work_queue: queue.Queue, translator: CerebrasTranslator):
    """Worker thread: process specialties from queue."""
    while True:
        try:
            sid = work_queue.get(timeout=10)
        except queue.Empty:
            return
        try:
            translate_specialty(sid, translator)
        except Exception as e:
            log.error(f"Error translating {sid}: {e}", exc_info=True)
        finally:
            work_queue.task_done()


def get_specialty_status():
    """Return dict of specialty_id -> (synthetic_count, translated_count)."""
    status = {}
    for f in sorted(SYNTHETIC_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        sid = f.stem
        with open(f, "r", encoding="utf-8") as fp:
            synth_n = len(json.load(fp))
        out = OUTPUT_DIR / f"{sid}.json"
        trans_n = 0
        if out.exists():
            with open(out, "r", encoding="utf-8") as fp:
                trans_n = len(json.load(fp))
        status[sid] = (synth_n, trans_n)
    return status


def main():
    log.info("=" * 60)
    log.info("FAST API TRANSLATION: EN → FR (Google) + EN → Darija (Cerebras)")
    log.info(f"Input: {SYNTHETIC_DIR}/  →  Output: {OUTPUT_DIR}/")
    log.info(f"Cerebras RPM: {CEREBRAS_RPM} | Darija batch: {DARIJA_BATCH} pairs/call")
    log.info("=" * 60)

    translator = CerebrasTranslator()
    if not translator.api_key:
        log.error("CEREBRAS_API_KEY not set in .env")
        sys.exit(1)

    # Continuous mode: keep checking for new data
    NUM_WORKERS = 3  # 3 threads × 20 RPM = 60 RPM total... but min_interval is shared
    # Actually each worker uses 20 RPM budget sequentially — fine

    run = 0
    while True:
        run += 1
        status = get_specialty_status()

        # Queue specialties that need translation
        work_q = queue.Queue()
        needs_work = []
        for sid, (synth_n, trans_n) in status.items():
            if trans_n < synth_n:
                needs_work.append((sid, synth_n - trans_n))

        if not needs_work:
            log.info(f"[Run {run}] All specialties translated. Waiting 60s for new synthetic data...")
            time.sleep(60)
            continue

        # Sort by most needing translation first
        needs_work.sort(key=lambda x: -x[1])

        total_pending = sum(n for _, n in needs_work)
        log.info(f"[Run {run}] {len(needs_work)} specialties need translation, {total_pending} pairs pending")

        for sid, _ in needs_work:
            work_q.put(sid)

        # Launch workers
        threads = []
        for _ in range(min(NUM_WORKERS, len(needs_work))):
            t = threading.Thread(target=worker, args=(work_q, translator), daemon=True)
            t.start()
            threads.append(t)

        work_q.join()
        for t in threads:
            t.join(timeout=5)

        # Summary
        status = get_specialty_status()
        total_syn = sum(s for s, t in status.values())
        total_tr = sum(t for s, t in status.values())
        log.info(f"[Run {run}] Progress: {total_tr}/{total_syn} pairs translated across {len(status)} specialties")

        # Print per-specialty status
        for sid, (synth_n, trans_n) in sorted(status.items()):
            pct = f"{100*trans_n//synth_n}%" if synth_n else "N/A"
            status_str = "DONE" if trans_n >= synth_n else f"{trans_n}/{synth_n} ({pct})"
            print(f"  {sid}: {status_str}")

        if total_tr >= total_syn:
            log.info("All caught up. Checking again in 120s (waiting for generation)...")
            time.sleep(120)


if __name__ == "__main__":
    main()
