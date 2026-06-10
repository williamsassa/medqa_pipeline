"""
Build BrainHealthAI/BrainMedCoT — 20K medically enriched Chain-of-Thought samples.

Pipeline (deterministic, reproducible):
  1. Load source from local data_with_audio/*.json (or HF when available)
  2. Apply quality filters
  3. Classify question type (rule-based, fast)
  4. Stratified sampling (specialty × language × question_type)
  5. For each sample, in parallel:
       a) extract (drugs, conditions) via Dorosz + INN list + specialty seeds
       b) enrich via PubMed + RxNorm + DailyMed + MedlinePlus
       c) prompt GPT-4o with the enrichment block (6-step + self-critique)
       d) parse JSON, validate against strict rules (real sources, self-critique, gold convergence)
       e) retry up to BMC_LLM_MAX_RETRIES on transient failures
  6. Split 15K train / 2.5K val / 2.5K test
  7. Write parquet + jsonl + metadata.json (with full rejection breakdown for the data card)

Usage:
    # Smoke
    python -m scripts.datasets.brain_med_cot.build --smoke

    # Full
    python -m scripts.datasets.brain_med_cot.build

    # Resume
    python -m scripts.datasets.brain_med_cot.build --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

for env_candidate in [_PROJECT_ROOT / ".env", _PROJECT_ROOT / ".env.local",
                       _PROJECT_ROOT.parent / "brain_repo" / ".env.local"]:
    if env_candidate.is_file():
        load_dotenv(env_candidate, override=False)
        break

from scripts.datasets.brain_med_cot.validators import validate_cot
from scripts.datasets.shared.classifiers import classify_question
from scripts.datasets.shared.dorosz_lookup import get_kg
from scripts.datasets.shared.entity_extract import extract_entities
from scripts.datasets.shared.filters import apply_filters
from scripts.datasets.shared.medical_apis import enrich_qa, format_enrichment_block
from scripts.datasets.shared.seeds import (
    ANTHROPIC_MODEL_DEFAULT,
    BMC_CONCURRENCY,
    BMC_LLM_MAX_RETRIES,
    BMC_LLM_MAX_TOKENS,
    BMC_TARGET_TEST,
    BMC_TARGET_TOTAL,
    BMC_TARGET_TRAIN,
    BMC_TARGET_VAL,
    LANG_DISTRIBUTION,
    LANGUAGE_SAMPLING_SEED,
    LLM_MODEL,
    LLM_PROVIDER_DEFAULT,
    LLM_SEED,
    LLM_TEMPERATURE,
    MASTER_SEED,
    PER_SPECIALTY_CAP,
    QTYPE_DISTRIBUTION,
    SAMPLING_SEED,
)

LLM_PROVIDER = os.getenv("BMC_LLM_PROVIDER", LLM_PROVIDER_DEFAULT).lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("build_brain_med_cot")


# ─── Paths ────────────────────────────────────────────────────────────────

OUTPUT_ROOT = _PROJECT_ROOT / "data_brain_med_cot"
OUTPUT_ROOT.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_ROOT / "checkpoint.jsonl"
REJECTIONS_FILE = OUTPUT_ROOT / "rejections.jsonl"
METADATA_FILE = OUTPUT_ROOT / "metadata.json"
TRAIN_FILE = OUTPUT_ROOT / "train.parquet"
VAL_FILE = OUTPUT_ROOT / "val.parquet"
TEST_FILE = OUTPUT_ROOT / "test.parquet"


# ─── 1. Source loading ────────────────────────────────────────────────────

def load_source(real_only: bool = False) -> list[dict]:
    """Load source QA rows.

    real_only=True → use ONLY the real scraped iCliniq Q&A (source == 'scraped_icliniq',
    from data_with_audio_icliniq/). This is the scientifically stronger gold (real patient
    questions + real doctor answers), used for the second CoT batch.
    """
    if real_only:
        # The iCliniq real QA (source == 'scraped_icliniq') — exactly the 3380 trilingual pairs.
        # We read the local data_with_audio_icliniq/ directly: it's the precise iCliniq scope,
        # instant (no multi-GB audio download), unlike the HF 'real' config which bundles ALL
        # scraped sources (~160K) and the audio files.
        data_dir = _PROJECT_ROOT / "data_with_audio_icliniq"
        all_rows: list[dict] = []
        for jf in sorted(data_dir.glob("*.json")):
            try:
                all_rows.extend(json.load(open(jf, encoding="utf-8")))
            except Exception as ee:
                log.warning("  skipped %s: %s", jf.name, ee)
        all_rows = [r for r in all_rows if r.get("source") == "scraped_icliniq"]
        # Drop rows whose QA fields are malformed (e.g. answer_fr stored as a dict {'texte': ...}).
        _qa = ("question_en", "answer_en", "question_fr", "answer_fr",
               "question_darija", "answer_darija")
        before = len(all_rows)
        all_rows = [r for r in all_rows
                    if all(r.get(k) is None or isinstance(r.get(k), str) for k in _qa)]
        dropped = before - len(all_rows)
        log.info("  real-only (iCliniq): loaded %d scraped_icliniq rows from %s (dropped %d malformed)",
                 len(all_rows), data_dir.name, dropped)
        return all_rows
    try:
        from datasets import load_dataset, Features, Value
        log.info("Loading source from HuggingFace (audio columns will be loaded as strings)…")
        # We don't need the audio bytes for CoT building — just the text fields.
        ds = load_dataset(
            "Williamsanderson/MedQA-Darija-MultiLingual",
            split="train",
            streaming=False,
            verification_mode="no_checks",
        )
        rows = [dict(r) for r in ds]
        log.info("  HF loaded %d rows", len(rows))
        return rows
    except Exception as e:
        log.warning("HF load failed (%s). Falling back to local data_with_audio/*.json", e)
        data_dir = _PROJECT_ROOT / "data_with_audio"
        all_rows: list[dict] = []
        for jf in sorted(data_dir.glob("*.json")):
            try:
                items = json.load(open(jf, encoding="utf-8"))
                all_rows.extend(items)
            except Exception as ee:
                log.warning("  skipped %s: %s", jf.name, ee)
        log.info("  local fallback loaded %d rows from %d files", len(all_rows), len(list(data_dir.glob('*.json'))))
        return all_rows


# ─── 2-3. Filter + classify ───────────────────────────────────────────────

def filter_and_classify(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    log.info("Applying quality filters…")
    survivors, rejections = apply_filters(rows)
    log.info("  %d survived (%d rejected: %s)",
             len(survivors), len(rows) - len(survivors), rejections)

    kg = get_kg()
    log.info("Classifying question types (fast regex Dorosz)…")
    t0 = time.time()
    for r in survivors:
        q = r.get("question_en") or r.get("question_fr") or r.get("question_darija") or ""
        r["_qtype"] = classify_question(q, language="en", kg=kg)
    log.info("  classification done in %.1fs, distribution: %s",
             time.time() - t0, dict(Counter(r["_qtype"] for r in survivors)))
    return survivors, rejections


# ─── 4. Stratified sampling ───────────────────────────────────────────────

def stratified_sample(rows: list[dict], target: int) -> list[dict]:
    rng = np.random.default_rng(SAMPLING_SEED)
    lang_rng = np.random.default_rng(LANGUAGE_SAMPLING_SEED)
    langs = list(LANG_DISTRIBUTION.keys())
    weights = list(LANG_DISTRIBUTION.values())
    for r in rows:
        r["_lang"] = str(lang_rng.choice(langs, p=weights))

    by_spec: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        spec = r.get("specialty_id") or r.get("specialty_en") or "unknown"
        by_spec[spec].append(r)
    log.info("  found %d specialties in survivors", len(by_spec))

    capped: list[dict] = []
    for spec, items in by_spec.items():
        if len(items) <= PER_SPECIALTY_CAP:
            capped.extend(items)
        else:
            idxs = rng.choice(len(items), size=PER_SPECIALTY_CAP, replace=False)
            capped.extend(items[i] for i in idxs)
    log.info("  after per-specialty cap (%d): %d candidates", PER_SPECIALTY_CAP, len(capped))

    # Target: stratify over (specialty, language, qtype) jointly
    keyed: dict[tuple, list[dict]] = defaultdict(list)
    for r in capped:
        key = (r.get("specialty_id", "unknown"), r["_lang"], r["_qtype"])
        keyed[key].append(r)

    if len(capped) <= target:
        log.info("  capped ≤ target, returning all %d", len(capped))
        return capped

    ratio = target / len(capped)
    selected: list[dict] = []
    for key, items in keyed.items():
        take = max(1, int(round(len(items) * ratio)))
        if take >= len(items):
            selected.extend(items)
        else:
            idxs = rng.choice(len(items), size=take, replace=False)
            selected.extend(items[i] for i in idxs)

    if len(selected) > target:
        idxs = rng.choice(len(selected), size=target, replace=False)
        selected = [selected[i] for i in sorted(idxs)]
    log.info("  final stratified sample: %d rows", len(selected))
    return selected


# ─── 5. Per-sample worker ─────────────────────────────────────────────────

def _load_prompts() -> tuple[str, str]:
    base = Path(__file__).parent / "prompts"
    return ((base / "system_v2.md").read_text(encoding="utf-8"),
            (base / "user_v2.md").read_text(encoding="utf-8"))


def _pair_key(row: dict) -> str:
    return str(row.get("pair_id") or row.get("question_en", "")[:200])


async def _llm_call_with_retry(client, system_prompt: str, user_msg: str,
                                logger_prefix: str = "") -> tuple[str | None, str]:
    """Returns (think_text or None, failure_reason).

    Routing logic:
      - If LLM_PROVIDER == "router" (default) → use llm_router with full provider chain
        (OpenRouter free models + Ollama local fallback, with 429 rotation)
      - If LLM_PROVIDER == "anthropic" → direct Anthropic call (paid)
      - If LLM_PROVIDER == "openai"    → direct OpenAI call (paid)
    """
    if LLM_PROVIDER == "router":
        from scripts.datasets.shared.llm_router import generate_think
        # In router mode the `client` arg IS the shared httpx.AsyncClient (passed through from _process_one).
        return await generate_think(
            system_prompt=system_prompt,
            user_msg=user_msg,
            max_tokens=BMC_LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            http_client=client,
        )
    if LLM_PROVIDER == "anthropic":
        return await _llm_call_anthropic(client, system_prompt, user_msg)
    return await _llm_call_openai(client, system_prompt, user_msg)


async def _llm_call_openai(client, system_prompt: str, user_msg: str) -> tuple[str | None, str]:
    from openai import APIError, APITimeoutError, RateLimitError
    last_err = ""
    for attempt in range(BMC_LLM_MAX_RETRIES):
        try:
            resp = await client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                seed=LLM_SEED,
                max_tokens=BMC_LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            content = resp.choices[0].message.content
            think, reason = _extract_think_robust(content)
            if think is None:
                return None, ("llm_returned_empty_think" if reason == "empty_think"
                              else f"json_decode_error: {reason}")
            return think, ""
        except RateLimitError as e:
            last_err = f"rate_limit_attempt{attempt}: {e}"
            await asyncio.sleep((attempt + 1) * 2)
        except APITimeoutError as e:
            last_err = f"timeout_attempt{attempt}: {e}"
            await asyncio.sleep((attempt + 1) * 1)
        except APIError as e:
            last_err = f"api_error_attempt{attempt}: {e}"
            await asyncio.sleep((attempt + 1) * 1)
        except Exception as e:
            last_err = f"unknown_error_attempt{attempt}: {type(e).__name__}: {e}"
            await asyncio.sleep(0.5)
    return None, last_err


def _extract_json_obj(text: str) -> dict | None:
    """Parse a JSON object from an LLM response (robust: fences, trailing commas, truncation)."""
    from scripts.datasets.shared.robust_json import loads_robust
    return loads_robust(text)


def _extract_think_robust(text: str) -> tuple[str | None, str]:
    """Extract the <think> string, tolerating alternate keys + broken JSON. (think, reason)."""
    from scripts.datasets.shared.robust_json import extract_think
    return extract_think(text)


async def _llm_call_anthropic(client, system_prompt: str, user_msg: str) -> tuple[str | None, str]:
    """
    Use Anthropic Claude with prompt caching on the system message.
    Caching keeps the ~1500-token system prompt across calls at 1/10 the read cost.
    """
    import anthropic
    last_err = ""
    for attempt in range(BMC_LLM_MAX_RETRIES):
        try:
            resp = await client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=BMC_LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            # Concatenate all text blocks in the response
            chunks = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            content = "\n".join(chunks).strip()
            think, reason = _extract_think_robust(content)
            if think is None:
                return None, ("llm_returned_empty_think" if reason == "empty_think"
                              else "json_decode_error: no_object_found")
            return think, ""
        except anthropic.RateLimitError as e:
            last_err = f"rate_limit_attempt{attempt}: {e}"
            await asyncio.sleep((attempt + 1) * 3)
        except anthropic.APITimeoutError as e:
            last_err = f"timeout_attempt{attempt}: {e}"
            await asyncio.sleep((attempt + 1) * 1)
        except anthropic.APIStatusError as e:
            last_err = f"api_status_attempt{attempt}: status={e.status_code} {e}"
            if e.status_code in (401, 403):
                return None, last_err  # auth — don't retry
            await asyncio.sleep((attempt + 1) * 2)
        except anthropic.APIError as e:
            last_err = f"api_error_attempt{attempt}: {e}"
            await asyncio.sleep((attempt + 1) * 1)
        except Exception as e:
            last_err = f"unknown_error_attempt{attempt}: {type(e).__name__}: {e}"
            await asyncio.sleep(0.5)
    return None, last_err


async def _process_one(row: dict, kg, system_prompt: str, user_template: str,
                        openai_client, http_client) -> tuple[dict | None, list[str]]:
    """
    Process one row end-to-end: extract entities → enrich via APIs → call LLM → validate.

    Returns (accepted_record_or_None, failed_reasons).
    """
    lang = row["_lang"]
    qfield = {"en": "question_en", "fr": "question_fr", "darija": "question_darija"}[lang]
    afield = {"en": "answer_en",   "fr": "answer_fr",   "darija": "answer_darija"}[lang]
    question = row[qfield]
    answer = row[afield]
    specialty = row.get("specialty_id", "unknown")

    # 5a. Extract entities
    drugs, conditions = extract_entities(question=question, answer=answer, specialty=specialty, kg=kg)

    # 5b. Enrich via APIs (parallel within enrich_qa)
    enr = await enrich_qa(http_client, drugs=drugs, conditions=conditions, language=lang)
    enrichment_block = format_enrichment_block(enr)

    allowed_pmids = {p["pmid"] for p in enr.get("pubmed", [])}
    allowed_setids = {d["setid"] for d in enr.get("dailymed", {}).values() if d.get("setid")}
    allowed_rxcuis = {d["rxcui"] for d in enr.get("rxnorm", {}).values() if d.get("rxcui")}

    # If we have NO sources at all, the sample cannot pass the ≥2-sources gate → skip cheaply
    if not (allowed_pmids or allowed_setids or allowed_rxcuis):
        return None, ["no_enrichment_sources_available"]

    user_msg = user_template.format(
        language=lang, specialty=specialty, question_type=row["_qtype"],
        question=question, answer=answer, enrichment_block=enrichment_block,
    )

    # 5c. LLM call — router mode reuses http_client; OpenAI/Anthropic modes use openai_client
    llm_client = http_client if LLM_PROVIDER == "router" else openai_client
    think, llm_err = await _llm_call_with_retry(llm_client, system_prompt, user_msg)
    if think is None:
        return None, [llm_err or "llm_failed"]

    # 5d. Validation
    passed, failed_rules, diag = validate_cot(
        think, question=question, answer=answer, language=lang,
        allowed_pmids=allowed_pmids, allowed_setids=allowed_setids,
        allowed_rxcuis=allowed_rxcuis, kg=kg,
    )
    if not passed:
        return None, failed_rules

    rec = {
        "_pair_key": _pair_key(row),
        "pair_id": row.get("pair_id"),
        "language": lang,
        "specialty": specialty,
        "question_type": row["_qtype"],
        "question": question,
        "answer": answer,
        "think": think,
        "drugs_extracted": drugs,
        "conditions_extracted": conditions,
        "sources": {
            "pubmed":  [p["pmid"] for p in enr.get("pubmed", [])],
            "dailymed": [d["setid"] for d in enr.get("dailymed", {}).values() if d.get("setid")],
            "rxnorm":   [d["rxcui"] for d in enr.get("rxnorm", {}).values() if d.get("rxcui")],
            "medlineplus": [m["url"] for m in enr.get("medlineplus", {}).values() if m.get("url")],
        },
        "validation_diagnostics": diag,
    }
    return rec, []


# ─── 5 (driver). Async generation loop ────────────────────────────────────

async def generate_dataset(
    selected: list[dict],
    *,
    resume: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """Provider-aware client init (router | openai | anthropic)."""
    if LLM_PROVIDER == "router":
        from scripts.datasets.shared.llm_router import get_provider_chain
        chain = get_provider_chain()
        log.info("LLM provider: router (chain of %d): %s",
                 len(chain), ", ".join(f"{p}:{m}" for p, m in chain))
        # In router mode we reuse the same httpx client for both LLM calls and enrichment calls
        openai_client = None  # router does its own httpx calls
    elif LLM_PROVIDER == "anthropic":
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed. pip install anthropic") from e
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in env / .env.local / .env")
        openai_client = AsyncAnthropic(api_key=api_key)
        log.info("LLM provider: anthropic / model=%s (prompt caching active on system message)", ANTHROPIC_MODEL)
    elif LLM_PROVIDER == "openai":
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in env / .env.local / .env")
        openai_client = AsyncOpenAI(api_key=api_key)
        log.info("LLM provider: openai / model=%s", LLM_MODEL)
    else:
        raise RuntimeError(f"Unknown BMC_LLM_PROVIDER={LLM_PROVIDER!r}, expected router|openai|anthropic")
    http_client = httpx.AsyncClient()
    system_prompt, user_template = _load_prompts()
    kg = get_kg()

    already_done: dict[str, dict] = {}
    if resume and CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_done[rec["_pair_key"]] = rec
                except Exception:
                    pass
        log.info("  resume: %d entries already on disk", len(already_done))

    todo = [r for r in selected if _pair_key(r) not in already_done]
    log.info("Generating %d samples (already_done=%d) with concurrency=%d, retries=%d",
             len(todo), len(already_done), BMC_CONCURRENCY, BMC_LLM_MAX_RETRIES)

    rejections: Counter = Counter()
    accepted: list[dict] = list(already_done.values())
    sem = asyncio.Semaphore(BMC_CONCURRENCY)
    progress_lock = asyncio.Lock()
    processed = 0
    started_at = time.time()

    async def worker(row: dict):
        nonlocal processed
        async with sem:
            try:
                rec, failed = await _process_one(row, kg, system_prompt, user_template,
                                                  openai_client, http_client)
            except Exception as e:
                rec, failed = None, [f"worker_exception:{type(e).__name__}"]

        async with progress_lock:
            processed += 1
            if rec is None:
                for r in failed:
                    rejections[r] += 1
                # Append to rejections log for analysis (without enrichment to keep it small)
                with open(REJECTIONS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "pair_key": _pair_key(row),
                        "language": row.get("_lang"),
                        "specialty": row.get("specialty_id"),
                        "qtype": row.get("_qtype"),
                        "reasons": failed,
                    }, ensure_ascii=False) + "\n")
                return

            accepted.append(rec)
            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if processed % 25 == 0:
                elapsed = time.time() - started_at
                rate = processed / max(elapsed, 1e-6)
                eta_min = (len(todo) - processed) / max(rate, 1e-6) / 60
                log.info("  progress %d/%d (%.1f/s, ETA %.1f min) — accepted=%d rejects=%s",
                         processed, len(todo), rate, eta_min, len(accepted),
                         dict(rejections.most_common(5)))

    await asyncio.gather(*(worker(r) for r in todo))
    await http_client.aclose()
    log.info("Generation done: accepted=%d, rejected breakdown: %s",
             len(accepted), dict(rejections))
    return accepted, dict(rejections)


# ─── 6. Split + save ──────────────────────────────────────────────────────

def split_and_save(accepted: list[dict]) -> dict[str, Any]:
    if not accepted:
        log.error("No accepted samples — nothing to save")
        return {"train": 0, "val": 0, "test": 0}

    df = pd.DataFrame(accepted)
    log.info("Accepted DataFrame: %d rows", len(df))
    df = df.sample(frac=1.0, random_state=MASTER_SEED).reset_index(drop=True)

    n = len(df)
    if n >= BMC_TARGET_TOTAL:
        n_train, n_val, n_test = BMC_TARGET_TRAIN, BMC_TARGET_VAL, BMC_TARGET_TEST
    else:
        # Proportional split for smaller (smoke) runs
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)
        n_train = n - n_test - n_val

    df.iloc[:n_train].to_parquet(TRAIN_FILE, index=False)
    df.iloc[n_train:n_train + n_val].to_parquet(VAL_FILE, index=False)
    df.iloc[n_train + n_val:n_train + n_val + n_test].to_parquet(TEST_FILE, index=False)

    log.info("Saved: train=%d val=%d test=%d -> %s", n_train, n_val, n_test, OUTPUT_ROOT)
    return {"train": n_train, "val": n_val, "test": n_test}


# ─── Main ─────────────────────────────────────────────────────────────────

def _llm_metadata() -> dict:
    """Honest record of how generation actually ran (router chain vs a single paid model)."""
    base = {"provider": LLM_PROVIDER, "temperature": LLM_TEMPERATURE,
            "max_tokens": BMC_LLM_MAX_TOKENS, "max_retries": BMC_LLM_MAX_RETRIES,
            "concurrency": BMC_CONCURRENCY}
    if LLM_PROVIDER == "router":
        try:
            from scripts.datasets.shared.llm_router import get_provider_chain, get_router_stats
            base["provider_chain"] = [f"{p}:{m}" for p, m in get_provider_chain()]
            base["router_stats"] = get_router_stats()
        except Exception:
            pass
    elif LLM_PROVIDER == "anthropic":
        base["model"] = ANTHROPIC_MODEL
    else:
        base["model"] = LLM_MODEL
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=BMC_TARGET_TOTAL,
                    help="Total samples to aim for (default %d)" % BMC_TARGET_TOTAL)
    ap.add_argument("--smoke", action="store_true", help="Smoke test: 12 samples")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--real-only", action="store_true",
                    help="Use only real scraped_icliniq QA (HF 'real' config / local iCliniq)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output subdir name under project root (keeps batches separate)")
    args = ap.parse_args()

    if args.smoke:
        args.target = 12

    # Redirect all outputs to a separate directory if requested (so batches don't clobber).
    if args.out_dir:
        global OUTPUT_ROOT, CHECKPOINT_FILE, REJECTIONS_FILE, METADATA_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE
        OUTPUT_ROOT = _PROJECT_ROOT / args.out_dir
        OUTPUT_ROOT.mkdir(exist_ok=True)
        CHECKPOINT_FILE = OUTPUT_ROOT / "checkpoint.jsonl"
        REJECTIONS_FILE = OUTPUT_ROOT / "rejections.jsonl"
        METADATA_FILE = OUTPUT_ROOT / "metadata.json"
        TRAIN_FILE = OUTPUT_ROOT / "train.parquet"
        VAL_FILE = OUTPUT_ROOT / "val.parquet"
        TEST_FILE = OUTPUT_ROOT / "test.parquet"

    started = time.time()
    log.info("=== BrainMedCoT build start (target=%d, smoke=%s, resume=%s, real_only=%s, out=%s) ===",
             args.target, args.smoke, args.resume, args.real_only, OUTPUT_ROOT.name)

    rows = load_source(real_only=args.real_only)
    survivors, filter_rejections = filter_and_classify(rows)
    selected = stratified_sample(survivors, target=args.target)
    accepted, llm_rejections = asyncio.run(generate_dataset(selected, resume=args.resume))
    split_stats = split_and_save(accepted)

    elapsed = time.time() - started
    metadata = {
        "dataset_name": "BrainHealthAI/BrainMedCoT",
        "version": "v1.0",
        "build_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "target_total": BMC_TARGET_TOTAL,
        "seeds": {"master": MASTER_SEED, "sampling": SAMPLING_SEED,
                  "language_sampling": LANGUAGE_SAMPLING_SEED, "llm": LLM_SEED},
        "llm": _llm_metadata(),
        "enrichment_apis": ["PubMed", "RxNorm", "DailyMed", "MedlinePlus"],
        "source_rows": len(rows),
        "after_filter": len(survivors),
        "filter_rejections": filter_rejections,
        "after_stratification": len(selected),
        "accepted": len(accepted),
        "llm_rejections": llm_rejections,
        "splits": split_stats,
        "target_distributions": {
            "language": LANG_DISTRIBUTION,
            "question_type": QTYPE_DISTRIBUTION,
            "per_specialty_cap": PER_SPECIALTY_CAP,
        },
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Metadata saved to %s", METADATA_FILE)
    log.info("=== Done in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
