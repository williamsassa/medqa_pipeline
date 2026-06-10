"""Multi-provider parallel EN -> FR + Darija translator for open_corpora_clean.

Reads real EN pairs from data_scraped_v3/open_corpora_clean/*.json and
translates each into French + Moroccan Darija (Arabic script) via 7 LLM
providers running concurrently with per-provider rate limits.

Output: data_trilingual_openc/{specialty}.json (schema-v3 compatible)

Providers used (free or cheap):
    Mistral, Groq, Cerebras, Together, OpenRouter, Gemini, Anthropic
Anthropic is optional and only enabled if ANTHROPIC_API_KEY is set.

Robustness: each worker loops on its own queue share; failures log and skip.
Resume-friendly: skips pairs whose hash is already in the output file.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SRC = Path("data_scraped_v3/open_corpora_clean")
DST = Path("data_trilingual_openc")
DST.mkdir(parents=True, exist_ok=True)

BATCH = 3
REQUEST_TIMEOUT = 90

PROMPT = (
    "You are a medical translator. For each English medical Q&A pair below, "
    "produce a natural French translation and a Moroccan Darija translation "
    "written in ARABIC SCRIPT ONLY (no Arabizi, no MSA, no Latin). "
    "Preserve medical accuracy. Return ONLY a JSON array with this shape:\n"
    '[{"fr_q":"...","fr_a":"...","da_q":"...","da_a":"..."}]\n'
    "One object per input pair, in the same order."
)


def key(name: str) -> str:
    return os.getenv(name, "").strip()


# provider specs: openai-compatible chat completions
OAI_PROVIDERS = [
    {
        "name": "mistral",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "key": key("MISTRAL_API_KEY"),
        "rpm": 40,
        "json_mode": True,
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "key": key("GROQ_API_KEY"),
        "rpm": 25,
        "json_mode": True,
    },
    {
        "name": "cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-8b",
        "key": key("CEREBRAS_API_KEY"),
        "rpm": 25,
        "json_mode": False,
    },
    {
        "name": "together",
        "url": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "key": key("TOGETHER_API_KEY"),
        "rpm": 50,
        "json_mode": False,
    },
    {
        "name": "openrouter",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key": key("OPENROUTER_API_KEY"),
        "rpm": 18,
        "json_mode": False,
    },
]


def call_openai_compat(prov: dict, pairs: list[dict]) -> Optional[list[dict]]:
    payload = {
        "model": prov["model"],
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(
                [{"q": p["question"], "a": p["answer"]} for p in pairs],
                ensure_ascii=False,
            )},
        ],
        "temperature": 0.3,
    }
    if prov.get("json_mode"):
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {prov['key']}",
        "Content-Type": "application/json",
    }
    r = requests.post(prov["url"], json=payload, headers=headers,
                      timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{prov['name']} HTTP {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"]["content"]
    return parse_json_array(content)


def call_gemini(prov: dict, pairs: list[dict]) -> Optional[list[dict]]:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{prov['model']}:generateContent?key={prov['key']}")
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": PROMPT + "\n\nINPUT:\n" + json.dumps(
                [{"q": p["question"], "a": p["answer"]} for p in pairs],
                ensure_ascii=False,
            )}],
        }],
        "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"},
    }
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"gemini HTTP {r.status_code}: {r.text[:200]}")
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    return parse_json_array(text)


def call_anthropic(prov: dict, pairs: list[dict]) -> Optional[list[dict]]:
    payload = {
        "model": prov["model"],
        "max_tokens": 4096,
        "system": PROMPT,
        "messages": [{"role": "user", "content": json.dumps(
            [{"q": p["question"], "a": p["answer"]} for p in pairs],
            ensure_ascii=False,
        )}],
    }
    headers = {
        "x-api-key": prov["key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"anthropic HTTP {r.status_code}: {r.text[:200]}")
    text = r.json()["content"][0]["text"]
    return parse_json_array(text)


_JSON_ARR_RE = re.compile(r"\[[\s\S]*\]")


def parse_json_array(text: str) -> Optional[list[dict]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            for k in ("data", "translations", "results", "items"):
                if k in val and isinstance(val[k], list):
                    return val[k]
            return [val]
        if isinstance(val, list):
            return val
    except Exception:
        m = _JSON_ARR_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def valid_translation(t: dict) -> bool:
    need = ("fr_q", "fr_a", "da_q", "da_a")
    if not all(isinstance(t.get(k), str) and t[k].strip() for k in need):
        return False
    da = t["da_q"] + t["da_a"]
    ar_chars = sum(1 for c in da if 0x0600 <= ord(c) <= 0x06FF or
                   0x0750 <= ord(c) <= 0x077F)
    letters = sum(1 for c in da if c.isalpha())
    return letters > 0 and ar_chars / letters >= 0.5


class Provider:
    def __init__(self, spec: dict, caller):
        self.spec = spec
        self.caller = caller
        self.min_interval = 60.0 / max(1, spec["rpm"])
        self._next_ok = 0.0
        self._lock = threading.Lock()
        self.ok = 0
        self.fail = 0

    def wait(self):
        with self._lock:
            now = time.time()
            if now < self._next_ok:
                time.sleep(self._next_ok - now)
            self._next_ok = time.time() + self.min_interval

    def call(self, pairs):
        self.wait()
        try:
            res = self.caller(self.spec, pairs)
            if res and len(res) >= len(pairs):
                self.ok += 1
                return res[: len(pairs)]
            self.fail += 1
            return None
        except Exception as e:
            self.fail += 1
            msg = str(e)[:140]
            print(f"  [{self.spec['name']}] err: {msg}", flush=True)
            return None


def build_providers() -> list[Provider]:
    provs: list[Provider] = []
    for spec in OAI_PROVIDERS:
        if spec["key"]:
            provs.append(Provider(spec, call_openai_compat))
    if key("GOOGLE_API_KEY"):
        provs.append(Provider(
            {"name": "gemini", "model": "gemini-2.5-flash",
             "key": key("GOOGLE_API_KEY"), "rpm": 14},
            call_gemini,
        ))
    if key("ANTHROPIC_API_KEY"):
        provs.append(Provider(
            {"name": "anthropic", "model": "claude-haiku-4-5-20251001",
             "key": key("ANTHROPIC_API_KEY"), "rpm": 40},
            call_anthropic,
        ))
    return provs


def translate_pair_block(pairs: list[dict], providers: list[Provider]) -> Optional[list[dict]]:
    order = list(range(len(providers)))
    random.shuffle(order)
    for idx in order:
        res = providers[idx].call(pairs)
        if not res:
            continue
        filled = []
        for t in res:
            if isinstance(t, dict) and valid_translation(t):
                filled.append(t)
            else:
                filled.append(None)
        if all(filled):
            return filled
    return None


def hash_q(q: str) -> str:
    return hashlib.md5(q.encode("utf-8")).hexdigest()


def run_specialty(fp: Path, providers: list[Provider], limit: int,
                  workers: int) -> dict:
    sid = fp.stem
    src_pairs = json.loads(fp.read_text(encoding="utf-8"))
    out_fp = DST / fp.name
    done: dict[str, dict] = {}
    if out_fp.exists():
        try:
            for p in json.loads(out_fp.read_text(encoding="utf-8")):
                done[hash_q(p.get("question") or p.get("question_en") or "")] = p
        except Exception:
            done = {}
    todo = [p for p in src_pairs if hash_q(p["question"]) not in done]
    if limit > 0:
        todo = todo[:limit]
    if not todo:
        print(f"[{sid}] nothing to do (already {len(done)})", flush=True)
        return {"sid": sid, "done": len(done), "added": 0, "failed": 0}

    print(f"[{sid}] translating {len(todo)} pairs ({len(done)} already done)",
          flush=True)

    q: "queue.Queue[Optional[list[dict]]]" = queue.Queue()
    for i in range(0, len(todo), BATCH):
        q.put(todo[i:i + BATCH])
    for _ in range(workers):
        q.put(None)

    out_lock = threading.Lock()
    stats = {"added": 0, "failed": 0}

    def worker():
        while True:
            blk = q.get()
            if blk is None:
                q.task_done()
                return
            res = translate_pair_block(blk, providers)
            if res:
                enriched = []
                for src, t in zip(blk, res):
                    rec = dict(src)
                    rec["question_en"] = src["question"]
                    rec["answer_en"] = src["answer"]
                    rec["question_fr"] = t["fr_q"]
                    rec["answer_fr"] = t["fr_a"]
                    rec["question_darija"] = t["da_q"]
                    rec["answer_darija"] = t["da_a"]
                    enriched.append(rec)
                with out_lock:
                    for rec in enriched:
                        done[hash_q(rec["question_en"])] = rec
                        stats["added"] += 1
                    out_fp.write_text(
                        json.dumps(list(done.values()),
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            else:
                with out_lock:
                    stats["failed"] += len(blk)
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    q.join()

    print(f"[{sid}] +{stats['added']} translated, {stats['failed']} failed, "
          f"total={len(done)}", flush=True)
    return {"sid": sid, "done": len(done), **stats}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--specialty", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="max pairs per specialty (0 = all)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    provs = build_providers()
    print(f"[multi] providers active: {[p.spec['name'] for p in provs]}",
          flush=True)
    if not provs:
        print("no providers configured (check .env)")
        sys.exit(2)

    files = sorted(SRC.glob("*.json"))
    if args.specialty:
        files = [f for f in files if f.stem in args.specialty]

    summary = []
    for fp in files:
        try:
            r = run_specialty(fp, provs, args.limit, args.workers)
            summary.append(r)
        except KeyboardInterrupt:
            print("[multi] interrupted", flush=True)
            break
        except Exception as e:
            print(f"[multi] {fp.stem} fatal: {e}", flush=True)
    (DST / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    for pr in provs:
        print(f"[multi] {pr.spec['name']:12} ok={pr.ok} fail={pr.fail}", flush=True)


if __name__ == "__main__":
    main()
