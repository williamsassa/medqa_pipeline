#!/usr/bin/env python3
"""
Free translation pipeline for MedQA dataset.
Translates EN → FR using deep_translator (Google) and EN → Darija using NLLB-200.
Falls back to deep_translator MyMemory if Google rate-limits.
"""

import json
import os
import sys
import time
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Translation backends ──────────────────────────────────────────────────────

def translate_en_to_fr_google(text: str) -> str:
    """Translate EN→FR using deep_translator (Google, free)."""
    from deep_translator import GoogleTranslator
    if not text or not text.strip():
        return ""
    text = text.strip()
    # Google Translate has a 5000 char limit per request
    if len(text) > 4500:
        chunks = _split_text(text, 4500)
        results = []
        for c in chunks:
            r = GoogleTranslator(source="en", target="fr").translate(c)
            results.append(r if r else "")
        return "\n\n".join(results)
    result = GoogleTranslator(source="en", target="fr").translate(text)
    return result if result else ""


def translate_en_to_fr_mymemory(text: str) -> str:
    """Fallback: EN→FR using MyMemoryTranslator (free, 1000 chars/request)."""
    from deep_translator import MyMemoryTranslator
    if len(text) > 900:
        chunks = _split_text(text, 900)
        return "\n\n".join(MyMemoryTranslator(source="en-GB", target="fr-FR").translate(c) for c in chunks)
    return MyMemoryTranslator(source="en-GB", target="fr-FR").translate(text)


def _get_device():
    """Get the best available device."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def translate_en_to_darija_nllb(text: str, model=None, tokenizer=None) -> str:
    """Translate EN→Moroccan Arabic (Darija) using Meta NLLB-200."""
    if model is None or tokenizer is None:
        raise ValueError("NLLB model and tokenizer must be provided")
    if not text or not text.strip():
        return ""

    tokenizer.src_lang = "eng_Latn"
    device = model.device

    if len(text) > 1000:
        chunks = _split_text(text, 1000)
        translated_chunks = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("ary_Arab")
            outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=512)
            translated_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return "\n\n".join(translated_chunks)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("ary_Arab")
    outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_fr_to_darija_nllb(text: str, model=None, tokenizer=None) -> str:
    """Translate FR→Moroccan Arabic (Darija) using Meta NLLB-200."""
    if model is None or tokenizer is None:
        raise ValueError("NLLB model and tokenizer must be provided")
    if not text or not text.strip():
        return ""

    tokenizer.src_lang = "fra_Latn"
    device = model.device

    if len(text) > 1000:
        chunks = _split_text(text, 1000)
        translated_chunks = []
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("ary_Arab")
            outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=512)
            translated_chunks.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
        return "\n\n".join(translated_chunks)

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("ary_Arab")
    outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _split_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_len:
            if current:
                chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text[:max_len]]


# ── Main pipeline ─────────────────────────────────────────────────────────────

class TranslationPipeline:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.nllb_model = None
        self.nllb_tokenizer = None
        self.fr_fail_count = 0

    def _load_nllb(self):
        """Load NLLB-200 distilled model (600M params, works on CPU)."""
        if self.nllb_model is not None:
            return
        log.info("Loading NLLB-200 distilled model (this may take a few minutes on first run)...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = "facebook/nllb-200-distilled-600M"
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        device = _get_device()
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        log.info(f"NLLB-200 model loaded successfully on {device}")

    def translate_pair(self, qa: dict) -> dict:
        """Translate a single QA pair from EN to FR and Darija."""
        question_en = qa.get("question", "") or ""
        answer_en = qa.get("answer", "") or ""

        # Clean text - remove problematic chars
        question_en = question_en.strip()
        answer_en = answer_en.strip()

        if not question_en or not answer_en:
            qa["question_en"] = question_en
            qa["answer_en"] = answer_en
            qa["question_fr"] = ""
            qa["answer_fr"] = ""
            qa["question_darija"] = ""
            qa["answer_darija"] = ""
            return qa

        # EN → FR
        try:
            question_fr = translate_en_to_fr_google(question_en)
            answer_fr = translate_en_to_fr_google(answer_en)
            self.fr_fail_count = 0
        except Exception as e:
            log.warning(f"Google Translate failed, using MyMemory: {e}")
            self.fr_fail_count += 1
            if self.fr_fail_count > 5:
                time.sleep(60)  # Back off if too many failures
                self.fr_fail_count = 0
            try:
                question_fr = translate_en_to_fr_mymemory(question_en)
                answer_fr = translate_en_to_fr_mymemory(answer_en)
            except Exception as e2:
                log.error(f"MyMemory also failed: {e2}")
                question_fr = ""
                answer_fr = ""

        # EN → Darija (Moroccan Arabic) via NLLB
        try:
            self._load_nllb()
            question_darija = translate_en_to_darija_nllb(
                question_en, self.nllb_model, self.nllb_tokenizer
            )
            answer_darija = translate_en_to_darija_nllb(
                answer_en, self.nllb_model, self.nllb_tokenizer
            )
        except Exception as e:
            log.error(f"NLLB translation failed: {e}")
            question_darija = ""
            answer_darija = ""

        qa["question_en"] = question_en
        qa["answer_en"] = answer_en
        qa["question_fr"] = question_fr
        qa["answer_fr"] = answer_fr
        qa["question_darija"] = question_darija
        qa["answer_darija"] = answer_darija
        return qa

    def run(self):
        """Translate all QA pairs."""
        with open(self.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        log.info(f"Translating {len(data)} QA pairs...")

        # Load existing progress if available
        translated = []
        progress_file = self.output_file + ".progress"
        start_idx = 0
        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                translated = json.load(f)
            start_idx = len(translated)
            log.info(f"Resuming from pair {start_idx}")

        for i, qa in enumerate(data[start_idx:], start=start_idx):
            # Skip if already has translations
            if qa.get("question_fr") and qa.get("question_darija"):
                translated.append(qa)
                continue

            try:
                translated_qa = self.translate_pair(qa)
                translated.append(translated_qa)
            except Exception as e:
                log.error(f"Error translating pair {i}: {e}")
                translated.append(qa)

            if (i + 1) % 5 == 0:
                log.info(f"  Translated {i + 1}/{len(data)}")
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(translated, f, ensure_ascii=False, indent=2)

            # Small delay for Google Translate rate limit
            time.sleep(0.3)

        # Save final output
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=4)

        # Cleanup progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)

        log.info(f"Translation complete. Saved to {self.output_file}")
        return translated


if __name__ == "__main__":
    input_f = sys.argv[1] if len(sys.argv) > 1 else "data_icliniq_new.json"
    output_f = sys.argv[2] if len(sys.argv) > 2 else "data_translated.json"
    pipeline = TranslationPipeline(input_f, output_f)
    pipeline.run()
