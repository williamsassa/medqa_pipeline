#!/usr/bin/env python3
"""
Multi-API parallel synthetic medical QA generator v2.
Architecture: 1 dedicated thread per provider, shared specialty queue.
Each thread respects its own rate limit independently.
"""

import json
import os
import sys
import time
import logging
import random
import re
import threading
import queue
import requests
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("generate_multi_api.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
SYNTHETIC_DIR = PROJECT_DIR / "data_synthetic"
SYNTHETIC_DIR.mkdir(exist_ok=True)

TARGET_PER_SPECIALTY = 1500  # default; overridden per-specialty if data_plan_v3.json exists
BATCH_SIZE = 10  # 10 EN-only pairs per call (fast ~5-8s latency); translation done separately

# Load per-specialty tiered targets if available (v3)
_PLAN_PATH = Path(__file__).parent / "data_plan_v3.json"
TARGETS_V3: dict[str, int] = {}
if _PLAN_PATH.exists():
    try:
        _plan = json.loads(_PLAN_PATH.read_text(encoding="utf-8"))
        TARGETS_V3 = {row["specialty"]: row["target"] for row in _plan}
    except Exception as _e:
        TARGETS_V3 = {}


def target_for(specialty_id: str) -> int:
    return TARGETS_V3.get(specialty_id, TARGET_PER_SPECIALTY)
FILE_LOCKS = defaultdict(threading.Lock)  # Per-specialty file lock
GLOBAL_LOCK = threading.Lock()

# ── All 70 specialties ───────────────────────────────────────────────────────

SPECIALTIES = {
    "allergy_immunology": {"label_en": "Allergy and Immunology", "label_fr": "Allergie et Immunologie", "label_ar": "الحساسية والمناعة"},
    "anesthesiology": {"label_en": "Anesthesiology", "label_fr": "Anesthésiologie", "label_ar": "التخدير"},
    "cardiology": {"label_en": "Cardiology", "label_fr": "Cardiologie", "label_ar": "أمراض القلب"},
    "cosmetic_dermatology": {"label_en": "Cosmetic Dermatology", "label_fr": "Dermatologie Esthétique", "label_ar": "الجلدية التجميلية"},
    "dentistry": {"label_en": "Dentistry", "label_fr": "Dentisterie", "label_ar": "طب الأسنان"},
    "dermatology": {"label_en": "Dermatology", "label_fr": "Dermatologie", "label_ar": "الأمراض الجلدية"},
    "diabetes": {"label_en": "Diabetes Mellitus", "label_fr": "Diabétologie", "label_ar": "مرض السكري"},
    "dietetics": {"label_en": "Dietetics", "label_fr": "Diététique", "label_ar": "التغذية"},
    "endocrinology": {"label_en": "Endocrinology", "label_fr": "Endocrinologie", "label_ar": "الغدد الصماء"},
    "general_practitioner": {"label_en": "General Practice", "label_fr": "Médecine Générale", "label_ar": "الطب العام"},
    "gynecology": {"label_en": "Gynecology", "label_fr": "Gynécologie", "label_ar": "أمراض النساء"},
    "hematology": {"label_en": "Hematology", "label_fr": "Hématologie", "label_ar": "أمراض الدم"},
    "infectious_diseases": {"label_en": "Infectious Diseases", "label_fr": "Maladies Infectieuses", "label_ar": "الأمراض المعدية"},
    "internal_diseases": {"label_en": "Internal Medicine", "label_fr": "Médecine Interne", "label_ar": "الطب الداخلي"},
    "mental_health": {"label_en": "Mental Health", "label_fr": "Santé Mentale", "label_ar": "الصحة النفسية"},
    "neurology": {"label_en": "Neurology", "label_fr": "Neurologie", "label_ar": "الأعصاب"},
    "oncology": {"label_en": "Oncology", "label_fr": "Oncologie", "label_ar": "الأورام"},
    "ophthalmology": {"label_en": "Ophthalmology", "label_fr": "Ophtalmologie", "label_ar": "طب العيون"},
    "otorhinolaryngology": {"label_en": "Otorhinolaryngology (ENT)", "label_fr": "ORL", "label_ar": "أمراض الأنف والأذن والحنجرة"},
    "pediatrics": {"label_en": "Pediatrics", "label_fr": "Pédiatrie", "label_ar": "طب الأطفال"},
    "psychiatry": {"label_en": "Psychiatry", "label_fr": "Psychiatrie", "label_ar": "الطب النفسي"},
    "respiratory": {"label_en": "Respiratory Diseases", "label_fr": "Maladies Respiratoires", "label_ar": "أمراض الجهاز التنفسي"},
    "rheumatology": {"label_en": "Rheumatology & Orthopedics", "label_fr": "Rhumatologie et Orthopédie", "label_ar": "الروماتيزم والعظام"},
    "andrology": {"label_en": "Andrology", "label_fr": "Andrologie", "label_ar": "طب الذكورة"},
    "audiology": {"label_en": "Audiology", "label_fr": "Audiologie", "label_ar": "علم السمع"},
    "bariatric_surgery": {"label_en": "Bariatric Surgery", "label_fr": "Chirurgie Bariatrique", "label_ar": "جراحة السمنة"},
    "cardiothoracic_surgery": {"label_en": "Cardiothoracic Surgery", "label_fr": "Chirurgie Cardiothoracique", "label_ar": "جراحة القلب والصدر"},
    "child_health": {"label_en": "Child Health", "label_fr": "Santé de l'Enfant", "label_ar": "صحة الطفل"},
    "clinical_genetics": {"label_en": "Clinical Genetics", "label_fr": "Génétique Clinique", "label_ar": "الوراثة السريرية"},
    "community_medicine": {"label_en": "Community Medicine", "label_fr": "Médecine Communautaire", "label_ar": "طب المجتمع"},
    "critical_care": {"label_en": "Critical Care", "label_fr": "Soins Intensifs", "label_ar": "العناية المركزة"},
    "endodontics": {"label_en": "Endodontics", "label_fr": "Endodontie", "label_ar": "علاج جذور الأسنان"},
    "family_medicine": {"label_en": "Family Medicine", "label_fr": "Médecine Familiale", "label_ar": "طب الأسرة"},
    "fetal_medicine": {"label_en": "Fetal Medicine", "label_fr": "Médecine Fœtale", "label_ar": "طب الجنين"},
    "forensic_medicine": {"label_en": "Forensic Medicine", "label_fr": "Médecine Légale", "label_ar": "الطب الشرعي"},
    "gastroenterology": {"label_en": "Gastroenterology", "label_fr": "Gastro-entérologie", "label_ar": "أمراض الجهاز الهضمي"},
    "general_medicine": {"label_en": "General Medicine", "label_fr": "Médecine Générale", "label_ar": "الطب العام"},
    "general_surgery": {"label_en": "General Surgery", "label_fr": "Chirurgie Générale", "label_ar": "الجراحة العامة"},
    "geriatrics": {"label_en": "Geriatrics", "label_fr": "Gériatrie", "label_ar": "طب الشيخوخة"},
    "hair_transplant": {"label_en": "Hair Transplant Surgery", "label_fr": "Greffe de Cheveux", "label_ar": "زراعة الشعر"},
    "hiv_aids": {"label_en": "HIV/AIDS Medicine", "label_fr": "Médecine VIH/SIDA", "label_ar": "طب الإيدز"},
    "infertility": {"label_en": "Infertility", "label_fr": "Infertilité", "label_ar": "العقم"},
    "interventional_radiology": {"label_en": "Interventional Radiology", "label_fr": "Radiologie Interventionnelle", "label_ar": "الأشعة التداخلية"},
    "microbiology": {"label_en": "Microbiology", "label_fr": "Microbiologie", "label_ar": "علم الأحياء الدقيقة"},
    "nephrology": {"label_en": "Nephrology", "label_fr": "Néphrologie", "label_ar": "أمراض الكلى"},
    "neurosurgery": {"label_en": "Neurosurgery", "label_fr": "Neurochirurgie", "label_ar": "جراحة الأعصاب"},
    "nuclear_medicine": {"label_en": "Nuclear Medicine", "label_fr": "Médecine Nucléaire", "label_ar": "الطب النووي"},
    "oral_maxillofacial_surgery": {"label_en": "Oral & Maxillofacial Surgery", "label_fr": "Chirurgie Maxillo-faciale", "label_ar": "جراحة الوجه والفكين"},
    "orthodontics": {"label_en": "Orthodontics", "label_fr": "Orthodontie", "label_ar": "تقويم الأسنان"},
    "orthopedics": {"label_en": "Orthopedics & Traumatology", "label_fr": "Orthopédie et Traumatologie", "label_ar": "جراحة العظام"},
    "pain_medicine": {"label_en": "Pain Medicine", "label_fr": "Médecine de la Douleur", "label_ar": "طب الألم"},
    "pathology": {"label_en": "Pathology", "label_fr": "Pathologie", "label_ar": "علم الأمراض"},
    "pediatric_allergy": {"label_en": "Pediatric Allergy & Asthma", "label_fr": "Allergie et Asthme Pédiatrique", "label_ar": "حساسية وربو الأطفال"},
    "pediatric_cardiology": {"label_en": "Pediatric Cardiology", "label_fr": "Cardiologie Pédiatrique", "label_ar": "قلب الأطفال"},
    "pediatric_dentistry": {"label_en": "Pediatric Dentistry", "label_fr": "Dentisterie Pédiatrique", "label_ar": "طب أسنان الأطفال"},
    "pediatric_surgery": {"label_en": "Pediatric Surgery", "label_fr": "Chirurgie Pédiatrique", "label_ar": "جراحة الأطفال"},
    "periodontics": {"label_en": "Periodontics", "label_fr": "Parodontologie", "label_ar": "أمراض اللثة"},
    "pharmacology": {"label_en": "Pharmacology", "label_fr": "Pharmacologie", "label_ar": "علم الأدوية"},
    "plastic_surgery": {"label_en": "Plastic Surgery", "label_fr": "Chirurgie Plastique", "label_ar": "الجراحة التجميلية"},
    "preventive_medicine": {"label_en": "Preventive Medicine", "label_fr": "Médecine Préventive", "label_ar": "الطب الوقائي"},
    "radiation_oncology": {"label_en": "Radiation Oncology", "label_fr": "Radio-oncologie", "label_ar": "علاج الأورام بالإشعاع"},
    "radiology": {"label_en": "Radiology", "label_fr": "Radiologie", "label_ar": "الأشعة"},
    "sexology": {"label_en": "Sexology", "label_fr": "Sexologie", "label_ar": "الطب الجنسي"},
    "sleep_medicine": {"label_en": "Sleep Medicine", "label_fr": "Médecine du Sommeil", "label_ar": "طب النوم"},
    "spine_surgery": {"label_en": "Spine Surgery", "label_fr": "Chirurgie du Rachis", "label_ar": "جراحة العمود الفقري"},
    "surgical_gastroenterology": {"label_en": "Surgical Gastroenterology", "label_fr": "Gastro-entérologie Chirurgicale", "label_ar": "جراحة الجهاز الهضمي"},
    "surgical_oncology": {"label_en": "Surgical Oncology", "label_fr": "Oncologie Chirurgicale", "label_ar": "جراحة الأورام"},
    "toxicology": {"label_en": "Toxicology", "label_fr": "Toxicologie", "label_ar": "علم السموم"},
    "urology": {"label_en": "Urology", "label_fr": "Urologie", "label_ar": "المسالك البولية"},
    "vascular_surgery": {"label_en": "Vascular Surgery", "label_fr": "Chirurgie Vasculaire", "label_ar": "جراحة الأوعية الدموية"},
}


# ── Utility functions ─────────────────────────────────────────────────────────

def load_seed_data() -> dict[str, list[dict]]:
    seeds = defaultdict(list)
    scraped_dir = PROJECT_DIR / "data_scraped"
    if scraped_dir.exists():
        for f in scraped_dir.glob("*.json"):
            if f.name.startswith("_"):
                continue
            sid = f.stem
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    for qa in json.load(fh):
                        seeds[sid].append({"question": qa["question"], "answer": qa["answer"]})
            except Exception:
                pass
    return dict(seeds)


def parse_json_from_response(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = re.sub(r",\s*}", "}", json_str)
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


def build_prompt(specialty_name: str, seed_examples: list[dict]) -> str:
    contexts = [
        "patients describing new symptoms", "patients seeking second opinions",
        "parents asking about children's conditions", "patients worried about test results",
        "elderly patients with chronic conditions", "patients asking about medication side effects",
        "patients with worsening chronic symptoms", "patients asking about surgical options",
        "patients discussing lifestyle changes", "patients asking about prevention",
        "patients with family history concerns", "patients describing acute episodes",
        "patients asking about treatment alternatives", "patients discussing recovery",
        "patients asking about diagnostic procedures",
    ]
    context = random.choice(contexts)

    return f"""Generate exactly {BATCH_SIZE} realistic medical Q&A pairs for {specialty_name}.
Context: {context}.

Rules:
1. Each pair: ONE patient question + ONE doctor answer
2. Patient questions: detailed, realistic (symptoms, age, history, concerns) - minimum 3 sentences
3. Doctor answers: professional, thorough, clinically accurate - minimum 5 sentences
4. Cover DIVERSE conditions within {specialty_name}
5. Include realistic patient details (age, gender, duration, medications)
6. Vary severity and demographics

Output ONLY a valid JSON array with "question" and "answer" keys. No markdown. No explanation."""


def load_specialty_file(specialty_id: str) -> list[dict]:
    f = SYNTHETIC_DIR / f"{specialty_id}.json"
    if f.exists():
        with open(f, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


def save_specialty_file(specialty_id: str, data: list[dict]):
    f = SYNTHETIC_DIR / f"{specialty_id}.json"
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def get_specialty_count(specialty_id: str) -> int:
    with FILE_LOCKS[specialty_id]:
        return len(load_specialty_file(specialty_id))


def append_pairs(specialty_id: str, new_pairs: list[dict]) -> int:
    """Append pairs with dedup, return new total."""
    with FILE_LOCKS[specialty_id]:
        existing = load_specialty_file(specialty_id)
        existing_qs = set(
            (p.get("question_en") or p.get("question") or "").lower()[:100]
            for p in existing
        )
        added = 0
        for p in new_pairs:
            qk = (p.get("question_en") or p.get("question") or "").lower()[:100]
            if qk not in existing_qs:
                existing.append(p)
                existing_qs.add(qk)
                added += 1
        if added:
            save_specialty_file(specialty_id, existing)
        return len(existing)


# ── Provider definitions ──────────────────────────────────────────────────────

class Provider:
    def __init__(self, name, url, model, api_key, rpm, is_gemini=False, json_mode=False):
        self.name = name
        self.url = url
        self.model = model
        self.api_key = api_key
        self.rpm = rpm
        self.min_interval = 60.0 / rpm
        self._last_call = 0
        self._rate_lock = threading.Lock()  # Thread-safe rate limiting
        self.is_gemini = is_gemini
        self.json_mode = json_mode  # Use response_format: json_object
        self.consecutive_errors = 0
        self.disabled = False

    def wait_for_rate_limit(self):
        """Thread-safe rate limiter — sleep outside lock so workers can overlap API calls."""
        while True:
            with self._rate_lock:
                now = time.time()
                elapsed = now - self._last_call
                if elapsed >= self.min_interval:
                    self._last_call = now
                    return  # Proceed
            # Not our turn yet — sleep briefly outside the lock
            time.sleep(min(0.1, self.min_interval / 4))

    def call(self, prompt: str) -> str:
        self.wait_for_rate_limit()

        if self.is_gemini:
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.9, "maxOutputTokens": 8192}
            }
        else:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a medical expert generating realistic patient-doctor Q&A pairs. Output ONLY valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.9,
                "max_tokens": 8192,
            }
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}

        resp = requests.post(self.url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        if self.is_gemini:
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""
        else:
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""


def build_providers() -> list[Provider]:
    providers = []

    # (name, url, model, env_key, rpm, json_mode)
    configs = [
        # Mistral: 60 req/min, 375K tokens/min, json_mode — PRIMARY WORKHORSE
        ("mistral", "https://api.mistral.ai/v1/chat/completions", "mistral-small-latest", "MISTRAL_API_KEY", 50, True),
        # Cerebras: 30 req/min, but frequent 429 — backup
        ("cerebras", "https://api.cerebras.ai/v1/chat/completions", "llama3.1-8b", "CEREBRAS_API_KEY", 5, False),
        # Groq: 12K tokens/min, json_mode — backup
        ("groq", "https://api.groq.com/openai/v1/chat/completions", "llama-3.3-70b-versatile", "GROQ_API_KEY", 1, True),
        # OpenRouter: backup
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "meta-llama/llama-3.3-70b-instruct:free", "OPENROUTER_API_KEY", 1, False),
    ]

    for name, url, model, env_key, rpm, json_mode in configs:
        key = os.getenv(env_key, "").strip()
        if key:
            providers.append(Provider(name, url, model, key, rpm, json_mode=json_mode))
            log.info(f"  {name}: {model} ({rpm} rpm, json_mode={json_mode})")

    # Gemini - disabled (quota exhausted)
    # gkey = os.getenv("GOOGLE_API_KEY", "").strip()
    # if gkey:
    #     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gkey}"
    #     providers.append(Provider("gemini", url, "gemini-2.0-flash", gkey, 5, is_gemini=True))

    return providers


# ── Worker thread ─────────────────────────────────────────────────────────────

def provider_worker(provider: Provider, work_queue: queue.Queue,
                    seeds: dict, stats: dict):
    """Dedicated worker thread for one provider. Pulls specialties from shared queue."""
    while True:
        try:
            specialty_id = work_queue.get(timeout=5)
        except queue.Empty:
            # Check if there's still work to do (other workers may re-queue items)
            if work_queue.empty():
                log.info(f"  [{provider.name}] Queue empty, worker done")
                return
            continue

        if provider.disabled:
            work_queue.put(specialty_id)  # Put back for other providers
            log.info(f"  [{provider.name}] Disabled, returning {specialty_id} to queue")
            return

        spec_info = SPECIALTIES[specialty_id]
        seed_examples = seeds.get(specialty_id, [])

        current_count = get_specialty_count(specialty_id)
        spec_target = target_for(specialty_id)
        if current_count >= spec_target:
            work_queue.task_done()
            continue

        needed = spec_target - current_count
        batches = min(200, (needed + BATCH_SIZE - 1) // BATCH_SIZE)

        log.info(f"  [{provider.name}] Starting {specialty_id} ({current_count}/{spec_target}, {batches} batches)")

        batch_done = 0
        total_added = 0

        for b in range(batches):
            if provider.disabled:
                break

            # Check if specialty is complete (another provider may have finished it)
            if b % 10 == 0 and b > 0:
                current_count = get_specialty_count(specialty_id)
                if current_count >= spec_target:
                    log.info(f"  [{provider.name}] {specialty_id} completed by another provider ({current_count})")
                    break

            prompt = build_prompt(spec_info["label_en"], seed_examples)

            for attempt in range(3):
                try:
                    response = provider.call(prompt)
                    pairs = parse_json_from_response(response)

                    valid = []
                    for p in pairs:
                        if not isinstance(p, dict):
                            continue
                        q = (p.get("question") or "").strip()
                        a = (p.get("answer") or "").strip()
                        if len(q) < 40 or len(a) < 80:
                            continue
                        valid.append({
                            "question": q,
                            "answer": a,
                            "source": f"synthetic_{provider.name}",
                            "specialty_id": specialty_id,
                            "specialty_en": spec_info["label_en"],
                            "specialty_fr": spec_info["label_fr"],
                            "specialty_ar": spec_info["label_ar"],
                        })

                    if valid:
                        new_total = append_pairs(specialty_id, valid)
                        total_added += len(valid)
                        provider.consecutive_errors = 0

                        if (b + 1) % 10 == 0 or b == 0:
                            log.info(f"    [{provider.name}] {specialty_id} batch {b+1}: +{len(valid)} (total: {new_total}/{spec_target})")

                        if new_total >= spec_target:
                            log.info(f"  [{provider.name}] {specialty_id} COMPLETE! ({new_total})")
                            break
                        break  # Success, next batch

                    else:
                        log.warning(f"    [{provider.name}] {specialty_id} batch {b+1} attempt {attempt+1}: {len(pairs)} raw, 0 valid")

                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    if status == 429:
                        wait = min(120, 10 * (attempt + 1))
                        log.warning(f"    [{provider.name}] 429 rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    elif status in (401, 403):
                        log.error(f"    [{provider.name}] AUTH ERROR {status}, disabling!")
                        provider.disabled = True
                        break
                    elif status == 402:
                        log.error(f"    [{provider.name}] PAYMENT REQUIRED, disabling!")
                        provider.disabled = True
                        break
                    else:
                        provider.consecutive_errors += 1
                        log.warning(f"    [{provider.name}] HTTP {status}: {e}")
                        time.sleep(5)
                except Exception as e:
                    provider.consecutive_errors += 1
                    log.warning(f"    [{provider.name}] Error: {e}")
                    time.sleep(5)

                if provider.consecutive_errors >= 10:
                    log.error(f"    [{provider.name}] Too many errors, disabling!")
                    provider.disabled = True
                    break

            if provider.disabled:
                break

            batch_done += 1

        # If specialty not yet complete, re-queue it
        current_count = get_specialty_count(specialty_id)
        if current_count < target_for(specialty_id) and not provider.disabled:
            work_queue.put(specialty_id)

        with GLOBAL_LOCK:
            stats[provider.name] = stats.get(provider.name, 0) + total_added

        work_queue.task_done()

        if provider.disabled:
            log.info(f"  [{provider.name}] Worker shutting down (disabled)")
            return


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seeds = load_seed_data()
    providers = build_providers()

    if not providers:
        log.error("No API providers configured! Add keys to .env")
        sys.exit(1)

    # Build work queue
    work_queue = queue.Queue()
    specialties_needing_work = []

    for sid in sorted(SPECIALTIES.keys()):
        count = get_specialty_count(sid)
        if count < target_for(sid):
            specialties_needing_work.append((count, sid))

    # Sort by least data first
    specialties_needing_work.sort()
    for _, sid in specialties_needing_work:
        work_queue.put(sid)

    total_existing = sum(get_specialty_count(sid) for sid in SPECIALTIES)
    total_target = sum(target_for(sid) for sid in SPECIALTIES)

    log.info(f"\n{'='*70}")
    log.info(f"MULTI-API PARALLEL SYNTHETIC QA GENERATION v2")
    log.info(f"{'='*70}")
    log.info(f"Providers: {len(providers)} ({', '.join(p.name for p in providers)})")
    log.info(f"Specialties needing data: {len(specialties_needing_work)}")
    log.info(f"Current total: {total_existing} / {total_target}")
    log.info(f"Target: tiered (v3 plan) total={total_target}")
    log.info(f"{'='*70}\n")

    stats = {}
    threads = []

    # Launch multiple threads for high-RPM providers, 1 for others
    for provider in providers:
        # Mistral can handle 6 concurrent workers (60 RPM limit / ~5s per req)
        num_workers = 6 if provider.name == "mistral" else 1
        for w in range(num_workers):
            t = threading.Thread(
                target=provider_worker,
                args=(provider, work_queue, seeds, stats),
                name=f"worker-{provider.name}-{w}",
                daemon=True
            )
            threads.append(t)
            t.start()
        log.info(f"  Started {num_workers} worker(s): {provider.name}")

    # Monitor progress
    start_time = time.time()
    last_report = 0
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            elapsed = time.time() - start_time

            if elapsed - last_report >= 120:  # Report every 2 min
                last_report = elapsed
                total_now = sum(get_specialty_count(sid) for sid in SPECIALTIES)
                complete = sum(1 for sid in SPECIALTIES if get_specialty_count(sid) >= target_for(sid))
                rate = (total_now - total_existing) / (elapsed / 60) if elapsed > 0 else 0
                eta_min = (total_target - total_now) / rate if rate > 0 else float('inf')

                log.info(f"\n  === PROGRESS ({elapsed/60:.0f} min) ===")
                log.info(f"  Total pairs: {total_now}/{total_target} (+{total_now - total_existing} new)")
                log.info(f"  Complete: {complete}/{len(SPECIALTIES)} specialties")
                log.info(f"  Rate: {rate:.0f} pairs/min")
                if eta_min < float('inf'):
                    log.info(f"  ETA: {eta_min:.0f} min ({eta_min/60:.1f} hours)")
                log.info(f"  Active providers: {sum(1 for p in providers if not p.disabled)}/{len(providers)}")
                log.info(f"  Per provider: {json.dumps(stats)}")

                # Save progress
                progress = {sid: get_specialty_count(sid) for sid in SPECIALTIES}
                with open(SYNTHETIC_DIR / "_progress.json", "w", encoding="utf-8") as f:
                    json.dump(progress, f, indent=2)

    except KeyboardInterrupt:
        log.info("\n\nInterrupted by user. Progress saved.")

    # Final report
    total_final = sum(get_specialty_count(sid) for sid in SPECIALTIES)
    complete = sum(1 for sid in SPECIALTIES if get_specialty_count(sid) >= target_for(sid))
    elapsed = time.time() - start_time

    log.info(f"\n{'='*70}")
    log.info(f"FINAL REPORT")
    log.info(f"{'='*70}")
    log.info(f"Total pairs: {total_final}/{total_target}")
    log.info(f"New pairs generated: {total_final - total_existing}")
    log.info(f"Complete specialties: {complete}/{len(SPECIALTIES)}")
    log.info(f"Time elapsed: {elapsed/60:.1f} min")
    log.info(f"Per provider: {json.dumps(stats)}")
    log.info(f"{'='*70}")

    for sid in sorted(SPECIALTIES.keys()):
        count = get_specialty_count(sid)
        scraped = len(seeds.get(sid, []))
        tgt = target_for(sid); status = "DONE" if count >= tgt else f"NEED {tgt - count}"
        print(f"  {sid:40s}: {count:5d} synth + {scraped:3d} scraped [{status}]")


if __name__ == "__main__":
    main()
