"""medical_metrics.py — Clinical-grade evaluation metrics beyond BLEU/ROUGE.

Functions
---------
extract_entities(text, lang)            → {"drugs": [...], "symptoms": [...], "exams": [...]}
entity_recall(pred, ref, lang)          → per-category recall + macro
hallucination_rate(pred, ref, lang)     → drug/dosage/exam mentions in pred but absent in ref
diagnostic_precision_at_k(pred, ref, k) → ref diagnosis present in pred top-K candidates
reference_validity(pred)                → DOI / PubMed PMID / ICD-10 patterns valid?
pathology_of(text)                      → bucket : cardiovascular, infectious, endocrine, …
aggregate_medical(predictions, references, langs, rows) → per-pathology + macro report

The module is **pure-Python regex-based** (no extra deps beyond what evaluate.py
already needs). For higher-fidelity NER, plug in scispaCy / medspaCy here later.
"""
from __future__ import annotations
import re
from collections import defaultdict, Counter
from typing import Iterable

from dorosz_kg import get_kg


# ── Multilingual symptom / exam lexicons ─────────────────────────────────
# Curated, non-exhaustive but covers the high-frequency clinical vocabulary
# we expect in BRAIN HEALTH evaluation rows.

_SYMPTOM_PATTERNS = {
    # canonical → list of patterns (lowercased)
    "fever":                ["fever", "fièvre", "pyrexia", "fébrile", "حمى", "سخانة", "سخونة"],
    "pain":                 ["pain", "ache", "aching", "douleur", "ألم", "وجع"],
    "chest_pain":           ["chest pain", "thoracic pain", "douleur thoracique", "ألم في الصدر"],
    "headache":             ["headache", "migraine", "céphalée", "cephalalgia", "صداع"],
    "cough":                ["cough", "toux", "سعال", "كحة"],
    "dyspnea":              ["shortness of breath", "dyspnea", "dyspnée", "essoufflement", "صعوبة التنفس", "ضيق نفس"],
    "fatigue":              ["fatigue", "tired", "asthenia", "asthénie", "épuisement", "تعب", "إرهاق"],
    "nausea":               ["nausea", "nauseous", "nausée", "غثيان"],
    "vomiting":             ["vomit", "vomiting", "vomissement", "تقيؤ"],
    "diarrhea":             ["diarrhea", "diarrhoea", "diarrhée", "إسهال"],
    "constipation":         ["constipation", "إمساك"],
    "rash":                 ["rash", "éruption", "exanthème", "طفح جلدي"],
    "itching":              ["itching", "pruritus", "prurit", "حكة"],
    "weight_loss":          ["weight loss", "perte de poids", "نقص في الوزن"],
    "weight_gain":          ["weight gain", "prise de poids", "زيادة في الوزن"],
    "dizziness":            ["dizziness", "vertigo", "vertige", "étourdissement", "دوخة"],
    "chest_tightness":      ["chest tightness", "oppression thoracique", "ضيق في الصدر"],
    "palpitations":         ["palpitation", "خفقان"],
    "edema":                ["edema", "œdème", "oedème", "swelling", "تورم"],
    "syncope":              ["syncope", "fainting", "perte de conscience", "إغماء"],
    "anxiety":              ["anxiety", "anxiété", "قلق"],
    "depression":           ["depression", "depressed", "dépression", "اكتئاب"],
    "insomnia":             ["insomnia", "insomnie", "أرق"],
    "polyuria":             ["polyuria", "polyurie", "تبول متكرر"],
    "polydipsia":           ["polydipsia", "polydipsie", "عطش شديد"],
    "abdominal_pain":       ["abdominal pain", "stomach pain", "douleur abdominale", "ألم في البطن"],
    "joint_pain":           ["joint pain", "arthralgia", "douleur articulaire", "ألم في المفاصل"],
    "dysuria":              ["dysuria", "dysurie", "حرقان عند التبول"],
    "hematuria":            ["hematuria", "hématurie", "دم في البول"],
}

_EXAM_PATTERNS = {
    "ecg":                  ["ecg", "ekg", "électrocardiogramme", "electrocardiogram", "تخطيط القلب"],
    "blood_test":           ["blood test", "blood work", "cbc", "complete blood count",
                              "nfs", "numération", "hémogramme", "تحليل الدم"],
    "urine_test":           ["urinalysis", "urine test", "ecbu", "تحليل البول"],
    "ct_scan":              ["ct scan", "computed tomography", "scanner", "tdm", "tomodensitométrie"],
    "mri":                  ["mri", "magnetic resonance", "irm", "imagerie par résonance", "رنين مغناطيسي"],
    "x_ray":                ["x-ray", "xray", "radiographie", "radio", "أشعة"],
    "ultrasound":           ["ultrasound", "echo", "échographie", "تصوير بالأمواج فوق الصوتية"],
    "biopsy":                ["biopsy", "biopsie", "خزعة"],
    "endoscopy":            ["endoscopy", "endoscopie", "تنظير"],
    "colonoscopy":          ["colonoscopy", "coloscopie", "تنظير القولون"],
    "echocardiography":     ["echocardiography", "échocardiographie", "تخطيط صدى القلب"],
    "stress_test":          ["stress test", "épreuve d'effort", "اختبار الجهد"],
    "glycemia":             ["blood glucose", "glycemia", "glycémie", "hba1c", "السكر في الدم"],
    "lipid_panel":          ["lipid panel", "bilan lipidique", "cholesterol", "cholestérol"],
    "lumbar_puncture":      ["lumbar puncture", "ponction lombaire", "البزل القطني"],
    "spirometry":           ["spirometry", "spirométrie", "efr", "epreuves fonctionnelles", "قياس التنفس"],
    "tsh":                  ["tsh", "thyroid stimulating hormone", "tsh", "هرمون الغدة الدرقية"],
}

_DIAGNOSTIC_KEYWORDS = (
    # Markers that often precede a diagnosis in a free-form medical answer.
    "diagnosis:", "diagnostic :", "diagnostic:", "likely diagnosis", "diagnostic différentiel",
    "probable diagnosis", "diagnostic probable", "could be", "il s'agit probablement",
    "the condition is", "il pourrait s'agir", "الأرجح", "التشخيص"
)

_PATHOLOGY_BUCKETS = {
    "cardiovascular":   ["heart", "cardio", "ischemic", "infarct", "myocard", "angina", "stroke", "avc",
                          "hypertension", "tachycardi", "bradycardi", "arrhythmi", "atrial fib",
                          "ventricular", "hypertensi", "fibrillation", "thrombos",
                          "chest pain", "douleur thoracique", "st elevation", "st-elevation",
                          "valvular", "valvulair", "embolism", "embolie",
                          "ihme", "qlb", "قلب", "ضغط الدم", "احتشاء"],
    "infectious":       ["infection", "infectious", "bacterial", "viral", "sepsis", "pneumonia",
                          "pneumonie", "tuberculosis", "tuberculose", "covid", "hiv", "vih",
                          "fever", "fièvre", "abscess", "abcès", "antibiotique", "antibiotic",
                          "تعفن", "عدوى", "حمى", "السيدا"],
    "endocrine":        ["diabet", "diabète", "thyroid", "thyroïd", "hba1c", "insulin", "insuline",
                          "hyperthyr", "hypothyr", "endocrin", "السكري", "الغدة"],
    "respiratory":      ["asthma", "asthme", "copd", "bpco", "pneumonia", "pneumonie", "bronchit",
                          "respirator", "lung", "poumon", "respirat", "تنفس", "ربو"],
    "gastrointestinal": ["ulcer", "ulcère", "gastrit", "reflux", "rgo", "diarrhea", "diarrhée",
                          "constipat", "ibs", "crohn", "colit", "hépatit", "hepatit", "appendic",
                          "معدة", "إسهال", "إمساك"],
    "neurological":     ["seizure", "epileps", "épileps", "parkinson", "alzheimer", "stroke",
                          "avc", "migraine", "headache", "céphalée", "neuropath", "encephal",
                          "تشنج", "صرع", "صداع"],
    "musculoskeletal":  ["arthritis", "arthrite", "osteoarthr", "rheumat", "rhumat", "fracture",
                          "tendin", "ligament", "back pain", "lombalgie", "ألم العظام", "كسر"],
    "psychiatric":      ["depression", "dépression", "anxiety", "anxiété", "bipolar", "bipolaire",
                          "schizophr", "ptsd", "psychos", "ocd", "tdah", "اكتئاب", "قلق"],
    "renal":            ["kidney", "rein", "renal", "rénal", "nephr", "néphr", "uremia",
                          "dialys", "كلية"],
    "oncology":         ["cancer", "tumor", "tumeur", "leukemi", "leucémie", "lymphom",
                          "metastas", "métastas", "chemo", "chimio", "ورم", "سرطان"],
    "dermatology":      ["rash", "éruption", "eczema", "eczéma", "psoriasis", "acne", "acné",
                          "dermatit", "dermatos", "طفح", "إكزيما"],
    "obgyn":            ["pregnan", "grossesse", "menstruat", "règles", "obstetric", "obstétr",
                          "gynecolog", "gynécolog", "menopaus", "ménopaus", "الحمل", "دورة"],
    "pediatrics":       ["pediatr", "pédiatr", "newborn", "nouveau-né", "infant", "nourrisson",
                          "vaccination", "vaccin", "أطفال", "رضيع"],
    "allergy_immuno":   ["allerg", "anaphyl", "auto-immun", "autoimmun", "lupus", "rheumat", "vasculit"],
}

# Dosage regex (numeric + unit) to detect drug-specific hallucinations
_DOSAGE_RE = re.compile(
    r"\b\d{1,4}(?:[\.,]\d+)?\s*(?:mg|g|mcg|µg|ug|ml|l|iu|ui|cp|comprim[eé]s?|gélules?|"
    r"capsules?|tablets?|drops?|gouttes?|times?|fois|daily|jour|by mouth|po|iv|im|sc)\b",
    flags=re.IGNORECASE)

# DOI / PubMed / ICD-10 patterns
_DOI_RE  = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", flags=re.IGNORECASE)
_PMID_RE = re.compile(r"\b(?:pmid[:\s]*|pubmed[:\s]*)(\d{6,9})\b", flags=re.IGNORECASE)
_ICD10_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d{1,3})?\b")


# ── Public API ────────────────────────────────────────────────────────────
def extract_entities(text: str, lang: str | None = None) -> dict[str, list[str]]:
    """Return dict with detected drugs/symptoms/exams. Lowercased canonical keys."""
    if not text:
        return {"drugs": [], "symptoms": [], "exams": [], "dosages": []}
    low = text.lower()
    drugs = []
    kg = get_kg()
    if not kg.empty and kg._scan_re is not None:
        for m in kg._scan_re.finditer(text):
            drugs.append(m.group(1).lower())
    drugs = sorted(set(drugs))
    syms = sorted({canon for canon, pats in _SYMPTOM_PATTERNS.items()
                   if any(p in low for p in pats)})
    exs  = sorted({canon for canon, pats in _EXAM_PATTERNS.items()
                   if any(p in low for p in pats)})
    doses = _DOSAGE_RE.findall(text)
    return {"drugs": drugs, "symptoms": syms, "exams": exs, "dosages": doses}


def entity_recall(pred: str, ref: str, lang: str | None = None) -> dict:
    """Recall = |entities in ref ∩ entities in pred| / |entities in ref| per category.
    Returns dict with per-category recall + macro average. NaN-safe (empty ref → None)."""
    ep = extract_entities(pred, lang)
    er = extract_entities(ref,  lang)
    out: dict[str, float | None] = {}
    macro_vals = []
    for cat in ("drugs", "symptoms", "exams"):
        ref_set, pred_set = set(er[cat]), set(ep[cat])
        if not ref_set:
            out[f"recall_{cat}"] = None
        else:
            r = len(ref_set & pred_set) / len(ref_set)
            out[f"recall_{cat}"] = r
            macro_vals.append(r)
    out["recall_entities_macro"] = (sum(macro_vals) / len(macro_vals)) if macro_vals else 0.0
    return out


def hallucination_rate(pred: str, ref: str, lang: str | None = None) -> dict:
    """Hallucination = entity present in pred but absent from ref (potentially invented).

    Returns rates per category in [0, 1]: 0 = no hallucination, 1 = everything invented.
    `dosage_hallucination_rate` is a critical signal for medical safety.
    """
    ep = extract_entities(pred, lang)
    er = extract_entities(ref,  lang)
    out = {}
    for cat in ("drugs", "symptoms", "exams"):
        pred_set, ref_set = set(ep[cat]), set(er[cat])
        if not pred_set:
            out[f"hallu_{cat}"] = 0.0
        else:
            out[f"hallu_{cat}"] = len(pred_set - ref_set) / len(pred_set)
    # Dosages: count exact-string matches
    pred_doses = set(_norm(d) for d in ep["dosages"])
    ref_doses  = set(_norm(d) for d in er["dosages"])
    if pred_doses:
        out["hallu_dosage"] = len(pred_doses - ref_doses) / len(pred_doses)
    else:
        out["hallu_dosage"] = 0.0
    out["hallu_macro"] = (out["hallu_drugs"] + out["hallu_symptoms"]
                          + out["hallu_exams"] + out["hallu_dosage"]) / 4.0
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def diagnostic_precision_at_k(pred: str, ref_diagnosis: str, k: int = 3) -> dict:
    """If the prediction lists candidate diagnoses (numbered or bullet list, OR
    after a 'diagnostic:' marker), check whether the reference diagnosis is in
    the top-K. We extract candidates by simple heuristics — numbered items
    "1. … 2. … 3. …" or bullets "- …" or split on " or "/" ou ".

    Returns: { hit@k: 0/1, k, candidates: [...] }
    """
    if not pred or not ref_diagnosis:
        return {"hit_at_k": None, "k": k, "candidates": []}
    candidates: list[str] = []
    text = pred
    # try to find a "diagnostic:" / "diagnosis:" cue and slice after
    for cue in _DIAGNOSTIC_KEYWORDS:
        idx = text.lower().find(cue)
        if idx != -1:
            text = text[idx + len(cue):].strip()
            break
    # numbered items "1. ..." — split on the numbered separator pattern
    # (won't break on digits inside a value like "Type 2 diabetes")
    padded = " " + text
    numbered = re.split(r"\s+\d+[\.\)]\s+", padded)
    numbered = [p for p in numbered if p.strip()]   # drop the lead empty chunk
    bullets  = re.findall(r"^\s*[-•\*]\s*([^\n]+)", text, flags=re.MULTILINE)
    parts: list[str] = []
    # Use numbered split when it actually fired (≥2 items extracted)
    if len(numbered) >= 2:
        parts = numbered
    elif bullets:
        parts = bullets
    else:
        # split on "or" / "ou" / "ou bien"
        parts = re.split(r"\bor\b|\bou\b", text, maxsplit=k)
    candidates = [c.strip(" .;:,—-\n") for c in parts if c.strip()]
    candidates = candidates[:k]
    ref_norm = _norm(ref_diagnosis)
    # match ref against any candidate (substring either way, case-insensitive)
    hit = 0
    for cand in candidates:
        c = _norm(cand)
        if not c or len(c) < 4:
            continue
        if ref_norm in c or c in ref_norm:
            hit = 1
            break
    return {"hit_at_k": hit, "k": k, "candidates": candidates}


def reference_validity(pred: str) -> dict:
    """Detect citation patterns the model might invent. Cannot verify content
    without a network call, but format validity catches the most common
    hallucination shape (made-up DOIs / PMIDs)."""
    if not pred:
        return {"n_dois": 0, "n_pmids": 0, "n_icd10": 0, "any_present": False}
    n_doi = len(_DOI_RE.findall(pred))
    n_pmid = len(_PMID_RE.findall(pred))
    n_icd = len(_ICD10_RE.findall(pred))
    return {"n_dois": n_doi, "n_pmids": n_pmid, "n_icd10": n_icd,
            "any_present": (n_doi + n_pmid + n_icd) > 0}


def pathology_of(text: str) -> str:
    """Categorize a question into one of _PATHOLOGY_BUCKETS by keyword vote."""
    if not text:
        return "other"
    low = text.lower()
    scores = Counter()
    for cat, kws in _PATHOLOGY_BUCKETS.items():
        for kw in kws:
            if kw in low:
                scores[cat] += 1
    if not scores:
        return "other"
    return scores.most_common(1)[0][0]


def aggregate_medical(predictions: list[str], references: list[str],
                       langs: list[str], rows: list[dict]
                      ) -> dict:
    """Master aggregator. Returns:
       {
         "macro": {recall_*, hallu_*, hit@3, ...},
         "per_pathology": { cat: { metrics... }, ... },
         "per_language":  { lang: { metrics... }, ... },
         "n_with_dosage_hallu": int,   # safety alarm count
       }
    """
    assert len(predictions) == len(references) == len(langs) == len(rows)
    rec: list[dict] = []
    halls: list[dict] = []
    p_at_k: list[int | None] = []
    valid_refs: list[bool] = []
    pathologies: list[str] = []
    n_dosage_hallu = 0

    for p, r, lang, row in zip(predictions, references, langs, rows):
        rec.append(entity_recall(p, r, lang))
        h = hallucination_rate(p, r, lang)
        halls.append(h)
        if h["hallu_dosage"] > 0:
            n_dosage_hallu += 1
        p_at_k.append(diagnostic_precision_at_k(p, r, k=3)["hit_at_k"])
        valid_refs.append(reference_validity(p)["any_present"])
        # bucket on the question (or context+question) text
        q_text = (row.get("question_en") or row.get("question_fr")
                  or row.get("question_darija") or row.get("question") or "")
        pathologies.append(pathology_of(q_text))

    def _mean(vals, drop_none=True):
        if drop_none:
            vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    macro = {
        "entity_recall_drugs":     _mean([r["recall_drugs"]    for r in rec]),
        "entity_recall_symptoms":  _mean([r["recall_symptoms"] for r in rec]),
        "entity_recall_exams":     _mean([r["recall_exams"]    for r in rec]),
        "entity_recall_macro":     _mean([r["recall_entities_macro"] for r in rec]),
        "hallu_drugs":             _mean([h["hallu_drugs"]     for h in halls]),
        "hallu_symptoms":          _mean([h["hallu_symptoms"]  for h in halls]),
        "hallu_exams":             _mean([h["hallu_exams"]     for h in halls]),
        "hallu_dosage":            _mean([h["hallu_dosage"]    for h in halls]),
        "hallu_macro":             _mean([h["hallu_macro"]     for h in halls]),
        "hit_at_k":                _mean([v for v in p_at_k    if v is not None] or [0]),
        "ref_citations_pct":       sum(valid_refs) / max(1, len(valid_refs)),
        "n_dosage_hallu_samples":  n_dosage_hallu,
    }

    # Per-pathology breakdown
    per_path: dict[str, dict] = {}
    by_path: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(pathologies):
        by_path[c].append(i)
    for cat, idxs in by_path.items():
        if len(idxs) < 3:
            continue   # too few to compute meaningful averages
        sub_rec  = [rec[i]   for i in idxs]
        sub_hall = [halls[i] for i in idxs]
        sub_pk   = [p_at_k[i] for i in idxs if p_at_k[i] is not None]
        per_path[cat] = {
            "n":                  len(idxs),
            "entity_recall_macro": _mean([r["recall_entities_macro"] for r in sub_rec]),
            "hallu_macro":         _mean([h["hallu_macro"]            for h in sub_hall]),
            "hit_at_k":            _mean(sub_pk if sub_pk else [0]),
        }

    # Per-language breakdown
    per_lang: dict[str, dict] = {}
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(langs):
        by_lang[l].append(i)
    for l, idxs in by_lang.items():
        sub_rec  = [rec[i]   for i in idxs]
        sub_hall = [halls[i] for i in idxs]
        sub_pk   = [p_at_k[i] for i in idxs if p_at_k[i] is not None]
        per_lang[l] = {
            "n":                  len(idxs),
            "entity_recall_macro": _mean([r["recall_entities_macro"] for r in sub_rec]),
            "hallu_macro":         _mean([h["hallu_macro"]            for h in sub_hall]),
            "hit_at_k":            _mean(sub_pk if sub_pk else [0]),
        }

    return {
        "macro":         macro,
        "per_pathology": per_path,
        "per_language":  per_lang,
    }
