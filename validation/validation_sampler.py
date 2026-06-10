#!/usr/bin/env python3
"""Stratified sampler for medical professional validation — 15 annotators.

Produces 16 CSV packets:
  - validation_pilot.csv (20 items, ALL 15 annotate this same set → kappa)
  - validation_<profile_id>.csv (75 items per profile, 15 profiles)

Stratification:
  - 50/50 real/synthetic per spec (for the synthetic side)
  - Real comes from data_trilingual_icliniq/<spec>.json
  - Synthetic comes from data_synthetic/<spec>.json
  - 25 items per spec × 3 specs per profile = 75 items

Kappa overlap by design:
  - The pilot (20 items) is annotated by everyone.
  - Some specs appear in 2 profiles (intentional cross-validation).

Each row has a unique pair_id so kappa_calculator can match annotations across pros.
"""
import csv
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

random.seed(42)

PROJECT_DIR = Path(__file__).parent
ICLINIQ_TRI_DIR = PROJECT_DIR / "data_trilingual_icliniq"
# data_trilingual/ holds synthetic with FR + Darija translations.
# (data_synthetic/ is EN-only — do NOT use for sampling.)
SYNTH_DIR = PROJECT_DIR / "data_trilingual"
OUT_DIR = PROJECT_DIR / "validation_packets"
OUT_DIR.mkdir(exist_ok=True)

# 15 profiles. Specs chosen by clinical adjacency. Overlaps create kappa pairs.
PROFILES = {
    "medecin_generaliste":   ["general_practitioner", "family_medicine", "general_medicine"],
    "cardiologue":            ["cardiology", "cardiothoracic_surgery", "hematology"],
    "pharma_etudiant_1":      ["infectious_diseases", "anesthesiology", "allergy_immunology"],
    "pharma_etudiant_2":      ["dentistry", "endodontics", "cosmetic_dermatology"],
    "etudiant_med_1":         ["child_health", "fetal_medicine", "pediatric_allergy"],
    "etudiant_med_2":         ["neurology", "mental_health", "geriatrics"],
    "etudiant_med_3":         ["endocrinology", "diabetes", "dietetics"],
    "etudiant_med_4":         ["gastroenterology", "bariatric_surgery", "general_surgery"],
    "etudiant_med_5":         ["sleep_medicine", "critical_care", "otorhinolaryngology"],
    "etudiant_med_6":         ["urology", "nephrology", "andrology"],
    "etudiant_med_7":         ["orthopedics", "spine_surgery", "vascular_surgery"],
    "etudiant_med_8":         ["ophthalmology", "audiology", "dermatology"],
    # Cross-overlap profiles (for kappa on overlap specs)
    "etudiant_med_9":         ["infectious_diseases", "critical_care", "microbiology"],
    "etudiant_med_10":        ["surgical_oncology", "pain_medicine", "hematology"],
    "etudiant_med_11":        ["gynecology", "infertility", "sexology"],
}
N_PER_SPEC = 25         # 75 items / pro = 15 min
PILOT_SIZE = 20         # common subset for kappa


def _short(s, n=400):
    if isinstance(s, dict):
        s = s.get("text") or s.get("translation") or ""
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip().replace("\n", " ").replace("\r", " ")
    return s[:n] + ("..." if len(s) > n else "")


def load_real_pairs(spec):
    """Load real iCliniq trilingual pairs. Drops rows missing FR or Darija."""
    f = ICLINIQ_TRI_DIR / f"{spec}.json"
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(d, dict) and "pairs" in d:
        d = d["pairs"]
    out = []
    for r in d:
        q_en = r.get("question_en") or ""
        a_en = r.get("answer_en") or ""
        q_fr = r.get("question_fr") or ""
        a_fr = r.get("answer_fr") or ""
        q_da = r.get("question_darija") or ""
        a_da = r.get("answer_darija") or ""
        if not (q_en and a_en and q_fr and a_fr and q_da and a_da):
            continue
        out.append({
            "spec": spec,
            "source": "real_icliniq",
            "pair_id": r.get("pair_id") or hashlib.md5(
                (q_en + spec).encode()).hexdigest()[:12],
            "q_en": q_en, "a_en": a_en,
            "q_fr": q_fr, "a_fr": a_fr,
            "q_da": q_da, "a_da": a_da,
        })
    return out


def load_synth_pairs(spec):
    """Load synthetic trilingual pairs. Filters out rows missing FR or Darija."""
    f = SYNTH_DIR / f"{spec}.json"
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(d, dict) and "pairs" in d:
        d = d["pairs"]
    out = []
    for r in d:
        q_en = r.get("question_en") or r.get("question") or ""
        a_en = r.get("answer_en") or r.get("answer") or ""
        q_fr = r.get("question_fr", "") or ""
        a_fr = r.get("answer_fr", "") or ""
        q_da = r.get("question_darija", "") or ""
        a_da = r.get("answer_darija", "") or ""
        if not q_en or not a_en:
            continue
        # Require trilingual coverage for validation packets
        if not (q_fr and a_fr and q_da and a_da):
            continue
        # Only keep synthetic rows (skip iCliniq/openc rows that share data_trilingual/)
        src_raw = str(r.get("source", "")).lower()
        if "icliniq" in src_raw:
            continue
        out.append({
            "spec": spec,
            "source": f"synthetic_{r.get('source','llm').replace('synthetic_', '')}",
            "pair_id": r.get("pair_id") or hashlib.md5((q_en + spec).encode()).hexdigest()[:12],
            "q_en": q_en,
            "a_en": a_en,
            "q_fr": q_fr,
            "a_fr": a_fr,
            "q_da": q_da,
            "a_da": a_da,
        })
    return out


def stratified_sample(spec, n):
    real = load_real_pairs(spec)
    synth = load_synth_pairs(spec)
    n_real = min(n // 2, len(real))
    n_synth = n - n_real
    if n_synth > len(synth):
        n_synth = len(synth)
        n_real = min(n - n_synth, len(real))
    sample = random.sample(real, n_real) + random.sample(synth, n_synth)
    random.shuffle(sample)
    return sample


def write_csv(path, items):
    cols = ["pair_id", "specialty", "source",
            "question_en", "answer_en",
            "question_fr", "answer_fr",
            "question_darija", "answer_darija",
            "C1_plausibility", "C2_coherence",
            "C3_safety", "C4_language",
            "comment"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for it in items:
            w.writerow([
                it["pair_id"], it["spec"], it["source"],
                _short(it["q_en"]), _short(it["a_en"], 800),
                _short(it["q_fr"]), _short(it["a_fr"], 800),
                _short(it["q_da"]), _short(it["a_da"], 800),
                "", "", "", "", "",
            ])


def main():
    # Pilot: spread across 5 representative specs
    pilot = []
    for spec in ["cardiology", "general_practitioner", "infectious_diseases",
                 "dentistry", "neurology"]:
        pilot.extend(stratified_sample(spec, 4))
    pilot = pilot[:PILOT_SIZE]
    write_csv(OUT_DIR / "validation_pilot.csv", pilot)
    print(f"  pilot: {len(pilot)} items -> validation_pilot.csv")

    print(f"\n  {len(PROFILES)} profile packets:")
    all_items = []
    for profile, specs in PROFILES.items():
        items = []
        for spec in specs:
            s = stratified_sample(spec, N_PER_SPEC)
            items.extend(s)
            real_n = sum(1 for x in s if x["source"].startswith("real"))
            print(f"    {profile:25s} / {spec:30s} {len(s)}  (real={real_n})")
        write_csv(OUT_DIR / f"validation_{profile}.csv", items)
        all_items.extend(items)
        print(f"    {profile:25s} TOTAL: {len(items)} -> validation_{profile}.csv\n")

    # Index for kappa calc
    idx = {it["pair_id"]: {"spec": it["spec"], "source": it["source"]}
           for it in pilot + all_items}
    (OUT_DIR / "_pair_index.json").write_text(
        json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Total items distributed: pilot=20, profiles=15×75={15*75}")
    print(f"  Specs covered: {len(set(s for sp in PROFILES.values() for s in sp))}")
    print(f"  Index: {len(idx):,} pair_ids -> _pair_index.json")
    print(f"\n  All packets in: {OUT_DIR}")


if __name__ == "__main__":
    main()
