#!/usr/bin/env python3
"""Compute Cohen's kappa from validation packets returned by professionals.

Usage:
  - Place each pro's filled CSV in validation_packets/returned/
    Filenames expected:
      validation_pilot_<pro>.csv     (the 20-item shared pilot)
      validation_<profile>.csv       (their specialty packet)
  - Run: python kappa_calculator.py

Outputs:
  validation_packets/REPORT.md  (human-readable)
  validation_packets/results.json (machine-readable)
"""
import csv
import json
import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict

PROJECT_DIR = Path(__file__).parent
PACKETS = PROJECT_DIR / "validation_packets"
RETURNED = PACKETS / "returned"

PROS = ["generalist", "cardio", "pharma1", "pharma2"]
CRITERIA = ["C1_plausibility", "C2_coherence", "C3_safety", "C4_language"]


def cohen_kappa(annot_a, annot_b):
    """Binary Cohen's kappa: (p_o - p_e) / (1 - p_e).

    Inputs are aligned 0/1 lists.
    """
    n = len(annot_a)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(annot_a, annot_b) if x == y)
    p_o = agree / n
    p1_a = sum(annot_a) / n
    p1_b = sum(annot_b) / n
    p_e = p1_a * p1_b + (1 - p1_a) * (1 - p1_b)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


def parse_csv(path):
    """Returns dict pair_id -> {C1: int, C2: int, C3: int, C4: int, comment: str, spec: str, source: str}."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("pair_id") or "").strip()
            if not pid:
                continue
            try:
                rec = {}
                for c in CRITERIA:
                    # Match by prefix to tolerate column-rename
                    matches = [k for k in row if k.startswith(c)]
                    if matches:
                        v = row[matches[0]].strip()
                        rec[c] = int(v) if v in ("0", "1") else None
                    else:
                        rec[c] = None
                rec["comment"] = (row.get("comment") or "").strip()
                rec["spec"] = (row.get("specialty") or "").strip()
                rec["source"] = (row.get("source") or "").strip()
                out[pid] = rec
            except Exception as e:
                print(f"  WARN: skipping row {pid}: {e}")
    return out


def main():
    if not RETURNED.exists():
        print(f"ERROR: place returned CSVs in {RETURNED}/")
        sys.exit(1)

    # ---- Pilot kappa (4 pros × same 20 items) ----
    pilot_data = {}  # pro -> {pair_id: rec}
    for pro in PROS:
        path = RETURNED / f"validation_pilot_{pro}.csv"
        if path.exists():
            pilot_data[pro] = parse_csv(path)
            print(f"  pilot {pro}: {len(pilot_data[pro])} rows")
        else:
            print(f"  pilot {pro}: MISSING ({path.name})")

    common_pids = None
    for d in pilot_data.values():
        common_pids = set(d.keys()) if common_pids is None else common_pids & set(d.keys())
    common_pids = sorted(common_pids or [])
    print(f"\n  pilot common pair_ids: {len(common_pids)}")

    kappa_scores = {}
    for c in CRITERIA:
        kappa_scores[c] = {}
        for a, b in combinations(pilot_data.keys(), 2):
            xs = [pilot_data[a][p][c] for p in common_pids if pilot_data[a][p][c] is not None
                                                            and pilot_data[b][p][c] is not None]
            ys = [pilot_data[b][p][c] for p in common_pids if pilot_data[a][p][c] is not None
                                                            and pilot_data[b][p][c] is not None]
            if len(xs) >= 2:
                k = cohen_kappa(xs, ys)
                kappa_scores[c][f"{a}_vs_{b}"] = round(k, 3)

    # Mean kappa per criterion
    mean_kappa = {c: round(sum(v.values()) / len(v), 3) if v else None
                  for c, v in kappa_scores.items()}

    # ---- Per-spec rejection rates from full packets ----
    spec_stats = defaultdict(lambda: {"n": 0, "real_n": 0, "synth_n": 0,
                                       "C1_pass": 0, "C2_pass": 0, "C3_pass": 0,
                                       "rejected_pairs": []})
    for pro in PROS:
        path = RETURNED / f"validation_{pro}.csv"
        if not path.exists():
            print(f"  packet {pro}: MISSING ({path.name})")
            continue
        d = parse_csv(path)
        for pid, rec in d.items():
            if not rec["spec"]:
                continue
            s = spec_stats[rec["spec"]]
            s["n"] += 1
            if rec["source"].startswith("real"):
                s["real_n"] += 1
            else:
                s["synth_n"] += 1
            for c, key in [("C1_plausibility", "C1_pass"),
                          ("C2_coherence", "C2_pass"),
                          ("C3_safety", "C3_pass")]:
                if rec[c] == 1:
                    s[key] += 1
            if any(rec[c] == 0 for c in ("C1_plausibility", "C2_coherence", "C3_safety")):
                s["rejected_pairs"].append({"pair_id": pid, "comment": rec["comment"]})

    # ---- Build report ----
    report = []
    report.append("# Validation Report — MedQA Darija MultiLingual\n")
    report.append("## Cohen's kappa (inter-annotator agreement, pilot)\n")
    report.append(f"Pilot common pair_ids: **{len(common_pids)}**\n")
    report.append(f"\n| Criterion | Mean kappa | Per-pair scores |\n|---|---:|---|")
    for c in CRITERIA:
        scores = kappa_scores.get(c, {})
        details = ", ".join(f"{k}={v}" for k, v in scores.items())
        report.append(f"| **{c}** | **{mean_kappa[c]}** | {details} |")

    target = 0.65
    overall_mean = sum(v for v in mean_kappa.values() if v is not None) / sum(
        1 for v in mean_kappa.values() if v is not None)
    report.append(f"\n**Overall mean kappa: {round(overall_mean, 3)}** "
                  f"(target ≥ {target}: {'✅ PASS' if overall_mean >= target else '❌ FAIL — revise guide'})\n")

    report.append("\n## Per-specialty acceptance rates\n")
    report.append("| Specialty | n | Real | Synth | C1 pass | C2 pass | C3 pass | Reject% |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for spec in sorted(spec_stats):
        s = spec_stats[spec]
        if s["n"] == 0:
            continue
        c1 = s["C1_pass"] / s["n"] * 100
        c2 = s["C2_pass"] / s["n"] * 100
        c3 = s["C3_pass"] / s["n"] * 100
        rej = len(s["rejected_pairs"]) / s["n"] * 100
        report.append(f"| {spec} | {s['n']} | {s['real_n']} | {s['synth_n']} | "
                      f"{c1:.0f}% | {c2:.0f}% | {c3:.0f}% | {rej:.0f}% |")

    report.append("\n## Safety incidents (C3 = 0)\n")
    n_safety = 0
    for spec in sorted(spec_stats):
        s = spec_stats[spec]
        for rp in s["rejected_pairs"]:
            if rp["comment"]:
                n_safety += 1
                report.append(f"- **{spec}** | `{rp['pair_id']}` | {rp['comment']}")
    if n_safety == 0:
        report.append("None reported.")

    out_md = PACKETS / "REPORT.md"
    out_md.write_text("\n".join(report), encoding="utf-8")
    print(f"\n  REPORT.md written to {out_md}")

    out_json = PACKETS / "results.json"
    out_json.write_text(json.dumps({
        "kappa_per_criterion": mean_kappa,
        "kappa_pairwise": kappa_scores,
        "kappa_overall": round(overall_mean, 3),
        "kappa_target": target,
        "kappa_pass": overall_mean >= target,
        "per_specialty": {k: dict(v) for k, v in spec_stats.items()},
    }, indent=2, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, set) else str(o)),
        encoding="utf-8")
    print(f"  results.json written to {out_json}")


if __name__ == "__main__":
    main()
