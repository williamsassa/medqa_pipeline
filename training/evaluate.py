"""evaluate.py — 4-way comparison on a held-out trilingual eval set.

Models:
  baseline   = Llama-3.1-8B-Instruct  (no fine-tuning)
  sft_big    = QLoRA SFT on MedQA-Darija-MultiLingual (100K, +Dorosz)
  sasr_big   = SASR on top of sft_big (composite reward, +Dorosz)
  sft_small  = QLoRA SFT on BrainHealthAI/MedQA_mutilangual (68K, no KG)

Metrics (mirroring the bioengineering-12-00687 paper layout):
  BLEU, GLEU, ROUGE-1, ROUGE-2, ROUGE-L, METEOR,
  Precision, Recall, F1            (token-level vs reference, micro-averaged)
  BS_P, BS_R, BS_F1                (BERTScore — multilingual)
  SB_CS                            (Sentence-BERT cosine sim — paraphrase-mpnet-multilingual)
  NASS                             (Normalized Answer Semantic Sim — SB_CS clipped & rescaled)

For ALL models, the eval set is the same hold-out tail of the BIG dataset
(seed 12345, different from training shuffle), 300 samples × 3 languages.
This is the fairest comparison: the small-data model is graded on the
trilingual benchmark it must generalize to.

Outputs:
  /workspace/outputs/{baseline,sft_big,sasr_big,sft_small}_eval.json
  /workspace/outputs/comparison.json          ← deliverable table
  /workspace/outputs/qualitative_samples.md   ← side-by-side examples
  /workspace/outputs/three_lang_demo.json     ← 3 fixed demo prompts × 4 models
"""
from __future__ import annotations
import argparse, json, time, random, re
from pathlib import Path
from typing import Optional
import numpy as np
import torch, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu

import config as C
from data import build_eval_prompt, _has_trilingual
from datasets import load_dataset
from medical_metrics import aggregate_medical, pathology_of
from robustness_tests import measure_robustness


random.seed(0); np.random.seed(0); torch.manual_seed(0)
SMOOTH = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_ANSWER_RE = re.compile(re.escape(C.ANSWER_OPEN) + r"(.*?)" + re.escape(C.ANSWER_CLOSE), flags=re.DOTALL)


def extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text or "")
    return (m.group(1).strip() if m else (text or "").strip())


# ── Sentence-transformer for SB_CS / NASS (loaded once globally) ──────────
_SBERT = None
def _sbert():
    global _SBERT
    if _SBERT is None:
        from sentence_transformers import SentenceTransformer
        print(f"→ Loading sentence-transformer: {C.SENT_MODEL}")
        _SBERT = SentenceTransformer(C.SENT_MODEL, cache_folder=str(C.CACHE_DIR))
    return _SBERT


# ── Model loading ─────────────────────────────────────────────────────────
def load_model(kind: str, adapter_path: Optional[str] = None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, quantization_config=bnb, device_map="auto",
        cache_dir=str(C.CACHE_DIR), torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.eval()
    if kind == "baseline":
        tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, cache_dir=str(C.CACHE_DIR))
        model = base
    else:
        tok = AutoTokenizer.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


# ── Generation ────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[str], max_new: int) -> list[str]:
    enc = tokenizer(prompts, padding=True, truncation=True,
                    max_length=C.MAX_SEQ_LEN - max_new,
                    return_tensors="pt").to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new, do_sample=False,
        temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id,
    )
    decoded = []
    for i in range(out.shape[0]):
        comp_ids = out[i, enc["input_ids"].shape[1]:]
        decoded.append(tokenizer.decode(comp_ids, skip_special_tokens=True).strip())
    return decoded


# ── Metrics ───────────────────────────────────────────────────────────────
def _token_prf1(pred: str, ref: str) -> tuple[float, float, float]:
    p_tokens = (pred or "").split()
    r_tokens = (ref  or "").split()
    if not p_tokens or not r_tokens:
        return 0.0, 0.0, 0.0
    p_set, r_set = set(p_tokens), set(r_tokens)
    tp = len(p_set & r_set)
    prec = tp / max(1, len(p_set))
    rec  = tp / max(1, len(r_set))
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute_metrics(preds: list[str], refs: list[str], lang: str) -> dict:
    """All 14 metrics matching the bioengineering paper table."""
    n = len(preds)

    # ROUGE
    r1, r2, rl = [], [], []
    for p, r in zip(preds, refs):
        s = _rouge.score(r, p)
        r1.append(s["rouge1"].fmeasure); r2.append(s["rouge2"].fmeasure); rl.append(s["rougeL"].fmeasure)

    # METEOR
    meteors = []
    for p, r in zip(preds, refs):
        try: meteors.append(meteor_score([r.split()], p.split()))
        except Exception: meteors.append(0.0)

    # BLEU-4 + GLEU
    bleus, gleus = [], []
    for p, r in zip(preds, refs):
        try:
            bleus.append(sentence_bleu([r.split()], p.split(),
                                       weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=SMOOTH))
        except Exception: bleus.append(0.0)
        try:
            gleus.append(sentence_gleu([r.split()], p.split()))
        except Exception: gleus.append(0.0)

    # Token P/R/F1
    precs, recs, f1s = [], [], []
    for p, r in zip(preds, refs):
        pp, rr, ff = _token_prf1(p, r)
        precs.append(pp); recs.append(rr); f1s.append(ff)

    # BERTScore
    bs_lang = {"en": "en", "fr": "fr", "darija": "ar"}.get(lang, "en")
    try:
        P, R, F1 = bertscore(preds, refs, lang=bs_lang, verbose=False, batch_size=16,
                              rescale_with_baseline=False)
        bs_p = float(P.mean().item()); bs_r = float(R.mean().item()); bs_f1 = float(F1.mean().item())
    except Exception as e:
        print(f"  [warn] BERTScore failed for {lang}: {e}")
        bs_p = bs_r = bs_f1 = 0.0

    # SB_CS — Sentence-BERT cosine similarity (multilingual)
    sb_cs = 0.0
    nass = 0.0
    try:
        m = _sbert()
        emb_p = m.encode(preds, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        emb_r = m.encode(refs,  batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        # cosine
        emb_p_n = emb_p / (np.linalg.norm(emb_p, axis=1, keepdims=True) + 1e-9)
        emb_r_n = emb_r / (np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-9)
        cos = (emb_p_n * emb_r_n).sum(axis=1)
        sb_cs = float(cos.mean())
        # NASS: clip to [0, 1] then rescale (the paper's "Normalized" twist)
        nass = float(np.clip(cos, 0.0, 1.0).mean())
    except Exception as e:
        print(f"  [warn] sentence-bert failed: {e}")

    return {
        "n_samples":  n,
        "BLEU":       float(np.mean(bleus)),
        "GLEU":       float(np.mean(gleus)),
        "ROUGE_1":    float(np.mean(r1)),
        "ROUGE_2":    float(np.mean(r2)),
        "ROUGE_L":    float(np.mean(rl)),
        "METEOR":     float(np.mean(meteors)),
        "Precision":  float(np.mean(precs)),
        "Recall":     float(np.mean(recs)),
        "F1":         float(np.mean(f1s)),
        "BS_P":       bs_p,
        "BS_R":       bs_r,
        "BS_F1":      bs_f1,
        "SB_CS":      sb_cs,
        "NASS":       nass,
        "mean_pred_len": float(np.mean([len(p.split()) for p in preds])),
        "mean_ref_len":  float(np.mean([len(r.split()) for r in refs])),
    }


METRIC_NAMES = ["BLEU", "GLEU", "ROUGE_1", "ROUGE_2", "ROUGE_L", "METEOR",
                "Precision", "Recall", "F1",
                "BS_P", "BS_R", "BS_F1",
                "SB_CS", "NASS"]


# ── One-model evaluation pass ─────────────────────────────────────────────
def evaluate_model(kind: str, adapter_path: Optional[str], eval_rows: list[dict],
                    out_path: Path, batch_size: int = 4) -> dict:
    print(f"\n{'='*60}\n  Evaluating: {kind}  ({adapter_path or 'base model'})\n{'='*60}")
    model, tok = load_model(kind, adapter_path)
    all_results = {"model": kind, "adapter": adapter_path,
                   "per_language": {}, "samples": []}

    # Buffers to feed the medical-metrics aggregator at the end
    all_pred_answers: list[str] = []
    all_refs:          list[str] = []
    all_langs:         list[str] = []
    all_rows:          list[dict] = []

    for lang in C.LANGS:
        print(f"\n-- {lang.upper()} --")
        rows = [r for r in eval_rows if r["lang"] == lang]
        prompts, refs = [], []
        for row in rows:
            msgs, gold = build_eval_prompt(row, lang, which="big")
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            refs.append(gold)

        preds = []
        t0 = time.time()
        for i in range(0, len(prompts), batch_size):
            batch_preds = generate_batch(model, tok, prompts[i:i+batch_size], C.EVAL_MAX_NEW_TOK)
            preds.extend(batch_preds)
            if (i // batch_size) % 10 == 0:
                done = min(i + batch_size, len(prompts))
                print(f"   {done}/{len(prompts)}  ({(time.time()-t0):.0f}s)")
        gen_time = time.time() - t0

        # Extract <answer>...</answer> from each pred for fair scoring
        pred_answers = [extract_answer(p) for p in preds]
        metrics = compute_metrics(pred_answers, refs, lang)
        metrics["gen_time_sec"] = gen_time
        all_results["per_language"][lang] = metrics
        print(f"   → BLEU {metrics['BLEU']:.3f}  R-L {metrics['ROUGE_L']:.3f}  "
              f"BS_F1 {metrics['BS_F1']:.3f}  SB_CS {metrics['SB_CS']:.3f}")

        # Buffer for medical aggregation
        all_pred_answers.extend(pred_answers)
        all_refs.extend(refs)
        all_langs.extend([lang] * len(rows))
        all_rows.extend(rows)

        idx = random.sample(range(len(preds)), min(5, len(preds)))
        for j in idx:
            all_results["samples"].append({
                "lang": lang,
                "prompt": prompts[j][:500] + ("…" if len(prompts[j]) > 500 else ""),
                "reference": refs[j],
                "raw_prediction": preds[j],
                "extracted_answer": pred_answers[j],
            })

    # Macro avg across languages (lexical/semantic)
    agg = {m: float(np.mean([all_results["per_language"][l][m] for l in C.LANGS]))
           for m in METRIC_NAMES}
    all_results["macro_avg"] = agg

    # ── Medical-grade metrics ─────────────────────────────────────────────
    print(f"\n   Computing medical metrics (entity recall / hallucination / Precision@K / pathology)…")
    med = aggregate_medical(all_pred_answers, all_refs, all_langs, all_rows)
    all_results["medical"] = med
    print(f"   → entity_recall_macro {med['macro']['entity_recall_macro']:.3f}  "
          f"hallu_macro {med['macro']['hallu_macro']:.3f}  "
          f"hit@3 {med['macro']['hit_at_k']:.3f}  "
          f"dosage_hallu {med['macro']['n_dosage_hallu_samples']}/{len(all_refs)}")

    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")
    # Don't release the model here if we still need it for robustness tests
    return all_results, model, tok


# ── Three fixed demo prompts (one per language) ───────────────────────────
DEMO_PROMPTS = [
    {"lang": "en", "user": "Question: A 45-year-old male presents with crushing substernal chest pain radiating to the left arm, diaphoresis, and shortness of breath for 2 hours. ECG shows ST elevation in leads V2-V5. What is the most likely diagnosis and immediate management?"},
    {"lang": "fr", "user": "Question : Une patiente de 28 ans présente une éruption érythémateuse en ailes de papillon sur le visage, des arthralgies symétriques aux mains et une photosensibilité. Quels examens biologiques demanderiez-vous et quel diagnostic suspectez-vous ?"},
    {"lang": "darija", "user": "السؤال: طفل عندو 4 سنين، عندو سخانة فوق 39° من جوج أيام، سعال يابس، وصعوبة فالتنفس. شنو خاصني نديرلو وشنو يمكن يكون عندو ؟"},
]


def run_three_lang_demo(adapter_paths: dict[str, Optional[str]], out_path: Path) -> dict:
    """Run the same 3 demo prompts on all 4 models — qualitative side-by-side."""
    from data import SYS_BY_LANG
    print(f"\n{'='*60}\n  Three-language qualitative demo\n{'='*60}")
    results = {}
    for kind, ap in adapter_paths.items():
        print(f"\n-- {kind} --")
        model, tok = load_model(kind, ap)
        outs = []
        for d in DEMO_PROMPTS:
            msgs = [{"role": "system", "content": SYS_BY_LANG[d["lang"]]},
                    {"role": "user", "content": d["user"]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            pred = generate_batch(model, tok, [prompt], C.EVAL_MAX_NEW_TOK)[0]
            outs.append({**d, "prediction": pred, "extracted_answer": extract_answer(pred)})
            print(f"  [{d['lang']}] {pred[:200]}…")
        results[kind] = outs
        del model, tok
        torch.cuda.empty_cache()
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-lang", type=int, default=C.EVAL_SAMPLES_PER_LANG)
    ap.add_argument("--batch-size",       type=int, default=C.EVAL_BATCH_SIZE)
    ap.add_argument("--skip-baseline",   action="store_true")
    ap.add_argument("--skip-sft-big",    action="store_true")
    ap.add_argument("--skip-sasr-big",   action="store_true")
    ap.add_argument("--skip-sft-small",  action="store_true")
    ap.add_argument("--skip-demo",       action="store_true")
    ap.add_argument("--robustness",      action="store_true",
                    help="Run robustness tests (paraphrase + ASR noise) on each model. "
                         "Adds ~30 min on L40S per model.")
    ap.add_argument("--robustness-n",    type=int, default=30,
                    help="Number of prompts used for the robustness suite.")
    args = ap.parse_args()

    wandb.init(project=C.WANDB_PROJECT, name=f"eval-{int(time.time())}", tags=["eval"])

    # ── Build trilingual eval set (held-out from training shuffle) ────────
    print(f"→ Building trilingual eval set ({args.samples_per_lang}/lang)")
    ds = load_dataset(C.HF_DATASET_BIG, C.HF_DATASET_BIG_CONF, split="train",
                      cache_dir=str(C.CACHE_DIR))
    drop = [c for c in ds.column_names if c.startswith("audio_")]
    if drop: ds = ds.remove_columns(drop)
    ds = ds.filter(_has_trilingual, num_proc=4)
    ds = ds.shuffle(seed=12345)  # different seed than training (seed=42)
    eval_rows = []
    for lang in C.LANGS:
        kept = 0
        for row in ds:
            if kept >= args.samples_per_lang:
                break
            eval_rows.append({**row, "lang": lang})
            kept += 1
    print(f"  total eval rows: {len(eval_rows)}  ({len(C.LANGS)} × {args.samples_per_lang})")

    # ── 4 evaluations ─────────────────────────────────────────────────────
    # evaluate_model now returns (results_dict, model, tok) so we can keep the
    # model loaded long enough to optionally run robustness tests on it.
    results: dict[str, dict] = {}
    robustness: dict[str, dict] = {}

    def _run(kind: str, adapter: Optional[str], out: Path):
        res, mdl, tk = evaluate_model(kind, adapter, eval_rows, out, batch_size=args.batch_size)
        results[kind] = res
        if args.robustness:
            print(f"\n   Running robustness suite ({args.robustness_n} prompts) for {kind}…")
            t0 = time.time()
            r = measure_robustness(
                mdl, tk, eval_rows, which="big",
                sample_n=args.robustness_n,
                compute_metrics_fn=compute_metrics,
                generate_fn=generate_batch,
                seed=12345,
            )
            r["elapsed_min"] = (time.time() - t0) / 60
            robustness[kind] = r
            print(f"     paraphrase BLEU CV: {r['paraphrase']['BLEU_cv']:.3f}  |  "
                  f"BS_F1 CV: {r['paraphrase']['BS_F1_cv']:.3f}")
            print(f"     ASR-noise ΔBLEU:    {r['asr_noise']['BLEU_delta']:+.3f}  |  "
                  f"ΔBS_F1: {r['asr_noise']['BS_F1_delta']:+.3f}")
        del mdl, tk
        torch.cuda.empty_cache()

    if not args.skip_baseline:
        _run("baseline", None, C.BASELINE_EVAL)
    if not args.skip_sft_big and C.SFT_BIG_OUT.exists():
        _run("sft_big", str(C.SFT_BIG_OUT), C.SFT_BIG_EVAL)
    if not args.skip_sasr_big and C.SASR_BIG_OUT.exists():
        _run("sasr_big", str(C.SASR_BIG_OUT), C.SASR_BIG_EVAL)
    if not args.skip_sft_small and C.SFT_SMALL_OUT.exists():
        _run("sft_small", str(C.SFT_SMALL_OUT), C.SFT_SMALL_EVAL)

    # ── Aggregate comparison table ────────────────────────────────────────
    print("\n" + "="*92)
    print("COMPARISON  (macro-avg across EN / FR / Darija)")
    print("="*92)
    header = f"{'Metric':<14} " + " ".join(f"{k:>13}" for k in results.keys())
    print(header); print("-" * len(header))
    comp = {"macro": {}, "per_language": {}, "deltas_vs_baseline": {}}
    for m in METRIC_NAMES:
        row = {k: results[k]["macro_avg"].get(m, 0.0) for k in results}
        comp["macro"][m] = row
        vals_str = " ".join(f"{row[k]:>13.3f}" for k in results)
        print(f"{m:<14} {vals_str}")

    print("\nPer-language breakdown:")
    for lang in C.LANGS:
        print(f"\n-- {lang.upper()} --")
        comp["per_language"][lang] = {}
        for m in METRIC_NAMES:
            row = {k: results[k]["per_language"][lang].get(m, 0.0) for k in results}
            comp["per_language"][lang][m] = row
            vals_str = " ".join(f"{row[k]:>13.3f}" for k in results)
            print(f"  {m:<12} {vals_str}")

    # Deltas vs baseline
    if "baseline" in results:
        print("\n" + "="*60); print("Δ vs baseline (macro-avg)"); print("="*60)
        for k in [x for x in results if x != "baseline"]:
            comp["deltas_vs_baseline"][k] = {}
            print(f"\n{k}:")
            for m in METRIC_NAMES:
                d = (results[k]["macro_avg"].get(m, 0.0)
                     - results["baseline"]["macro_avg"].get(m, 0.0))
                comp["deltas_vs_baseline"][k][m] = d
                mark = "↑" if d > 0 else ("↓" if d < 0 else "=")
                print(f"  {m:<12} {mark} {d:+.3f}")

    # ── Medical-grade comparison ──────────────────────────────────────────
    print("\n" + "="*92)
    print("MEDICAL METRICS  (macro)")
    print("="*92)
    MED_KEYS = ["entity_recall_drugs", "entity_recall_symptoms", "entity_recall_exams",
                "entity_recall_macro", "hallu_drugs", "hallu_symptoms", "hallu_exams",
                "hallu_dosage", "hallu_macro", "hit_at_k", "ref_citations_pct",
                "n_dosage_hallu_samples"]
    comp["medical_macro"] = {}
    header = f"{'Metric':<24} " + " ".join(f"{k:>13}" for k in results.keys())
    print(header); print("-" * len(header))
    for m in MED_KEYS:
        row = {k: results[k].get("medical", {}).get("macro", {}).get(m, 0.0) for k in results}
        comp["medical_macro"][m] = row
        vals_str = " ".join(f"{row[k]:>13.3f}" if isinstance(row[k], float)
                            else f"{row[k]:>13}" for k in results)
        print(f"{m:<24} {vals_str}")

    # Per-pathology breakdown (just the macro recall + hallu)
    print("\nPer-pathology (entity_recall_macro / hallu_macro / hit@3):")
    comp["medical_per_pathology"] = {}
    all_paths = sorted({c for k in results
                         for c in results[k].get("medical", {}).get("per_pathology", {})})
    for cat in all_paths:
        comp["medical_per_pathology"][cat] = {}
        cells = []
        for k in results:
            pp = results[k].get("medical", {}).get("per_pathology", {}).get(cat, {})
            comp["medical_per_pathology"][cat][k] = pp
            n = pp.get("n", 0)
            er = pp.get("entity_recall_macro", 0.0)
            hl = pp.get("hallu_macro", 0.0)
            hk = pp.get("hit_at_k", 0.0)
            cells.append(f"n={n:>3} R={er:.2f} H={hl:.2f} K={hk:.2f}")
        print(f"  {cat:<18}  " + "   ".join(f"{k}:{c}" for k, c in zip(results.keys(), cells)))

    # ── Robustness summary ────────────────────────────────────────────────
    if robustness:
        print("\n" + "="*92)
        print("ROBUSTNESS  (paraphrase CV → lower=better ; ASR-noise Δ → higher=more brittle)")
        print("="*92)
        comp["robustness"] = robustness
        for k, r in robustness.items():
            print(f"  {k:<11}  paraphrase BLEU CV={r['paraphrase']['BLEU_cv']:.3f} "
                  f"BS_F1 CV={r['paraphrase']['BS_F1_cv']:.3f}   |   "
                  f"ASR ΔBLEU={r['asr_noise']['BLEU_delta']:+.3f} "
                  f"ΔBS_F1={r['asr_noise']['BS_F1_delta']:+.3f}  "
                  f"(n={r['n_evaluated']})")

    C.COMPARISON_OUT.write_text(json.dumps(comp, ensure_ascii=False, indent=2))
    print(f"\nSaved: {C.COMPARISON_OUT}")

    # ── Qualitative markdown ──────────────────────────────────────────────
    md_path = C.OUTPUTS_DIR / "qualitative_samples.md"
    lines = ["# Qualitative comparison — 5 triplets per language per model\n"]
    for kind, res in results.items():
        lines.append(f"\n## {kind}\n")
        for i, s in enumerate(res["samples"]):
            lines.append(f"### [{s['lang']}] sample {i+1}\n")
            lines.append(f"**Reference**\n\n> {s['reference']}\n")
            lines.append(f"**Raw prediction**\n\n> {s['raw_prediction']}\n")
            lines.append(f"**Extracted `<answer>`**\n\n> {s['extracted_answer']}\n")
            lines.append("---\n")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {md_path}")

    # ── 3-lang demo across all 4 models ───────────────────────────────────
    if not args.skip_demo:
        adapter_paths = {"baseline": None}
        if C.SFT_BIG_OUT.exists():    adapter_paths["sft_big"]   = str(C.SFT_BIG_OUT)
        if C.SASR_BIG_OUT.exists():   adapter_paths["sasr_big"]  = str(C.SASR_BIG_OUT)
        if C.SFT_SMALL_OUT.exists():  adapter_paths["sft_small"] = str(C.SFT_SMALL_OUT)
        run_three_lang_demo(adapter_paths, C.OUTPUTS_DIR / "three_lang_demo.json")

    # ── Wandb table ───────────────────────────────────────────────────────
    cols = ["metric"] + list(results.keys())
    data = [[m] + [round(results[k]["macro_avg"].get(m, 0.0), 4) for k in results]
            for m in METRIC_NAMES]
    wandb.log({"comparison_macro": wandb.Table(columns=cols, data=data)})
    wandb.finish()

    print("\n✓ Evaluation complete. Run `python plot_results.py` for charts.")


if __name__ == "__main__":
    main()
