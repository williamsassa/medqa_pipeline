"""train_sft.py — QLoRA SFT on Llama-3-8B-Instruct.

Two modes:
  --dataset big   → MedQA-Darija-MultiLingual (100K, +Dorosz KG), out=SFT_BIG_OUT
  --dataset small → BrainHealthAI/MedQA_mutilangual (68K, no KG), out=SFT_SMALL_OUT

Adds early-stopping on eval_loss to avoid overfitting at 3 epochs.
Records final ‖∇L_SFT‖ to {OUT}/sft_meta.json — the SASR warm-up uses it as fallback.
"""
from __future__ import annotations
import argparse, os, json, time
from pathlib import Path
import torch, wandb
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                           EarlyStoppingCallback)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

import config as C
from data import load_full_dataset, format_for_sft


def _out_dir(which: str) -> Path:
    return C.SFT_BIG_OUT if which == "big" else C.SFT_SMALL_OUT


def _data_limit(which: str) -> int | None:
    return C.BIG_LIMIT if which == "big" else C.SMALL_LIMIT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["big", "small"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Override the per-dataset limit (smoke tests).")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = _out_dir(args.dataset)
    limit = args.limit if args.limit is not None else _data_limit(args.dataset)

    wandb.init(
        project=C.WANDB_PROJECT,
        name=f"sft-{args.dataset}-{int(time.time())}",
        tags=["sft", args.dataset],
        config={
            "method": "SFT",
            "dataset": args.dataset,
            "base_model": C.BASE_MODEL,
            "lora_r": C.LORA_R,
            "lora_alpha": C.LORA_ALPHA,
            "epochs": C.NUM_EPOCHS,
            "effective_batch": C.PER_DEVICE_BATCH * C.GRAD_ACCUM_STEPS,
            "lr": C.LEARNING_RATE_SFT,
            "max_seq_len": C.MAX_SEQ_LEN,
            "limit": limit,
        },
    )

    print(f"→ Loading tokenizer: {C.BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL, cache_dir=str(C.CACHE_DIR))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("→ Loading base model in 4-bit NF4")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, quantization_config=bnb, device_map="auto",
        cache_dir=str(C.CACHE_DIR), torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"→ Loading + formatting '{args.dataset}' dataset (limit={limit})")
    ds = load_full_dataset(args.dataset, limit=limit)
    ds = format_for_sft(ds, tokenizer, which=args.dataset)
    print(f"  total samples: {len(ds):,}")
    print(f"  preview:\n{ds[0]['text'][:500]}\n---")

    # Hold-out 2% (capped at 1000) for eval / early-stopping
    eval_size = min(1000, max(50, len(ds) // 50))
    split = ds.train_test_split(test_size=eval_size, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  train={len(train_ds):,}, eval={len(eval_ds):,}")

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=C.NUM_EPOCHS,
        per_device_train_batch_size=C.PER_DEVICE_BATCH,
        per_device_eval_batch_size=C.PER_DEVICE_BATCH,
        gradient_accumulation_steps=C.GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=C.LEARNING_RATE_SFT,
        warmup_ratio=C.WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_grad_norm=C.MAX_GRAD_NORM,
        logging_steps=C.LOGGING_STEPS,
        save_steps=C.SAVE_STEPS,
        save_total_limit=C.SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=C.EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to="wandb",
        optim="paged_adamw_8bit",
        max_seq_length=C.MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model, args=sft_cfg,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=C.EARLY_STOP_PATIENCE)],
    )

    print(f"→ Starting SFT ({args.dataset})")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = (time.time() - t0) / 60
    print(f"  done in {elapsed:.1f} min")

    print(f"→ Saving best adapters to {out_dir}")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Probe gradient norm for SASR warm-up reference (only useful for big-data).
    # OOM-safe: small batch + short sequences + cache flushed beforehand.
    gnorm = 1.0
    if args.dataset == "big":
        torch.cuda.empty_cache()
        model.eval()
        sample_texts = [train_ds[i]["text"] for i in range(min(4, len(train_ds)))]
        batch = tokenizer(sample_texts, padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(model.device)
        batch["labels"] = batch["input_ids"].clone()
        model.zero_grad()
        try:
            loss = model(**batch).loss
            loss.backward()
            sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sq += p.grad.detach().norm(2).item() ** 2
            gnorm = sq ** 0.5
        except Exception as e:
            print(f"  [warn] grad-norm probe failed: {e}")
        model.zero_grad()

    meta = {
        "dataset":          args.dataset,
        "train_samples":    len(train_ds),
        "eval_samples":     len(eval_ds),
        "elapsed_min":      elapsed,
        "sft_final_grad_norm": gnorm,
        "best_eval_loss":   getattr(trainer.state, "best_metric", None),
    }
    (out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  meta: {meta}")
    wandb.finish()


if __name__ == "__main__":
    main()
