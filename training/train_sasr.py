"""train_sasr.py — Step-wise Adaptive SFT + GRPO on top of SFT-Big checkpoint.

Implements Algorithm 1 from Chen et al. 2025 (arXiv:2505.13026), with a
**composite reward** designed for medical reasoning:

  reward = SASR_W_FORMAT * format_reward + SASR_W_CONTENT * content_reward

  format_reward  ∈ {0.0, 0.5, 1.0}
       1.0  if both <think>...</think> AND <answer>...</answer> are well-formed
       0.5  if only <answer>...</answer> is well-formed
       0.0  otherwise

  content_reward ∈ [0, 1+0.1] (clipped to 1.0)
       base       = ROUGE-L(extracted_<answer>, gold)
       +0.10      if the question is COMPLEX (>30 tokens) and the answer
                  contains medical-JSON cues (symptoms/diagnostic/references)
       +0.10      if the question is SIMPLE (<10 tokens) and the answer is
                  concise (<50 tokens) — rewards the model for not over-talking

This composite is computed per rollout, then group-normalized to advantages
(GRPO). KL is regularized to a frozen SFT-warm reference.

ONLY runs on the BIG dataset.
"""
from __future__ import annotations
import argparse, os, json, time, math, random, re
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from rouge_score import rouge_scorer

import config as C
from data import load_full_dataset, prepare_sasr_dataset


# ── Reward ──────────────────────────────────────────────────────────────────
# The reward lives in sasr_reward.py (semantic SB_CS content + format + anti-hallucination).
# It REPLACES the original ROUGE-L content reward, which adversarially punished the
# SFT-Big model's paraphrasing and caused the first SASR run to plateau. Configure it once
# in main() via sasr_reward.configure(...).
import sasr_reward
from sasr_reward import extract_answer, format_reward, composite_reward as reward_composite


# ── Loading ───────────────────────────────────────────────────────────────
def load_policy_and_ref(sft_path: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    print("→ Loading policy (trainable, LoRA on top of SFT-Big)")
    base = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, quantization_config=bnb, device_map="auto",
        cache_dir=str(C.CACHE_DIR), torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    policy = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
    policy.print_trainable_parameters()

    print("→ Loading reference (frozen SFT-warm)")
    ref_base = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, quantization_config=bnb, device_map="auto",
        cache_dir=str(C.CACHE_DIR), torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    ref = PeftModel.from_pretrained(ref_base, sft_path)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return policy, ref


def sft_step_loss(policy, tokenizer, batch_texts: list[str]) -> torch.Tensor:
    enc = tokenizer(batch_texts, padding=True, truncation=True,
                    max_length=C.MAX_SEQ_LEN, return_tensors="pt").to(policy.device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return policy(**enc, labels=labels).loss


def current_grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().norm(2).item() ** 2
    return sq ** 0.5


# ── GRPO step ─────────────────────────────────────────────────────────────
def _logprobs_on_completion(model, input_ids, attn_mask, completion_start: int):
    with torch.set_grad_enabled(model.training):
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :]
    tgt    = input_ids[:, 1:]
    logp   = F.log_softmax(logits.float(), dim=-1)
    gathered = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = attn_mask[:, 1:].float()
    if completion_start - 1 > 0:
        mask[:, :completion_start - 1] = 0.0
    return (gathered * mask).sum(dim=-1)  # [B]


def grpo_step_loss(policy, ref, tokenizer, prompt: str, gold: str
                  ) -> tuple[torch.Tensor, dict]:
    """One GRPO update. Implements paper Eq. 4 with Schulman K3 KL estimator.

    Maps to Algorithm 1 lines 16-21:
      L17  Generate {e_i}_i=1..G ~ π_θ(·|x)
      L18  {y_i} ← EXTRACT({e_i})                             # extract <answer>
      L19  Compute rewards {R(y_i)}                           # composite
      L20  Form groups G+, G- (here: z-score normalization)   # standard GRPO
      L21  OPTIMIZATION_STEP(L_GRPO)                          # paper Eq. 4
    """
    # ── L17: sample G completions from current policy ─────────────────────
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=C.MAX_SEQ_LEN - C.SASR_GEN_MAX_NEW).to(policy.device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    policy.eval()
    with torch.no_grad():
        gen = policy.generate(
            **prompt_enc, do_sample=True,
            temperature=C.SASR_GEN_TEMP, top_p=0.95,
            num_return_sequences=C.SASR_G_SAMPLES,
            max_new_tokens=C.SASR_GEN_MAX_NEW,
            pad_token_id=tokenizer.eos_token_id,
        )
    policy.train()

    # ── L18-L19: extract answers + composite reward ──────────────────────
    rewards: list[float] = []; fmt_rs: list[float] = []; cnt_rs: list[float] = []
    for i in range(gen.shape[0]):
        comp_ids = gen[i, prompt_len:]
        txt = tokenizer.decode(comp_ids, skip_special_tokens=True)
        r, parts = reward_composite(txt, gold, prompt)
        rewards.append(r); fmt_rs.append(parts["r_format"]); cnt_rs.append(parts["r_content"])
    rewards_t = torch.tensor(rewards, device=policy.device, dtype=torch.float32)

    # ── L20: group-relative advantages (z-score) ──────────────────────────
    # Paper Eq. 3 uses median-split over token-level Â; in practice every
    # public GRPO implementation (TRL GRPOTrainer, OpenRLHF, DeepSeek-Math)
    # uses sequence-level z-score below. Both yield same gradient-signal
    # sign; z-score has lower variance and is the de-facto reference.
    adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-6)

    # ── Build attention mask directly from generated tensor ───────────────
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    attn = (gen != pad_id).long()
    attn[:, :prompt_len] = 1   # prompt is always attended

    # ── L21: log-probs under current policy (with grad) and frozen ref ────
    logp_policy = _logprobs_on_completion(policy, gen, attn, prompt_len)   # [G], grad
    with torch.no_grad():
        logp_ref = _logprobs_on_completion(ref, gen, attn, prompt_len)     # [G], no grad
    logp_old = logp_policy.detach()       # single-step update → ratio≈1 here

    # PPO-style clipped surrogate (paper Eq. 4, first term)
    log_ratio = logp_policy - logp_old
    ratio = torch.exp(log_ratio)
    unclip  = ratio * adv
    clipped = torch.clamp(ratio, 1 - C.SASR_CLIP_EPS, 1 + C.SASR_CLIP_EPS) * adv
    surrogate = -torch.min(unclip, clipped).mean()

    # KL[πθ ‖ π_ref]: Schulman's K3 estimator — always ≥ 0, low variance,
    # matches DeepSeek-Math + OpenRLHF reference implementations.
    # Reference: http://joschu.net/blog/kl-approx.html
    log_ratio_ref = logp_policy - logp_ref
    kl = (torch.exp(-log_ratio_ref) - 1.0 + log_ratio_ref).mean()

    loss = surrogate + C.SASR_BETA * kl    # paper Eq. 4 with KL inside the loss

    stats = {
        "grpo/reward_mean":  float(rewards_t.mean().item()),
        "grpo/reward_std":   float(rewards_t.std().item()),
        "grpo/reward_max":   float(rewards_t.max().item()),
        "grpo/format_mean":  float(np.mean(fmt_rs)),
        "grpo/content_mean": float(np.mean(cnt_rs)),
        "grpo/kl_k3":        float(kl.item()),
        "grpo/surrogate":    float(surrogate.item()),
        "grpo/adv_std":      float(adv.std().item()),
    }
    return loss, stats


# ── Main loop ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sft-path", type=str, default=str(C.SFT_BIG_OUT))
    args = ap.parse_args()

    limit = args.limit if args.limit is not None else C.BIG_LIMIT

    wandb.init(
        project=C.WANDB_PROJECT,
        name=f"sasr-big-{int(time.time())}",
        tags=["sasr", "big"],
        config={
            "method": "SASR", "dataset": "big", "base_model": C.BASE_MODEL,
            "warmup_steps": C.SASR_WARMUP_STEPS, "gamma": C.SASR_GAMMA,
            "g_samples": C.SASR_G_SAMPLES, "beta_kl": C.SASR_BETA,
            "clip_eps": C.SASR_CLIP_EPS, "lr": C.LEARNING_RATE_SASR,
            "epochs": C.NUM_EPOCHS,
            "w_format": C.SASR_W_FORMAT, "w_content": C.SASR_W_CONTENT,
            "limit": limit,
        },
    )

    print(f"→ Loading tokenizer from {args.sft_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("→ Loading + preparing big SASR dataset")
    ds = load_full_dataset("big", limit=limit)
    ds = prepare_sasr_dataset(ds, tokenizer)
    print(f"  samples: {len(ds):,}")

    # Configure the reward once: semantic SB_CS content + anti-hallucination on Dorosz drugs.
    drug_vocab = sasr_reward.load_drug_vocab_from_kg(C.DOROSZ_KG)
    sasr_reward.configure(
        drug_vocab=drug_vocab,
        sent_model_name=C.SENT_MODEL,
        w_format=C.SASR_W_FORMAT,
        w_content=C.SASR_W_CONTENT,
        w_hallu=getattr(C, "SASR_W_HALLU", 0.30),
    )
    print(f"  reward: SB_CS content + anti-hallu over {len(drug_vocab)} Dorosz drugs "
          f"(W_format={C.SASR_W_FORMAT}, W_content={C.SASR_W_CONTENT})")

    policy, ref = load_policy_and_ref(args.sft_path)
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=C.LEARNING_RATE_SASR, weight_decay=0.0,
    )

    ds_iter_seq = ds.shuffle(seed=42)
    data_iter = iter(ds_iter_seq)

    def _next_rows(n: int):
        nonlocal data_iter
        out = []
        for _ in range(n):
            try:
                out.append(next(data_iter))
            except StopIteration:
                data_iter = iter(ds_iter_seq)
                out.append(next(data_iter))
        return out

    # ── Warm-up ───────────────────────────────────────────────────────────
    print(f"\n=== WARM-UP: {C.SASR_WARMUP_STEPS} pure SFT optimizer steps ===")
    policy.train()
    g_warmup = None; g_last_sft = None
    step = 0; micro = 0
    t0 = time.time()
    while step < C.SASR_WARMUP_STEPS:
        rows = _next_rows(C.PER_DEVICE_BATCH)
        loss = sft_step_loss(policy, tokenizer, [r["text"] for r in rows])
        (loss / C.GRAD_ACCUM_STEPS).backward()
        micro += 1
        if micro == C.GRAD_ACCUM_STEPS:
            g_last_sft = current_grad_norm(policy)
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], C.MAX_GRAD_NORM)
            optim.step(); optim.zero_grad()
            step += 1; micro = 0
            if step % C.LOGGING_STEPS == 0:
                wandb.log({"warmup/step": step, "warmup/loss": float(loss.item()),
                           "warmup/grad_norm": g_last_sft})
                print(f"  [warmup {step}/{C.SASR_WARMUP_STEPS}] "
                      f"loss={float(loss.item()):.4f}  ‖∇‖={g_last_sft:.3f}")
    g_warmup = g_last_sft
    print(f"  G_warmup = {g_warmup:.4f}")
    wandb.log({"sasr/G_warmup": g_warmup})

    # ── Adaptive ──────────────────────────────────────────────────────────
    if C.SASR_ADAPTIVE_STEPS_OVERRIDE is not None:
        total_steps = max(1, int(C.SASR_ADAPTIVE_STEPS_OVERRIDE))
    else:
        total_steps = max(1, (len(ds) * C.NUM_EPOCHS) // (C.PER_DEVICE_BATCH * C.GRAD_ACCUM_STEPS))
    print(f"\n=== ADAPTIVE: target {total_steps} optimizer steps ===")

    sft_n = grpo_n = 0
    for step_idx in range(total_steps):
        p_t = g_last_sft / (g_last_sft + C.SASR_GAMMA * g_warmup + 1e-8)
        alpha = random.random()
        is_sft = alpha < p_t

        if is_sft:
            for _ in range(C.GRAD_ACCUM_STEPS):
                rows = _next_rows(C.PER_DEVICE_BATCH)
                loss = sft_step_loss(policy, tokenizer, [r["text"] for r in rows])
                (loss / C.GRAD_ACCUM_STEPS).backward()
            g_last_sft = current_grad_norm(policy)
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], C.MAX_GRAD_NORM)
            optim.step(); optim.zero_grad()
            sft_n += 1
            log = {"sasr/phase": 0, "sasr/p_t": p_t, "sasr/alpha": alpha,
                   "sasr/sft_loss": float(loss.item()), "sasr/grad_norm": g_last_sft}
        else:
            tot_loss = 0.0; stats_acc: dict[str, float] = {}
            for _ in range(C.GRAD_ACCUM_STEPS):
                row = _next_rows(1)[0]
                loss, stats = grpo_step_loss(policy, ref, tokenizer,
                                             row["prompt_text"], row["gold"])
                (loss / C.GRAD_ACCUM_STEPS).backward()
                tot_loss += float(loss.item()) / C.GRAD_ACCUM_STEPS
                for k, v in stats.items():
                    stats_acc[k] = stats_acc.get(k, 0.0) + v / C.GRAD_ACCUM_STEPS
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], C.MAX_GRAD_NORM)
            optim.step(); optim.zero_grad()
            grpo_n += 1
            log = {"sasr/phase": 1, "sasr/p_t": p_t, "sasr/alpha": alpha,
                   "sasr/grpo_loss": tot_loss, **stats_acc}

        if (step_idx + 1) % C.LOGGING_STEPS == 0:
            pct_sft = 100 * sft_n / max(1, sft_n + grpo_n)
            print(f"  [step {step_idx+1:>5}/{total_steps}] {'SFT ' if is_sft else 'GRPO'}  "
                  f"p_t={p_t:.3f}  SFT:{sft_n} GRPO:{grpo_n} ({pct_sft:.0f}% SFT)")
            wandb.log(log | {"sasr/step": step_idx + 1,
                             "sasr/sft_count": sft_n,
                             "sasr/grpo_count": grpo_n})

        if (step_idx + 1) % (C.SAVE_STEPS * 4) == 0:
            ckpt = C.SASR_BIG_OUT / f"checkpoint-{step_idx+1}"
            policy.save_pretrained(str(ckpt))

    elapsed = (time.time() - t0) / 60
    print(f"\n=== DONE in {elapsed:.1f} min ===  SFT:{sft_n}  GRPO:{grpo_n}")

    print(f"→ Saving to {C.SASR_BIG_OUT}")
    policy.save_pretrained(str(C.SASR_BIG_OUT))
    tokenizer.save_pretrained(str(C.SASR_BIG_OUT))
    meta = {
        "G_warmup": g_warmup, "G_last_sft_final": g_last_sft,
        "sft_steps": sft_n, "grpo_steps": grpo_n, "elapsed_min": elapsed,
        "epochs_planned": C.NUM_EPOCHS, "data_size": len(ds),
    }
    (C.SASR_BIG_OUT / "sasr_meta.json").write_text(json.dumps(meta, indent=2))
    wandb.finish()


if __name__ == "__main__":
    main()
