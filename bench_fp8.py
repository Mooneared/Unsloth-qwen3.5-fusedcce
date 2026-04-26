"""
bench_fp8.py — BF16 vs FP8 training benchmark for Qwen3.5-0.8B.

Usage:
  python bench_fp8.py              # BF16 baseline
  python bench_fp8.py --fp8        # FP8 via TransformerEngine

Requires SM89+ GPU for FP8 (L4, H100, B200, B300, RTX PRO 6000).
A100 (SM80) does NOT support FP8.

Install for FP8:
  pip install transformer-engine[pytorch]
"""

import os, time, json
from contextlib import nullcontext

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import torch
import torch.nn as nn
from unsloth import FastModel

# ── Config ─────────────────────────────────────────────────────────────
USE_FP8 = False                  # <-- Toggle this: False=BF16, True=FP8
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 32768
NUM_STEPS = 20
BATCH_SIZE = 1
LR = 2e-4

# ── Load model + LoRA ──────────────────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    cache_dir=CACHE_DIR,
)

model = FastModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing=True,
)

# ── FP8 conversion ────────────────────────────────────────────────────
# Replace the frozen base nn.Linear inside each PEFT LoRA module with
# te.Linear. LoRA A/B adapters stay BF16 (small, precision matters).
# Under te.fp8_autocast, te.Linear runs matmuls in FP8 (E4M3 fwd,
# E5M2 bwd), everything else stays BF16.

fp8_ctx = nullcontext
fp8_kwargs = {}

if USE_FP8:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe

    fp8_recipe = recipe.DelayedScaling(
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
    )

    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and isinstance(module.base_layer, nn.Linear):
            old = module.base_layer
            new_layer = te.Linear(
                old.in_features,
                old.out_features,
                bias=old.bias is not None,
                params_dtype=torch.bfloat16,
            )
            with torch.no_grad():
                new_layer.weight.copy_(old.weight)
                if old.bias is not None:
                    new_layer.bias.copy_(old.bias)
            module.base_layer = new_layer
            del old
            replaced += 1

    torch.cuda.empty_cache()
    print(f"[fp8] Replaced {replaced} base nn.Linear -> te.Linear")

    fp8_ctx = te.fp8_autocast
    fp8_kwargs = {"enabled": True, "fp8_recipe": fp8_recipe}

# ── Tokenize a single training sequence ────────────────────────────────
text = "The quick brown fox jumps over the lazy dog. " * 3500
tokens = tokenizer(text, max_length=MAX_SEQ_LEN, truncation=True, return_tensors="pt")
input_ids = tokens["input_ids"].to(model.device)
labels = input_ids.clone()
print(f"[bench] Sequence: {input_ids.shape[1]} tokens")

# ── Pre-compile kernels ───────────────────────────────────────────────
print("[bench] Pre-compiling kernels...")
dummy = torch.randint(0, 100, (1, 128), device=model.device)
with fp8_ctx(**fp8_kwargs):
    out = model(input_ids=dummy, labels=dummy)
    out.loss.backward()
model.zero_grad(set_to_none=True)
del dummy, out
torch.cuda.empty_cache()
print("[bench] Compilation done.")

# ── Optimizer ──────────────────────────────────────────────────────────
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR)

# ── Training loop ──────────────────────────────────────────────────────
mode_str = "FP8" if USE_FP8 else "BF16"
print(f"[bench] {NUM_STEPS} steps ({mode_str}), bs={BATCH_SIZE}, seqlen={MAX_SEQ_LEN}")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()

for step in range(NUM_STEPS):
    optimizer.zero_grad(set_to_none=True)

    with fp8_ctx(**fp8_kwargs):
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss

    loss.backward()
    optimizer.step()

    if step == 0 or (step + 1) % 5 == 0:
        print(f"  step {step+1}/{NUM_STEPS}  loss={loss.item():.4f}")

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ── Report ─────────────────────────────────────────────────────────────
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
total_tokens = BATCH_SIZE * MAX_SEQ_LEN * NUM_STEPS
gpu_name = torch.cuda.get_device_name(0)

report = {
    "mode": mode_str,
    "model": MODEL_NAME,
    "gpu": gpu_name,
    "num_steps": NUM_STEPS,
    "batch_size": BATCH_SIZE,
    "max_seq_len": MAX_SEQ_LEN,
    "gradient_checkpointing": True,
    "wall_time_s": round(elapsed, 2),
    "tokens_per_sec": round(total_tokens / elapsed, 1),
    "sec_per_step": round(elapsed / NUM_STEPS, 3),
    "peak_vram_mb": round(peak_mb, 1),
}

print(f"\n{'='*60}")
print(f"BENCHMARK: Qwen3.5-0.8B @ 32K ({mode_str} on {gpu_name})")
print(f"{'='*60}")
for k, v in report.items():
    print(f"  {k:>25s}: {v}")
print(f"{'='*60}")

fname = f"bench_fp8_{mode_str.lower()}_result.json"
with open(fname, "w") as f:
    json.dump(report, f, indent=2)
print(f"Saved to {fname}")
