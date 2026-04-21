"""
bench_cce.py — Our fused CCE kernel patched into Unsloth for Qwen3.5-0.8B.

Run on Colab with a T4/A100 GPU. Measures training throughput and peak VRAM.

The monkey-patch replaces unsloth-zoo's `fused_linear_cross_entropy` with our
Triton CCE kernel so it's used in the compiled forward pass.
"""

import os, time, json

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import torch

# ── Monkey-patch BEFORE importing unsloth ──────────────────────────────────
# We replace the fused_linear_cross_entropy function that unsloth's compiler
# injects into the model forward.  This is the CCE path (line 1606 in
# compiler.py) that calls cut_cross_entropy's linear_cross_entropy.
# We also patch unsloth_fused_ce_loss (the fallback chunked path).

from qwen3_cce.cce_triton import FusedCrossEntropyLossTriton

def patched_fused_linear_cross_entropy(
    hidden_states,
    lm_weight,
    labels,
    num_items_in_batch=None,
    ignore_index=-100,
    reduction="mean",
    logit_softcapping=None,
    accuracy_threshold="auto",
):
    """Drop-in replacement using our Triton CCE kernel."""
    if logit_softcapping == 0:
        logit_softcapping = None

    # Flatten for our kernel (it expects [N, H] and [N] labels)
    if hidden_states.dim() == 3:
        B, S, H = hidden_states.shape
        hidden_states = hidden_states.reshape(B * S, H)
        labels = labels.reshape(B * S)

    # Shift labels (causal LM): hidden[:-1] predicts labels[1:]
    hidden_states = hidden_states[:-1].contiguous()
    labels = labels[1:].contiguous()

    if num_items_in_batch is not None and torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(hidden_states.device, non_blocking=True)

    loss = FusedCrossEntropyLossTriton.apply(
        hidden_states, lm_weight, labels, ignore_index,
    )

    # If num_items_in_batch is set, the caller expects sum/n_items averaging.
    # Our kernel already returns mean, so rescale.
    if num_items_in_batch is not None:
        mask = labels != ignore_index
        n_valid = mask.sum().clamp(min=1).float()
        # Convert from mean to sum/num_items_in_batch
        loss = loss * n_valid / num_items_in_batch

    return loss


# Also provide a patched version for unsloth_fused_ce_loss (the fallback path)
from qwen3_cce.cce_torch import FusedCrossEntropyLossTorch

def patched_unsloth_fused_ce_loss(
    trainer,
    hidden_states,
    lm_head_weight,
    lm_head_bias,
    labels,
    mask=None,
    n_items=None,
    scaling=None,
    target_gb=None,
    torch_compile=True,
    overwrite=False,
    **kwargs,
):
    """Drop-in replacement for unsloth_fused_ce_loss using our kernel."""
    device = lm_head_weight.device

    if hidden_states.dim() == 3:
        B, S, H = hidden_states.shape
        hidden_states = hidden_states.reshape(B * S, H)
        labels = labels.reshape(B * S)

    # Shift labels
    _labels = torch.empty_like(labels, device=device)
    _labels[:-1] = labels[1:]
    _labels[-1] = -100
    labels = _labels

    hidden_flat = hidden_states.to(device=device, dtype=lm_head_weight.dtype)

    # Use Triton if available, else PyTorch
    try:
        loss = FusedCrossEntropyLossTriton.apply(
            hidden_flat, lm_head_weight, labels, -100,
        )
    except Exception:
        loss = FusedCrossEntropyLossTorch.apply(
            hidden_flat, lm_head_weight, labels, lm_head_bias, -100, 4096,
        )

    # Handle n_items normalization
    if n_items is not None:
        mask_valid = labels != -100
        n_valid = mask_valid.sum().clamp(min=1).float()
        loss = loss * n_valid / n_items.to(device=device, dtype=torch.float32)

    if scaling is not None:
        loss = loss * scaling

    return loss


# ── Apply patches ──────────────────────────────────────────────────────────
import unsloth_zoo.loss_utils
import unsloth_zoo.compiler

unsloth_zoo.loss_utils.fused_linear_cross_entropy = patched_fused_linear_cross_entropy
unsloth_zoo.compiler.fused_linear_cross_entropy = patched_fused_linear_cross_entropy

# Patch the fused CE loss used in the compiler templates
import unsloth_zoo.fused_losses.cross_entropy_loss
unsloth_zoo.fused_losses.cross_entropy_loss.unsloth_fused_ce_loss = patched_unsloth_fused_ce_loss
unsloth_zoo.compiler.unsloth_fused_ce_loss = patched_unsloth_fused_ce_loss

print("[bench_cce] Patched unsloth with our CCE kernel")

# ── Now load unsloth and train ─────────────────────────────────────────────
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 1024
NUM_STEPS = 20
BATCH_SIZE = 2
GRAD_ACCUM = 4

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
    use_gradient_checkpointing="unsloth",
)

texts = [
    "The quick brown fox jumps over the lazy dog. " * 40
] * 200

dataset = Dataset.from_dict({"text": texts})

training_args = SFTConfig(
    output_dir="./bench_cce_out",
    max_steps=NUM_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-4,
    logging_steps=1,
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    fp16=not (torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False),
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

start = time.perf_counter()
result = trainer.train()
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
tokens_per_step = BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN
total_tokens = tokens_per_step * NUM_STEPS

report = {
    "mode": "cce_kernel",
    "model": MODEL_NAME,
    "num_steps": NUM_STEPS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "max_seq_len": MAX_SEQ_LEN,
    "wall_time_s": round(elapsed, 2),
    "tokens_per_sec": round(total_tokens / elapsed, 1),
    "peak_vram_mb": round(peak_mb, 1),
    "final_loss": round(result.training_loss, 4),
}

print("\n" + "=" * 60)
print("BENCHMARK RESULT (our CCE kernel)")
print("=" * 60)
for k, v in report.items():
    print(f"  {k:>20s}: {v}")
print("=" * 60)

with open("bench_cce_result.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved to bench_cce_result.json")
