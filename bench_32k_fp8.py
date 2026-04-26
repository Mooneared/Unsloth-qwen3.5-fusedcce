"""
bench_32k_fp8.py — Qwen3.5 SFT benchmark with FP8 + custom CCE.

- Replaces nn.Linear base layers in LoRA modules with te.Linear
- Wraps model.forward in te.fp8_autocast (FP8 matmuls)
- Monkey-patches unsloth's CE with our streaming chunked CCE kernel
  to avoid materializing the full (seqlen, vocab) logits tensor.

Requires: SM89+ GPU and transformer-engine.
"""

import os, time, json
from contextlib import nullcontext

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import torch
import torch.nn as nn

# ── Import unsloth first (sets UNSLOTH_IS_PRESENT for unsloth_zoo) ───────
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

# ── Patch unsloth's CE with our streaming CCE BEFORE model load ──────────
# Must run before FastModel.from_pretrained so unsloth's compiled cache
# picks up the patched callable.
from qwen3_cce.cce_triton import FusedCrossEntropyLossTriton
from qwen3_cce.cce_torch import FusedCrossEntropyLossTorch

_ce_call_count = [0]

def patched_fused_linear_cross_entropy(
    hidden_states, lm_weight, labels,
    num_items_in_batch=None, ignore_index=-100,
    reduction="mean", logit_softcapping=None, accuracy_threshold="auto",
):
    _ce_call_count[0] += 1
    if logit_softcapping == 0:
        logit_softcapping = None
    if hidden_states.dim() == 3:
        B, S, H = hidden_states.shape
        hidden_states = hidden_states.reshape(B * S, H)
        labels = labels.reshape(B * S)
    hidden_states = hidden_states[:-1].contiguous()
    labels = labels[1:].contiguous()
    if num_items_in_batch is not None and torch.is_tensor(num_items_in_batch):
        num_items_in_batch = num_items_in_batch.to(hidden_states.device, non_blocking=True)
    loss = FusedCrossEntropyLossTriton.apply(
        hidden_states, lm_weight, labels, ignore_index,
    )
    if num_items_in_batch is not None:
        mask = labels != ignore_index
        n_valid = mask.sum().clamp(min=1).float()
        loss = loss * n_valid / num_items_in_batch
    return loss

def patched_unsloth_fused_ce_loss(
    trainer, hidden_states, lm_head_weight, lm_head_bias, labels,
    mask=None, n_items=None, scaling=None, target_gb=None,
    torch_compile=True, overwrite=False, **kwargs,
):
    _ce_call_count[0] += 1
    device = lm_head_weight.device
    if hidden_states.dim() == 3:
        B, S, H = hidden_states.shape
        hidden_states = hidden_states.reshape(B * S, H)
        labels = labels.reshape(B * S)
    _labels = torch.empty_like(labels, device=device)
    _labels[:-1] = labels[1:]
    _labels[-1] = -100
    labels = _labels
    hidden_flat = hidden_states.to(device=device, dtype=lm_head_weight.dtype)
    try:
        loss = FusedCrossEntropyLossTriton.apply(
            hidden_flat, lm_head_weight, labels, -100,
        )
    except Exception:
        loss = FusedCrossEntropyLossTorch.apply(
            hidden_flat, lm_head_weight, labels, lm_head_bias, -100, 4096,
        )
    if n_items is not None:
        mask_valid = labels != -100
        n_valid = mask_valid.sum().clamp(min=1).float()
        loss = loss * n_valid / n_items.to(device=device, dtype=torch.float32)
    if scaling is not None:
        loss = loss * scaling
    return loss

import unsloth_zoo.loss_utils
import unsloth_zoo.compiler
import unsloth_zoo.fused_losses.cross_entropy_loss
unsloth_zoo.loss_utils.fused_linear_cross_entropy = patched_fused_linear_cross_entropy
unsloth_zoo.compiler.fused_linear_cross_entropy = patched_fused_linear_cross_entropy
unsloth_zoo.fused_losses.cross_entropy_loss.unsloth_fused_ce_loss = patched_unsloth_fused_ce_loss
unsloth_zoo.compiler.unsloth_fused_ce_loss = patched_unsloth_fused_ce_loss
print("[cce] Patched unsloth's fused_linear_cross_entropy + unsloth_fused_ce_loss")

def sweep_patch_ce_refs():
    """Replace CE references in every loaded module (handles compiled cache modules)."""
    import sys
    swept = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if getattr(mod, "unsloth_fused_ce_loss", None) is not None and \
           mod.unsloth_fused_ce_loss is not patched_unsloth_fused_ce_loss:
            mod.unsloth_fused_ce_loss = patched_unsloth_fused_ce_loss
            swept += 1
        if getattr(mod, "fused_linear_cross_entropy", None) is not None and \
           mod.fused_linear_cross_entropy is not patched_fused_linear_cross_entropy:
            mod.fused_linear_cross_entropy = patched_fused_linear_cross_entropy
            swept += 1
    print(f"[cce] Swept {swept} module-level CE references")
    return swept

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"
DATA_DIR = "./data_prepared"
MAX_SEQ_LEN = 77695
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 4

# ── Load model ────────────────────────────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    cache_dir=CACHE_DIR,
)

model = FastModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
    lora_dropout=0,
    use_gradient_checkpointing=False,
)

# Sweep CE refs again — unsloth's compiled cache module imported the original
# unsloth_fused_ce_loss at codegen time, binding a local reference we missed.
sweep_patch_ce_refs()

# ── Import TE AFTER model load (it patches transformers on import) ──────
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# ── Replace base nn.Linear in LoRA modules with te.Linear ────────────────
fp8_recipe = recipe.DelayedScaling(
    fp8_format=recipe.Format.HYBRID,   # E4M3 fwd, E5M2 bwd
    amax_history_len=16,
    amax_compute_algo="max",
)

replaced = 0
for name, module in model.named_modules():
    if hasattr(module, "base_layer") and isinstance(module.base_layer, nn.Linear):
        old = module.base_layer
        old_requires_grad = old.weight.requires_grad
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
        # Match original frozen state — base_layer should stay frozen for LoRA
        new_layer.weight.requires_grad = old_requires_grad
        if old.bias is not None:
            new_layer.bias.requires_grad = old_requires_grad
        module.base_layer = new_layer
        del old
        replaced += 1

torch.cuda.empty_cache()
print(f"[fp8] Replaced {replaced} base nn.Linear -> te.Linear")

# ── Wrap model.forward: pad seqlen to multiple of 16 + fp8_autocast ──────
# FP8 requires product of leading dims % 8 == 0 and last dim % 16 == 0.
# Packed sequences from SFTTrainer can be arbitrary length, so pad here.
FP8_PAD = 16
_orig_forward = model.forward

def fp8_forward(*args, **kwargs):
    input_ids = kwargs.get("input_ids")
    labels = kwargs.get("labels")
    attention_mask = kwargs.get("attention_mask")

    if input_ids is not None and input_ids.dim() == 2:
        seq = input_ids.shape[-1]
        pad_to = ((seq + FP8_PAD - 1) // FP8_PAD) * FP8_PAD
        pad = pad_to - seq
        if pad > 0:
            kwargs["input_ids"] = torch.nn.functional.pad(input_ids, (0, pad), value=0)
            if labels is not None:
                kwargs["labels"] = torch.nn.functional.pad(labels, (0, pad), value=-100)
            if attention_mask is not None:
                kwargs["attention_mask"] = torch.nn.functional.pad(attention_mask, (0, pad), value=0)

    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        return _orig_forward(*args, **kwargs)

model.forward = fp8_forward

# ── Pre-compile: dummy forward + backward (under fp8_autocast) ───────────
print("[bench] Compiling kernels (dummy forward+backward)...")
device = next(model.parameters()).device
vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
dummy_ids = torch.randint(0, vocab_size, (1, 128), device=device)
dummy_labels = dummy_ids.clone()

out = model(input_ids=dummy_ids, labels=dummy_labels)
out.loss.backward()

model.zero_grad(set_to_none=True)
del dummy_ids, dummy_labels, out
torch.cuda.empty_cache()
torch.cuda.synchronize()
print("[bench] Compilation done.")

# ── Dataset ───────────────────────────────────────────────────────────────
assert os.path.exists(DATA_DIR), (
    f"Data not found at {DATA_DIR}. Run prepare_data.py first."
)
dataset = load_from_disk(DATA_DIR)
print(f"[bench] Loaded {len(dataset)} examples from {DATA_DIR}")

# ── Trainer ───────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./bench_32k_fp8_out",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=True,
    seed=42,
    report_to="none",
    gradient_checkpointing=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

# ── Mask user/system/tool turns (only train on assistant responses) ───────
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# ── Timed run ─────────────────────────────────────────────────────────────
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
print(f"[bench] Starting training (FP8): {NUM_EPOCHS} epochs, bs={BATCH_SIZE}, seqlen={MAX_SEQ_LEN}")

start = time.perf_counter()
trainer.train()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

print(f"[cce] Patched CE was called {_ce_call_count[0]} times during training")

# ── Report ────────────────────────────────────────────────────────────────
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
gpu_name = torch.cuda.get_device_name(0)
num_steps = trainer.state.global_step
total_tokens = BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN * num_steps

report = {
    "mode": "fp8_te",
    "model": MODEL_NAME,
    "gpu": gpu_name,
    "num_epochs": NUM_EPOCHS,
    "num_steps": num_steps,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "max_seq_len": MAX_SEQ_LEN,
    "lora_r": 64,
    "gradient_checkpointing": False,
    "packing": True,
    "te_linear_replaced": replaced,
    "wall_time_s": round(elapsed, 2),
    "tokens_per_sec": round(total_tokens / elapsed, 1),
    "peak_vram_mb": round(peak_mb, 1),
}

print("\n" + "=" * 60)
print(f"BENCHMARK: Qwen3.5-0.8B @ 32K seqlen FP8 ({gpu_name})")
print("=" * 60)
for k, v in report.items():
    print(f"  {k:>25s}: {v}")
print("=" * 60)

with open("bench_32k_fp8_result.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved to bench_32k_fp8_result.json")
