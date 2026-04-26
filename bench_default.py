"""
bench_default.py — Baseline: Unsloth SFT with default loss on Qwen3.5-0.8B.

Run on Colab with a T4/L4/A100 GPU. Measures training throughput and peak VRAM.
Uses realistic SFT sequence lengths.
"""

import os, time, json

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import torch
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 8192
NUM_STEPS = 20
BATCH_SIZE = 2
GRAD_ACCUM = 1

# ── Load model ─────────────────────────────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    cache_dir=CACHE_DIR,
)

# ── Apply LoRA ─────────────────────────────────────────────────────────────
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

# ── Synthetic dataset ──────────────────────────────────────────────────────
texts = [
    "The quick brown fox jumps over the lazy dog. " * 800
] * 100

dataset = Dataset.from_dict({"text": texts})

# ── Trainer ────────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./bench_default_out",
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

# ── Warmup ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

# ── Train + measure ────────────────────────────────────────────────────────
start = time.perf_counter()
result = trainer.train()
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ── Report ─────────────────────────────────────────────────────────────────
peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
tokens_per_step = BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN
total_tokens = tokens_per_step * NUM_STEPS

report = {
    "mode": "default",
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
print("BENCHMARK RESULT (default loss)")
print("=" * 60)
for k, v in report.items():
    print(f"  {k:>20s}: {v}")
print("=" * 60)

with open("bench_default_result.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved to bench_default_result.json")
