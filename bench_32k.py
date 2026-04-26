"""
bench_32k.py — Qwen3.5 SFT benchmark, BF16 baseline.

Vanilla unsloth training, no custom kernels. Companion to bench_32k_fp8.py
for apples-to-apples FP8 vs BF16 comparison.
"""

import os, time, json

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-4B"
DATA_DIR = "./data_prepared"     # output of prepare_data.py
MAX_SEQ_LEN = 77695              # max(tokens) + 1 from prepare_data.py output
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

# ── Pre-compile: dummy forward + backward at target seqlen ───────────────
print("[bench] Compiling kernels (dummy forward+backward)...")
device = next(model.parameters()).device
vocab_size = getattr(model.config, "vocab_size", None) or model.config.text_config.vocab_size
dummy_ids = torch.randint(0, vocab_size, (1, 128), device=device)
dummy_labels = dummy_ids.clone()

with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
    output_dir="./bench_32k_out",
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
print(f"[bench] Starting training: {NUM_EPOCHS} epochs, bs={BATCH_SIZE}, seqlen={MAX_SEQ_LEN}")

start = time.perf_counter()
trainer.train()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

# ── Report ────────────────────────────────────────────────────────────────
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
gpu_name = torch.cuda.get_device_name(0)
num_steps = trainer.state.global_step
total_tokens = BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN * num_steps

report = {
    "mode": "vanilla_unsloth",
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
    "wall_time_s": round(elapsed, 2),
    "tokens_per_sec": round(total_tokens / elapsed, 1),
    "peak_vram_mb": round(peak_mb, 1),
}

print("\n" + "=" * 60)
print(f"BENCHMARK: Qwen3.5-0.8B @ 32K seqlen ({gpu_name})")
print("=" * 60)
for k, v in report.items():
    print(f"  {k:>25s}: {v}")
print("=" * 60)

with open("bench_32k_result.json", "w") as f:
    json.dump(report, f, indent=2)
print("Saved to bench_32k_result.json")
