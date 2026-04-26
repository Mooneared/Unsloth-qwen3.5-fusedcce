"""
train.py — Qwen3.5 LoRA SFT with unsloth.

Usage:
  # Single GPU
  python train.py

  # Multi-GPU DDP (recommended for LoRA — base model fits on each GPU)
  torchrun --nproc_per_node=2 train.py    # 2× RTX PRO 6000
  torchrun --nproc_per_node=4 train.py    # 4× RTX PRO 6000
  torchrun --nproc_per_node=8 train.py    # 8× H100/B200

DDP vs FSDP for LoRA:
  - DDP: each GPU has full base model. Only LoRA grads (~170 MB) sync per step.
    Near-linear scaling. Use this whenever base model fits on one GPU.
  - FSDP: shards base weights across GPUs. Adds all-gather comm per layer,
    which is *slower* on PCIe (no NVLink on RTX PRO 6000). Only worth it if
    the base model doesn't fit on one GPU (e.g., 27B on 24-48 GB GPUs).
"""

import os, json

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

import torch
from unsloth import FastModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

# ── Config ────────────────────────────────────────────────────────────────
# Model
MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_SEQ_LEN = 77695              # max(tokens) + 1 from prepare_data.py
LOAD_IN_4BIT = False            # True for QLoRA (halves weight VRAM)

# LoRA
LORA_R = 64                     # 64 for 9B/27B, 16 for 0.8B/4B
LORA_ALPHA = 64                 # typically = LORA_R
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Gradient checkpointing: False | True | "unsloth"
#   False   — no GC, fastest, needs VRAM headroom
#   True    — recompute activations, saves ~60% act VRAM, +33% compute
#   "unsloth" — CPU offload, saves same VRAM, speed depends on PCIe BW
GRAD_CKPT = True

# Training
DATA_DIR = "./data_prepared"        # output of prepare_data.py (has "text" column)
BATCH_SIZE = 1
GRAD_ACCUM = 4                  # effective batch = BATCH_SIZE * GRAD_ACCUM * num_gpus
                                # 2 GPUs DDP × bs=1 × accum=4 = effective 8
NUM_EPOCHS = 3                  # small datasets (≤500 traces) benefit from 3 epochs
LR = 2e-5                       # 2e-4 for 0.8B, 2e-5 for 9B, 1e-5 for 27B
WARMUP_RATIO = 0.1              # 10% for small datasets, 5% for larger
OUTPUT_DIR = "./output"

# ── Load model ────────────────────────────────────────────────────────────
model, tokenizer = FastModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=LOAD_IN_4BIT,
    cache_dir=CACHE_DIR,
)

model = FastModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=LORA_TARGETS,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    use_gradient_checkpointing=GRAD_CKPT,
)

# ── Dataset ───────────────────────────────────────────────────────────────
assert os.path.exists(DATA_DIR), (
    f"Data not found at {DATA_DIR}. Run prepare_data.py first."
)
dataset = load_from_disk(DATA_DIR)
assert "text" in dataset.column_names, (
    f"Dataset must have 'text' column, got: {dataset.column_names}"
)
print(f"Loaded {len(dataset)} examples from {DATA_DIR}")

# ── Trainer ───────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    logging_steps=1,
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    max_seq_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    packing=True,                # pack short sequences to fill seqlen
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

# ── Train ─────────────────────────────────────────────────────────────────
trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
