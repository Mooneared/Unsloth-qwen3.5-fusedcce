"""
train.py — Qwen3.5 LoRA SFT with unsloth.

Usage:
  # Single GPU
  python train.py

  # Multi-GPU (DDP)
  torchrun --nproc_per_node=8 train.py

  # Multi-GPU (accelerate)
  accelerate launch --num_processes=8 train.py
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
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
MAX_SEQ_LEN = 65536              # set to longest seq in dataset; shorter seqs get packed
LOAD_IN_4BIT = False            # True for QLoRA (halves weight VRAM)

# LoRA
LORA_R = 16                     # 64 for 9B/27B
LORA_ALPHA = 16                 # typically = LORA_R
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
NUM_EPOCHS = 2
LR = 2e-4                       # 2e-4 for 0.8B, 2e-5 for 9B, 1e-5 for 27B
WARMUP_RATIO = 0.05
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
