"""
prepare_data.py — Apply chat template to produce plain text for training.

Input:  {"messages": [{"role": "user", "content": "..."}, ...]}
Output: {"text": "<formatted conversation>"}

Uses the model's chat template by default, or a custom Jinja template.

Usage:
  python prepare_data.py
"""

import os, json
import numpy as np

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DATASET = "./neko-data.jsonl"    # HF name or local path (json/jsonl/parquet/csv)
CUSTOM_TEMPLATE = "./qwen35_chat_template.jinja"  # path to .jinja file, or None to use model's template
OUTPUT_DIR = "./data_prepared"
HF_REPO = None                   # e.g. "Mooneared/qwen35-sft-data" to push to HF Hub
NUM_PROC = os.cpu_count()

# ── Load tokenizer (for chat template + length estimation) ────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

if CUSTOM_TEMPLATE:
    with open(CUSTOM_TEMPLATE) as f:
        tokenizer.chat_template = f.read()
    print(f"[data] Using custom template: {CUSTOM_TEMPLATE}")
else:
    assert tokenizer.chat_template is not None, (
        f"{MODEL_NAME} has no chat template. Provide CUSTOM_TEMPLATE."
    )
    print(f"[data] Using model chat template")

# ── Load dataset ──────────────────────────────────────────────────────────
print(f"[data] Loading {DATASET}...")
dataset = load_dataset("json", data_files=DATASET, split="train")
print(f"[data] {len(dataset)} examples loaded.")

assert "messages" in dataset.column_names, (
    f"Dataset must have 'messages' column, got: {dataset.column_names}"
)

# ── Apply chat template ──────────────────────────────────────────────────
def normalize_messages(msgs):
    """Parse JSON-encoded messages and tool_call arguments to fit the template."""
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, str):
                try:
                    tc["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    tc["function"]["arguments"] = {"raw": args}
    return msgs

def apply_template(batch):
    texts = [
        tokenizer.apply_chat_template(
            normalize_messages(msgs), tokenize=False, add_generation_prompt=False
        )
        for msgs in batch["messages"]
    ]
    return {"text": texts}

print(f"[data] Applying chat template (num_proc={NUM_PROC})...")
dataset = dataset.map(
    apply_template,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=dataset.column_names,
    desc="Formatting",
)

# ── Stats (full tokenization for accurate max) ───────────────────────────
char_lengths = np.array([len(t) for t in dataset["text"]])
print(f"\n[data] Tokenizing all {len(dataset)} examples for accurate length stats...")
tok_lengths = np.array([len(tokenizer.encode(t, add_special_tokens=False)) for t in dataset["text"]])

print(f"\n[data] Stats ({len(dataset)} examples):")
print(f"  chars  — min: {char_lengths.min():,}  median: {int(np.median(char_lengths)):,}  max: {char_lengths.max():,}")
print(f"  tokens — min: {tok_lengths.min():,}  median: {int(np.median(tok_lengths)):,}  max: {tok_lengths.max():,}  (full tokenization)")

for boundary in [1024, 4096, 8192, 16384, 32768, 65536]:
    count = (tok_lengths <= boundary).sum()
    print(f"  <= {boundary:>6,} tok: {count:>6,} ({100*count/len(tok_lengths):.1f}%)")

# ── Preview ───────────────────────────────────────────────────────────────
print(f"\n[data] Preview (first example, truncated to 500 chars):")
print("-" * 60)
print(dataset["text"][0][:500])
print("-" * 60)

# ── Save ──────────────────────────────────────────────────────────────────
dataset.save_to_disk(OUTPUT_DIR)
print(f"\n[data] Saved to {OUTPUT_DIR}")

if HF_REPO:
    print(f"[data] Pushing to {HF_REPO}...")
    dataset.push_to_hub(HF_REPO, private=True)
    print(f"[data] Uploaded to https://huggingface.co/datasets/{HF_REPO}")
