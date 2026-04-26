"""
upload_data.py — Push ./data_prepared to a private HF dataset repo.

Usage:
  huggingface-cli login  # need write-scoped token
  python upload_data.py
"""

import os

CACHE_DIR = os.path.join(os.getcwd(), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR

from datasets import load_from_disk

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR = "./data_prepared"
HF_REPO = "NaruseShiroha/neko-data-prepared"
PRIVATE = True

# ── Upload ────────────────────────────────────────────────────────────────
ds = load_from_disk(DATA_DIR)
print(f"Uploading {len(ds)} examples to {HF_REPO} (private={PRIVATE})...")
ds.push_to_hub(HF_REPO, private=PRIVATE)
print(f"Done. View at https://huggingface.co/datasets/{HF_REPO}")
