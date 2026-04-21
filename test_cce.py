"""
Test suite for the Fused Linear Cross-Entropy (CCE) kernel.

Tests:
  1. Forward correctness vs naive PyTorch (random weights)
  2. Backward correctness vs naive PyTorch (random weights)
  3. Forward correctness with real Qwen3.5-0.8B weights
  4. ignore_index handling
  5. Numerical stability on extreme values
"""

import os
import sys
import math
import torch
import torch.nn.functional as F

# ── setup ──────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float32  # test in fp32 for exact comparison

from qwen3_cce import FusedLinearCrossEntropyLoss, FusedCrossEntropyLossTorch
from qwen3_cce.cce_torch import fused_cross_entropy_forward_torch


def _naive_ce(hidden_states, weight, labels, bias=None, ignore_index=-100):
    """Standard (non-fused) cross-entropy for reference."""
    logits = hidden_states.float() @ weight.float().T
    if bias is not None:
        logits = logits + bias.float().unsqueeze(0)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


# ── Test 1: Forward correctness (small) ───────────────────────────────────
def test_forward_small():
    print("Test 1: Forward correctness (small) ... ", end="", flush=True)
    torch.manual_seed(42)
    N, H, V = 32, 64, 256
    hidden = torch.randn(N, H, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(V, H, device=DEVICE, dtype=DTYPE)
    labels = torch.randint(0, V, (N,), device=DEVICE)
    labels[0] = -100  # one ignored position

    ref_loss = _naive_ce(hidden, weight, labels)
    fused_loss, _, _ = fused_cross_entropy_forward_torch(
        hidden, weight, labels, chunk_size=64,
    )

    err = (ref_loss - fused_loss).abs().item()
    assert err < 1e-5, f"Forward mismatch: {err}"
    print(f"PASS  (err={err:.2e})")


# ── Test 2: Backward correctness (small) ──────────────────────────────────
def test_backward_small():
    print("Test 2: Backward correctness (small) ... ", end="", flush=True)
    torch.manual_seed(42)
    N, H, V = 16, 32, 128
    hidden = torch.randn(N, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
    weight = torch.randn(V, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
    labels = torch.randint(0, V, (N,), device=DEVICE)

    # Reference gradients
    ref_loss = _naive_ce(hidden, weight, labels)
    ref_loss.backward()
    ref_grad_h = hidden.grad.clone()
    ref_grad_w = weight.grad.clone()

    hidden.grad = None
    weight.grad = None

    # Fused gradients
    fused_loss = FusedCrossEntropyLossTorch.apply(
        hidden, weight, labels, None, -100, 32,
    )
    fused_loss.backward()

    h_err = (hidden.grad - ref_grad_h).abs().max().item()
    w_err = (weight.grad - ref_grad_w).abs().max().item()
    assert h_err < 1e-4, f"grad_hidden mismatch: {h_err}"
    assert w_err < 1e-4, f"grad_weight mismatch: {w_err}"
    print(f"PASS  (grad_h_err={h_err:.2e}, grad_w_err={w_err:.2e})")


# ── Test 3: All-ignored labels ────────────────────────────────────────────
def test_all_ignored():
    print("Test 3: All-ignored labels ... ", end="", flush=True)
    torch.manual_seed(42)
    N, H, V = 8, 32, 64
    hidden = torch.randn(N, H, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(V, H, device=DEVICE, dtype=DTYPE)
    labels = torch.full((N,), -100, device=DEVICE, dtype=torch.long)

    loss, _, _ = fused_cross_entropy_forward_torch(hidden, weight, labels)
    assert loss.item() == 0.0, f"Expected 0 loss for all-ignored, got {loss.item()}"
    print("PASS")


# ── Test 4: Numerical stability ──────────────────────────────────────────
def test_numerical_stability():
    print("Test 4: Numerical stability (large logits) ... ", end="", flush=True)
    torch.manual_seed(42)
    N, H, V = 8, 32, 64
    # Create hidden states that produce very large logits
    hidden = torch.randn(N, H, device=DEVICE, dtype=DTYPE) * 100.0
    weight = torch.randn(V, H, device=DEVICE, dtype=DTYPE) * 100.0
    labels = torch.randint(0, V, (N,), device=DEVICE)

    ref_loss = _naive_ce(hidden, weight, labels)
    fused_loss, _, _ = fused_cross_entropy_forward_torch(hidden, weight, labels)

    # Both should be finite
    assert torch.isfinite(ref_loss), "Reference loss is not finite"
    assert torch.isfinite(fused_loss), "Fused loss is not finite"
    err = (ref_loss - fused_loss).abs().item()
    rel_err = err / (ref_loss.abs().item() + 1e-8)
    assert rel_err < 1e-4, f"Stability mismatch: rel_err={rel_err}"
    print(f"PASS  (rel_err={rel_err:.2e})")


# ── Test 5: FusedLinearCrossEntropyLoss module (3D input + shift) ─────────
def test_module_with_shift():
    print("Test 5: Module with shift_labels=True ... ", end="", flush=True)
    torch.manual_seed(42)
    B, S, H, V = 2, 16, 32, 64
    hidden = torch.randn(B, S, H, device=DEVICE, dtype=DTYPE, requires_grad=True)
    weight = torch.randn(V, H, device=DEVICE, dtype=DTYPE)
    labels = torch.randint(0, V, (B, S), device=DEVICE)

    loss_fn = FusedLinearCrossEntropyLoss(
        shift_labels=True, chunk_size=32, backend="torch",
    )
    fused_loss = loss_fn(hidden, weight, labels)

    # Reference: manual shift + naive CE
    flat_h = hidden.reshape(B * S, H)[:-1]
    flat_l = labels.reshape(B * S)[1:]
    ref_loss = _naive_ce(flat_h, weight, flat_l)

    err = (ref_loss - fused_loss).abs().item()
    assert err < 1e-5, f"Module mismatch: {err}"
    # Check backward runs without error
    fused_loss.backward()
    assert hidden.grad is not None
    print(f"PASS  (err={err:.2e})")


# ── Test 6: Qwen3.5-0.8B real weights ────────────────────────────────────
def test_qwen3_real_weights():
    print("Test 6: Qwen3.5-0.8B real weights ... ", end="", flush=True)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    except ImportError:
        print("SKIP (transformers not installed)")
        return

    # Only load what we need: config + embedding weights (tied to lm_head)
    try:
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen3.5-0.8B", cache_dir=CACHE_DIR, trust_remote_code=True,
        )
    except Exception as e:
        print(f"SKIP (cannot load config: {e})")
        return

    text_cfg = config.text_config if hasattr(config, "text_config") else config
    H = text_cfg.hidden_size       # 1024
    V = text_cfg.vocab_size        # 248320

    print(f"\n       H={H}, V={V}, downloading embedding weights...", flush=True)

    # Download just the safetensors index to find which shard has embed_tokens
    from huggingface_hub import hf_hub_download
    import json
    try:
        idx_path = hf_hub_download(
            "Qwen/Qwen3.5-0.8B", "model.safetensors.index.json", cache_dir=CACHE_DIR,
        )
        with open(idx_path) as f:
            idx = json.load(f)
        # Find the embed_tokens shard by scanning keys
        embed_file = None
        for k, v in idx["weight_map"].items():
            if "embed_tokens" in k and "visual" not in k and "mtp" not in k:
                embed_file = v
                break
    except Exception:
        # Single-shard model
        embed_file = "model.safetensors"

    shard_path = hf_hub_download(
        "Qwen/Qwen3.5-0.8B", embed_file, cache_dir=CACHE_DIR,
    )
    from safetensors.torch import load_file
    tensors = load_file(shard_path)

    # Find the embedding tensor (== lm_head weight due to tied embeddings)
    embed_key = None
    for k in tensors:
        if "embed_tokens" in k:
            embed_key = k
            break
    if embed_key is None:
        print("SKIP (embed_tokens not found in shard)")
        return

    lm_head_weight = tensors[embed_key].to(device=DEVICE, dtype=torch.float32)
    assert lm_head_weight.shape == (V, H), f"Unexpected shape: {lm_head_weight.shape}"
    print(f"       Loaded {embed_key}: {lm_head_weight.shape}", flush=True)

    # Synthetic hidden states + labels
    torch.manual_seed(123)
    N = 64
    hidden = torch.randn(N, H, device=DEVICE, dtype=torch.float32) * 0.02
    labels = torch.randint(0, V, (N,), device=DEVICE)
    labels[0] = -100

    # Reference
    ref_loss = _naive_ce(hidden, lm_head_weight, labels)

    # Fused (use small chunk_size to exercise the chunking logic heavily)
    fused_loss, _, _ = fused_cross_entropy_forward_torch(
        hidden, lm_head_weight, labels, chunk_size=2048,
    )

    err = (ref_loss - fused_loss).abs().item()
    rel_err = err / (ref_loss.abs().item() + 1e-8)
    print(f"       ref_loss={ref_loss.item():.6f}, fused_loss={fused_loss.item():.6f}")
    assert rel_err < 1e-4, f"Real-weight mismatch: rel_err={rel_err}"
    print(f"       PASS  (rel_err={rel_err:.2e})")

    # Also test backward with real weights
    hidden_rg = hidden.clone().requires_grad_(True)
    loss = FusedCrossEntropyLossTorch.apply(
        hidden_rg, lm_head_weight, labels, None, -100, 2048,
    )
    loss.backward()
    assert hidden_rg.grad is not None
    assert torch.isfinite(hidden_rg.grad).all()
    print("       Backward with real weights: PASS")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {DEVICE}\n")
    test_forward_small()
    test_backward_small()
    test_all_ignored()
    test_numerical_stability()
    test_module_with_shift()
    test_qwen3_real_weights()
    print("\nAll tests passed!")
