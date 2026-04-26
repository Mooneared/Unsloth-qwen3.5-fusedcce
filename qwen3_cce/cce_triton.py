"""
Triton CCE kernel for Qwen3.5 (and any LLM with a linear lm_head).

Fuses  loss = cross_entropy(hidden_states @ lm_head_weight.T, labels)
into a single kernel pass that tiles over the vocabulary dimension,
computing an online log-sum-exp so the full logits tensor is never
materialised.

Requires: Linux + NVIDIA GPU with Triton installed.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_ce_fwd_kernel(
    # Pointers
    hidden_ptr,       # [N, H]  input hidden states
    weight_ptr,       # [V, H]  lm_head weight
    labels_ptr,       # [N]     target token ids
    loss_ptr,         # [N]     per-token loss (output)
    lse_ptr,          # [N]     log-sum-exp (output, for backward)
    # Dimensions
    N: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    # Block sizes
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    # Constants
    ignore_index: tl.constexpr,
):
    """Each program instance handles one row (one token position)."""
    row = tl.program_id(0)
    if row >= N:
        return

    label = tl.load(labels_ptr + row)

    # Load this row's hidden state [H] in tiles
    # Accumulate: logit_true, running_max, running_sum for online LSE
    logit_true = 0.0
    running_max = -float('inf')
    running_sum = 0.0

    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V

        # Compute dot products for this vocab chunk: hidden[row] @ weight[v_start:v_end].T
        # Result: chunk_logits [BLOCK_V]
        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
        for h_start in range(0, H, BLOCK_H):
            h_offs = h_start + tl.arange(0, BLOCK_H)
            h_mask = h_offs < H

            # hidden[row, h_start:h_end]
            h_vals = tl.load(
                hidden_ptr + row * H + h_offs,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)

            # weight[v_start:v_end, h_start:h_end]  shape [BLOCK_V, BLOCK_H]
            w_vals = tl.load(
                weight_ptr + v_offs[:, None] * H + h_offs[None, :],
                mask=v_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            acc += tl.sum(w_vals * h_vals[None, :], axis=1)  # [BLOCK_V]

        chunk_logits = acc  # [BLOCK_V]

        # Extract logit for the correct token if it's in this chunk
        is_target = (v_offs == label) & v_mask
        logit_true += tl.sum(tl.where(is_target, chunk_logits, 0.0))

        # Online LSE update: only consider valid vocab positions
        chunk_logits = tl.where(v_mask, chunk_logits, -float('inf'))
        chunk_max = tl.max(chunk_logits)
        new_max = tl.maximum(running_max, chunk_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + \
                      tl.sum(tl.exp(chunk_logits - new_max))
        running_max = new_max

    lse = running_max + tl.log(running_sum)

    # Per-token loss: -logit_true + lse
    is_valid = label != ignore_index
    token_loss = tl.where(is_valid, -logit_true + lse, 0.0)

    tl.store(loss_ptr + row, token_loss)
    tl.store(lse_ptr + row, lse)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------
@triton.jit
def _fused_ce_bwd_kernel(
    # Pointers
    hidden_ptr,       # [N, H]
    weight_ptr,       # [V, H]
    labels_ptr,       # [N]
    lse_ptr,          # [N]
    grad_hidden_ptr,  # [N, H]   output
    grad_weight_ptr,  # [V, H]   output (atomically accumulated)
    # Scalars
    scale,            # float: grad_output / n_valid
    # Dimensions
    N: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    # Block sizes
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
    # Constants
    ignore_index: tl.constexpr,
):
    """
    Each program handles one row.  For each vocab chunk, computes
    grad_logit = softmax - 1_{target}, then accumulates into grad_hidden
    and grad_weight.
    """
    row = tl.program_id(0)
    if row >= N:
        return

    label = tl.load(labels_ptr + row)
    is_valid = label != ignore_index
    row_lse = tl.load(lse_ptr + row)

    for h_start in range(0, H, BLOCK_H):
        h_offs = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H

        grad_h_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        h_vals = tl.load(
            hidden_ptr + row * H + h_offs,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)

        for v_start in range(0, V, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            v_mask = v_offs < V

            # Recompute chunk logits
            w_vals = tl.load(
                weight_ptr + v_offs[:, None] * H + h_offs[None, :],
                mask=v_mask[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # Full dot product for softmax: need all H, but we're tiling H
            # So we need a separate inner loop for the full logit.
            # Instead, we compute full logits in an inner loop over H.
            full_logits = tl.zeros((BLOCK_V,), dtype=tl.float32)
            for h2_start in range(0, H, BLOCK_H):
                h2_offs = h2_start + tl.arange(0, BLOCK_H)
                h2_mask = h2_offs < H

                h2_vals = tl.load(
                    hidden_ptr + row * H + h2_offs,
                    mask=h2_mask,
                    other=0.0,
                ).to(tl.float32)

                w2_vals = tl.load(
                    weight_ptr + v_offs[:, None] * H + h2_offs[None, :],
                    mask=v_mask[:, None] & h2_mask[None, :],
                    other=0.0,
                ).to(tl.float32)

                full_logits += tl.sum(w2_vals * h2_vals[None, :], axis=1)

            # Compute softmax probabilities
            probs = tl.exp(full_logits - row_lse)          # [BLOCK_V]
            is_target = (v_offs == label) & v_mask
            grad_logit = tl.where(is_target, probs - 1.0, probs)
            grad_logit = tl.where(v_mask, grad_logit, 0.0)
            grad_logit = tl.where(is_valid, grad_logit * scale, 0.0)

            # grad_hidden += grad_logit @ w_chunk
            grad_h_acc += tl.sum(grad_logit[:, None] * w_vals, axis=0)

            # grad_weight[v] += grad_logit[v] * hidden[row]
            grad_w_update = grad_logit[:, None] * h_vals[None, :]  # [BLOCK_V, BLOCK_H]
            tl.atomic_add(
                grad_weight_ptr + v_offs[:, None] * H + h_offs[None, :],
                grad_w_update,
                mask=v_mask[:, None] & h_mask[None, :],
            )

        # Store grad_hidden for this H-chunk
        tl.store(
            grad_hidden_ptr + row * H + h_offs,
            grad_h_acc,
            mask=h_mask,
        )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
def fused_cross_entropy_forward_triton(
    hidden_states: Tensor,     # [N, H]
    weight: Tensor,            # [V, H]
    labels: Tensor,            # [N]
    bias: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor]:
    """
    Triton forward.  Returns (loss_scalar, lse_per_token).
    bias is not yet supported in the Triton path.
    """
    assert bias is None, "Triton CCE kernel does not support bias yet"
    N, H = hidden_states.shape
    V = weight.shape[0]

    loss_per_token = torch.empty(N, device=hidden_states.device, dtype=torch.float32)
    lse = torch.empty(N, device=hidden_states.device, dtype=torch.float32)

    # Autotuning block sizes for Qwen3.5: H=1024, V=248320
    BLOCK_H = min(triton.next_power_of_2(H), 1024)
    BLOCK_V = 4096  # ~4K vocab per tile keeps SRAM usage reasonable

    grid = (N,)
    _fused_ce_fwd_kernel[grid](
        hidden_states, weight, labels, loss_per_token, lse,
        N=N, H=H, V=V,
        BLOCK_H=BLOCK_H, BLOCK_V=BLOCK_V,
        ignore_index=ignore_index,
    )

    mask = labels != ignore_index
    n_valid = mask.sum().clamp(min=1).float()
    loss = loss_per_token.sum() / n_valid
    return loss, lse


def fused_cross_entropy_backward_triton(
    grad_output: Tensor,
    hidden_states: Tensor,
    weight: Tensor,
    labels: Tensor,
    lse: Tensor,
    bias: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Triton backward.  Returns (grad_hidden, grad_weight, None)."""
    assert bias is None, "Triton CCE kernel does not support bias yet"
    N, H = hidden_states.shape
    V = weight.shape[0]

    mask = labels != ignore_index
    n_valid = mask.sum().clamp(min=1).float()
    scale = (grad_output / n_valid).item()

    grad_hidden = torch.zeros_like(hidden_states, dtype=torch.float32)
    grad_weight = torch.zeros(V, H, device=weight.device, dtype=torch.float32)

    BLOCK_H = min(triton.next_power_of_2(H), 1024)
    BLOCK_V = 4096

    grid = (N,)
    _fused_ce_bwd_kernel[grid](
        hidden_states, weight, labels, lse, grad_hidden, grad_weight,
        scale=scale,
        N=N, H=H, V=V,
        BLOCK_H=BLOCK_H, BLOCK_V=BLOCK_V,
        ignore_index=ignore_index,
    )

    return grad_hidden.to(hidden_states.dtype), grad_weight.to(weight.dtype), None


class FusedCrossEntropyLossTriton(torch.autograd.Function):
    """Autograd wrapper around the Triton forward/backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        weight: Tensor,
        labels: Tensor,
        ignore_index: int = -100,
    ) -> Tensor:
        loss, lse = fused_cross_entropy_forward_triton(
            hidden_states, weight, labels, ignore_index=ignore_index,
        )
        ctx.save_for_backward(hidden_states, weight, labels, lse)
        ctx.ignore_index = ignore_index
        ctx.weight_requires_grad = weight.requires_grad
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, labels, lse = ctx.saved_tensors

        if not ctx.weight_requires_grad:
            # Frozen lm_head (LoRA case): skip grad_weight entirely. Use
            # PyTorch's matmul path which is well-tuned, no atomic contention.
            grad_hidden = _ce_backward_grad_hidden_only(
                grad_output, hidden_states, weight, labels, lse, ctx.ignore_index,
            )
            return grad_hidden, None, None, None

        grad_hidden, grad_weight, _ = fused_cross_entropy_backward_triton(
            grad_output, hidden_states, weight, labels, lse,
            ignore_index=ctx.ignore_index,
        )
        return grad_hidden, grad_weight, None, None


def _ce_backward_grad_hidden_only(
    grad_output: Tensor,
    hidden_states: Tensor,    # [N, H]
    weight: Tensor,           # [V, H]
    labels: Tensor,           # [N]
    lse: Tensor,              # [N]
    ignore_index: int,
) -> Tensor:
    """
    Compute only grad_hidden when lm_head is frozen.

    Tiles over V in chunks of CHUNK rows. For each chunk:
        chunk_logits = hidden @ weight[v0:v1].T      # [N, CHUNK]
        chunk_probs  = exp(chunk_logits - lse[:, None])
        # subtract 1 for the target column if it falls in this chunk
        grad_hidden += chunk_probs @ weight[v0:v1] * scale
    Uses dense matmul (well-tuned, FP8/BF16 cuBLAS) instead of Triton atomics.
    """
    N, H = hidden_states.shape
    V = weight.shape[0]
    device = hidden_states.device
    w_dtype = weight.dtype

    mask = labels != ignore_index
    n_valid = mask.sum().clamp(min=1).float()
    scale = float((grad_output / n_valid).item())  # 1 sync, once

    # Larger chunks → fewer kernel launches, fewer Python iterations.
    # 32K * N * 4 bytes = 32K * 77K * 4 ≈ 10 GB FP32 — fine on 96 GB.
    CHUNK = 32768
    grad_hidden = torch.zeros_like(hidden_states, dtype=torch.float32)
    lse_col = lse.unsqueeze(1)                # [N, 1]
    mask_col = mask.unsqueeze(1).to(torch.float32)  # [N, 1] FP32 for mul_

    # Pre-compute target offsets relative to vocab (per-row, used every chunk)
    target_offsets = labels.clone()           # [N], will compare per chunk
    valid_neg_one = -mask.to(torch.float32)   # [N] (-1 for valid, 0 for ignored)

    for v0 in range(0, V, CHUNK):
        v1 = min(v0 + CHUNK, V)
        w_chunk = weight[v0:v1]               # [chunk, H] in BF16
        chunk_size = v1 - v0

        # BF16 matmul, FP32 accumulate via cuBLAS — tensor cores active
        logits = hidden_states @ w_chunk.T    # [N, chunk] BF16
        probs = torch.exp(logits.float() - lse_col)  # FP32 for stability

        # Vectorized: subtract valid_neg_one at column (label - v0) for rows
        # whose label falls in this chunk. No CPU sync.
        # Compute clamped target column; rows with out-of-range labels write
        # to an arbitrary in-range column but we mask the contribution to 0.
        in_chunk = (target_offsets >= v0) & (target_offsets < v1)  # [N] bool
        col = (target_offsets - v0).clamp(0, chunk_size - 1)        # [N] safe
        contrib = valid_neg_one * in_chunk.to(torch.float32)        # [N] in {-1, 0}
        probs.scatter_add_(1, col.unsqueeze(1), contrib.unsqueeze(1))

        # Zero out grad rows for ignored tokens
        probs.mul_(mask_col)

        # grad_hidden += (probs @ w_chunk) * scale  via fused addmm_
        grad_hidden.addmm_(probs.to(w_dtype), w_chunk, alpha=scale)

    return grad_hidden.to(hidden_states.dtype)
