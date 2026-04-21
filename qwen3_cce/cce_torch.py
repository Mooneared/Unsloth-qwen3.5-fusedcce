"""
Pure-PyTorch reference implementation of Fused Linear Cross-Entropy (CCE).

The key idea: instead of computing  logits = hidden @ W^T  (shape [N, V]) and
then cross_entropy(logits, labels), we tile over the vocabulary dimension and
maintain a running log-sum-exp so we never materialise the full logits tensor.

This file is the "ground-truth" reference; it works on any device (CPU, MPS,
CUDA) and is used for correctness testing of the Triton kernel.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


def _online_lse_update(
    running_max: Tensor,  # [N]
    running_sum: Tensor,  # [N]
    chunk_logits: Tensor,  # [N, chunk_V]
) -> Tuple[Tensor, Tensor]:
    """Numerically-stable online log-sum-exp accumulation."""
    chunk_max = chunk_logits.max(dim=-1).values          # [N]
    new_max = torch.maximum(running_max, chunk_max)      # [N]
    # Renormalise the running sum to the new max
    running_sum = running_sum * torch.exp(running_max - new_max) + \
                  torch.exp(chunk_logits - new_max.unsqueeze(-1)).sum(dim=-1)
    return new_max, running_sum


def fused_cross_entropy_forward_torch(
    hidden_states: Tensor,     # [N, H]  (flattened batch*seq)
    weight: Tensor,            # [V, H]
    labels: Tensor,            # [N]     (token ids, -100 = ignore)
    bias: Optional[Tensor] = None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Forward pass: computes per-token cross-entropy loss without materialising
    the full [N, V] logits matrix.

    Returns:
        loss:       scalar, mean over non-ignored tokens
        logit_true: [N]  logit of the correct token (needed for backward)
        lse:        [N]  log-sum-exp across vocab (needed for backward)
    """
    N, H = hidden_states.shape
    V = weight.shape[0]

    # Compute logit for the correct class directly: dot(h, w[label])
    # For ignored positions we pick index 0 (value unused).
    safe_labels = labels.clone()
    mask = labels != ignore_index
    safe_labels[~mask] = 0
    target_weight = weight[safe_labels]                   # [N, H]
    logit_true = (hidden_states.float() * target_weight.float()).sum(dim=-1)  # [N]
    if bias is not None:
        logit_true = logit_true + bias[safe_labels].float()

    # Online log-sum-exp over vocabulary chunks
    running_max = torch.full((N,), float('-inf'), device=hidden_states.device, dtype=torch.float32)
    running_sum = torch.zeros(N, device=hidden_states.device, dtype=torch.float32)

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        w_chunk = weight[v_start:v_end]                   # [chunk_V, H]
        # chunk logits: [N, chunk_V]
        chunk_logits = hidden_states.float() @ w_chunk.float().T
        if bias is not None:
            chunk_logits = chunk_logits + bias[v_start:v_end].float().unsqueeze(0)
        running_max, running_sum = _online_lse_update(running_max, running_sum, chunk_logits)

    lse = running_max + running_sum.log()                 # [N]

    # loss = -logit_true + lse, masked and averaged
    per_token_loss = -logit_true + lse                    # [N]
    per_token_loss[~mask] = 0.0
    n_valid = mask.sum().clamp(min=1)
    loss = per_token_loss.sum() / n_valid

    return loss, logit_true, lse


def fused_cross_entropy_backward_torch(
    grad_output: Tensor,       # scalar
    hidden_states: Tensor,     # [N, H]
    weight: Tensor,            # [V, H]
    labels: Tensor,            # [N]
    lse: Tensor,               # [N]
    bias: Optional[Tensor] = None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Backward pass: computes gradients w.r.t. hidden_states and weight.

    grad_logit[n, v] = softmax[n, v] - 1_{v == label[n]}
    grad_hidden[n] = sum_v( grad_logit[n, v] * W[v] )
    grad_W[v]      = sum_n( grad_logit[n, v] * hidden[n] )

    Done in vocabulary chunks to keep memory O(N * chunk_size).
    """
    N, H = hidden_states.shape
    V = weight.shape[0]
    mask = labels != ignore_index
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    n_valid = mask.sum().clamp(min=1).float()
    scale = grad_output / n_valid  # scalar

    grad_hidden = torch.zeros_like(hidden_states, dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    grad_bias = torch.zeros_like(bias, dtype=torch.float32) if bias is not None else None

    for v_start in range(0, V, chunk_size):
        v_end = min(v_start + chunk_size, V)
        w_chunk = weight[v_start:v_end]                   # [cV, H]
        chunk_logits = hidden_states.float() @ w_chunk.float().T  # [N, cV]
        if bias is not None:
            chunk_logits = chunk_logits + bias[v_start:v_end].float().unsqueeze(0)

        # softmax probability for this chunk (using pre-computed lse)
        chunk_probs = torch.exp(chunk_logits - lse.unsqueeze(-1))  # [N, cV]

        # Subtract 1 at the correct label position if it falls in this chunk
        for v_offset in range(v_end - v_start):
            v_idx = v_start + v_offset
            token_mask = (safe_labels == v_idx) & mask
            chunk_probs[token_mask, v_offset] -= 1.0

        # Zero out ignored positions
        chunk_probs[~mask] = 0.0

        # Scale
        chunk_probs = chunk_probs * scale

        # Accumulate gradients
        grad_hidden += chunk_probs @ w_chunk.float()              # [N, H]
        grad_weight[v_start:v_end] += chunk_probs.T @ hidden_states.float()  # [cV, H]
        if grad_bias is not None:
            grad_bias[v_start:v_end] += chunk_probs.sum(dim=0)

    return grad_hidden.to(hidden_states.dtype), grad_weight.to(weight.dtype), \
           grad_bias.to(bias.dtype) if grad_bias is not None else None


class FusedCrossEntropyLossTorch(torch.autograd.Function):
    """Autograd wrapper around the chunked torch forward/backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: Tensor,
        weight: Tensor,
        labels: Tensor,
        bias: Optional[Tensor] = None,
        ignore_index: int = -100,
        chunk_size: int = 4096,
    ) -> Tensor:
        loss, logit_true, lse = fused_cross_entropy_forward_torch(
            hidden_states, weight, labels, bias, ignore_index, chunk_size,
        )
        ctx.save_for_backward(hidden_states, weight, labels, lse)
        if bias is not None:
            ctx.bias = bias
        else:
            ctx.bias = None
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, labels, lse = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = fused_cross_entropy_backward_torch(
            grad_output, hidden_states, weight, labels, lse,
            ctx.bias, ctx.ignore_index, ctx.chunk_size,
        )
        return grad_hidden, grad_weight, None, grad_bias, None, None
