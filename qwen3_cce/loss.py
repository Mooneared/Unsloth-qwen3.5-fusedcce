"""
High-level loss module that auto-selects Triton or PyTorch backend.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from . import TRITON_AVAILABLE
from .cce_torch import FusedCrossEntropyLossTorch


class FusedLinearCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for:

        logits = hidden_states @ lm_head.weight.T
        loss = F.cross_entropy(logits.view(-1, V), labels.view(-1))

    Fuses the matmul + cross-entropy so the [N, V] logits tensor is never
    materialised.  Saves ~V * N * 4 bytes of memory (e.g. 248K * 2048 * 4
    ≈ 2 GB per forward pass for a batch of 2048 tokens on Qwen3.5).

    Args:
        ignore_index: label value to ignore (default -100)
        shift_labels: if True, auto-shifts labels by 1 for causal LM
        chunk_size:   vocab chunk size for the PyTorch backend (Triton auto-tunes)
        backend:      "auto" (Triton if available, else PyTorch), "triton", or "torch"
    """

    def __init__(
        self,
        ignore_index: int = -100,
        shift_labels: bool = True,
        chunk_size: int = 4096,
        backend: str = "auto",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.shift_labels = shift_labels
        self.chunk_size = chunk_size

        if backend == "auto":
            self.use_triton = TRITON_AVAILABLE and torch.cuda.is_available()
        elif backend == "triton":
            if not TRITON_AVAILABLE:
                raise RuntimeError("Triton backend requested but triton is not installed")
            self.use_triton = True
        else:
            self.use_triton = False

    def forward(
        self,
        hidden_states: Tensor,  # [B, S, H] or [N, H]
        weight: Tensor,         # [V, H]  (lm_head.weight)
        labels: Tensor,         # [B, S] or [N]
        bias: Optional[Tensor] = None,
    ) -> Tensor:
        # Flatten to 2D
        if hidden_states.dim() == 3:
            B, S, H = hidden_states.shape
            hidden_states = hidden_states.reshape(B * S, H)
            labels = labels.reshape(B * S)

        # Optional causal-LM label shift
        if self.shift_labels:
            # hidden_states[:-1] predicts labels[1:]
            hidden_states = hidden_states[:-1].contiguous()
            labels = labels[1:].contiguous()

        if self.use_triton:
            from .cce_triton import FusedCrossEntropyLossTriton
            return FusedCrossEntropyLossTriton.apply(
                hidden_states, weight, labels, self.ignore_index,
            )
        else:
            return FusedCrossEntropyLossTorch.apply(
                hidden_states, weight, labels, bias,
                self.ignore_index, self.chunk_size,
            )
