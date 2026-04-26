"""
Fused Cut Cross-Entropy (CCE) kernel for Qwen3.5.

Provides a memory-efficient cross-entropy loss that fuses the lm_head linear
projection with the loss computation, avoiding materialisation of the full
(batch*seq, vocab_size) logits tensor.

Two backends:
  - Triton kernel  (NVIDIA GPU, imported when triton is available)
  - PyTorch reference  (CPU / MPS / any device, always available)
"""

from .cce_torch import (
    fused_cross_entropy_forward_torch,
    fused_cross_entropy_backward_torch,
    FusedCrossEntropyLossTorch,
)

try:
    from .cce_triton import (
        fused_cross_entropy_forward_triton,
        fused_cross_entropy_backward_triton,
        FusedCrossEntropyLossTriton,
    )
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from .loss import FusedLinearCrossEntropyLoss
from .fused_swiglu import fused_swiglu_mlp, FusedSwiGLUFunction, fused_swiglu_activation

__all__ = [
    "FusedLinearCrossEntropyLoss",
    "FusedCrossEntropyLossTorch",
    "fused_swiglu_mlp",
    "fused_swiglu_activation",
    "FusedSwiGLUFunction",
    "TRITON_AVAILABLE",
]
