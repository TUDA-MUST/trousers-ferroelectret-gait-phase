"""TCN residual-correction extension for the adaptive oscillator gait phase estimator.

This package provides the trained multi-head TCN from the published paper
together with the inference logic that combines its outputs with the
adaptive oscillator's phase. The bundled `weights/tcn_final_7L48f.pt`
matches the configuration described in supplementary Section S4 (7 layers,
48 filters, 71 099 parameters).

For most users the entry point is `estimate_hybrid_phase` in `inference.py`.
"""

from .inference import (
    HybridConfig,
    blend_hybrid,
    build_windows,
    estimate_hybrid_phase,
    load_pretrained,
    run_tcn_inference,
)
from .model import CausalConv1d, MultiHeadTCN

__all__ = [
    "HybridConfig",
    "MultiHeadTCN",
    "CausalConv1d",
    "load_pretrained",
    "run_tcn_inference",
    "blend_hybrid",
    "build_windows",
    "estimate_hybrid_phase",
]
