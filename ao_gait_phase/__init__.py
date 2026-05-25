"""Minimal real-time gait phase estimator built around an adaptive oscillator.

This package implements the single-sensor configuration described in:

    Seiler, J. et al. "Hybrid Adaptive Oscillator and Temporal Convolutional
    Network Estimator for Textile-Integrated Gait Phase Sensing." (2026).

The four building blocks live in their own modules so that each can be
replaced or studied in isolation:

    bandpass.py    - Causal bandpass filter with steady-state initialization.
    oscillator.py  - Yan / Righetti adaptive oscillator.
    events.py      - Online dominant-peak event detector.
    correction.py  - Event-triggered phase correction (Yan Eqs. 8-11).
    pipeline.py    - High-level driver that wires the four together.

For most users the entry point is `estimate_gait_phase` in `pipeline.py`.
"""

from .bandpass import BandpassConfig, CausalBandpass, design_bandpass
from .correction import PhaseCorrectorConfig, PhaseErrorLearner
from .events import EventDetectorConfig, HeelStrikeDetector
from .oscillator import (
    AdaptiveOscillator,
    OscillatorConfig,
    OscillatorState,
)
from .pipeline import (
    GaitPhaseConfig,
    GaitPhaseEstimator,
    GaitPhaseResult,
    estimate_gait_phase,
)

__all__ = [
    "BandpassConfig",
    "CausalBandpass",
    "design_bandpass",
    "PhaseCorrectorConfig",
    "PhaseErrorLearner",
    "EventDetectorConfig",
    "HeelStrikeDetector",
    "AdaptiveOscillator",
    "OscillatorConfig",
    "OscillatorState",
    "GaitPhaseConfig",
    "GaitPhaseEstimator",
    "GaitPhaseResult",
    "estimate_gait_phase",
]
