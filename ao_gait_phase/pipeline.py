"""
High-level driver: wire the bandpass, oscillator, event detector, and phase
corrector together into a sample-by-sample loop.

The function `estimate_gait_phase` is what most users will call. It accepts
a raw single-channel ferroelectret recording and returns the corrected gait
phase for every sample, plus the supporting diagnostics (oscillator phase,
estimated frequency, detected event times). Two consumption patterns are
supported:

  - Batch: pass a NumPy array and read the result. Internally still a strict
    causal sample loop; the array form is provided for convenience.
  - Online: instantiate `GaitPhaseEstimator` and call its `step` method
    once per incoming sample. This is the form that runs on a microcontroller.

The function deliberately exposes the four building blocks as constructor
arguments. Replacing any one of them (a different filter design, a different
event detector, a different phase corrector) does not require modifying the
loop itself.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .bandpass import BandpassConfig, CausalBandpass
from .correction import PhaseCorrectorConfig, PhaseErrorLearner
from .events import EventDetectorConfig, HeelStrikeDetector
from .oscillator import AdaptiveOscillator, OscillatorConfig


@dataclass
class GaitPhaseConfig:
    """Bundle the four sub-configurations so callers pass a single object."""

    fs: float
    bandpass: Optional[BandpassConfig] = None
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    detector: EventDetectorConfig = field(default_factory=EventDetectorConfig)
    corrector: PhaseCorrectorConfig = field(default_factory=PhaseCorrectorConfig)

    def __post_init__(self) -> None:
        # If the user did not supply a bandpass config, build the default at
        # the requested sample rate. Other configs are sample-rate independent.
        if self.bandpass is None:
            self.bandpass = BandpassConfig(fs=self.fs)


@dataclass
class GaitPhaseResult:
    """Outputs returned by `estimate_gait_phase`."""

    time: np.ndarray              # Sample timestamps in seconds.
    bandpass: np.ndarray          # Bandpass-filtered input signal.
    phase_raw: np.ndarray         # Oscillator fundamental phase, no correction.
    phase_corrected: np.ndarray   # Corrected gait phase after the Yan correction loop.
    frequency_hz: np.ndarray      # Oscillator-estimated stride frequency.
    event_times: np.ndarray       # Detected heel-strike timestamps.


class GaitPhaseEstimator:
    """Stateful per-sample estimator. Use this for true real-time inference.

    For batch processing of an entire recording, call `estimate_gait_phase`
    instead. The estimator and the batch function share the same underlying
    sample loop, so their results agree to numerical precision.
    """

    def __init__(self, config: GaitPhaseConfig):
        self.config = config
        self.bandpass = CausalBandpass(config.bandpass)
        self.oscillator = AdaptiveOscillator(config.oscillator)
        self.detector = HeelStrikeDetector(config.detector)
        self.corrector = PhaseErrorLearner(config.corrector)
        self._dt = 1.0 / config.fs

    @property
    def event_times(self):
        """Heel-strike timestamps detected since construction (or last reset)."""
        return list(self.detector.event_times)

    def reset(self) -> None:
        """Drop all internal state. Useful when starting a new recording."""
        self.bandpass.reset()
        self.oscillator = AdaptiveOscillator(self.config.oscillator)
        self.detector.reset()
        self.corrector = PhaseErrorLearner(self.config.corrector)

    def step(self, sample: float, time: float) -> float:
        """Process one raw sample and return the corrected gait phase.

        The loop order matches Yan et al.: bandpass, oscillator update,
        event detection on the bandpassed signal, phase-correction integration,
        and finally the corrected phase output.
        """
        # 1. Causal bandpass: remove DC drift and high-frequency noise.
        bp = self.bandpass.step(sample)

        # 2. Adaptive oscillator: update phi, omega, alpha from the bandpassed
        #    signal. Using the bandpassed signal (rather than the raw one)
        #    keeps the AO from spending capacity on DC tracking.
        self.oscillator.step(bp, self._dt)
        phi_n = self.oscillator.phase()
        omega = self.oscillator.state.omega
        stride_period = self.oscillator.stride_period()

        # 3. Event detection on the same bandpassed signal.
        event_time = self.detector.step(bp, time, stride_period)
        if event_time is not None:
            self.corrector.on_event(phi_n, event_time, omega)

        # 4. Phase-correction integration (runs every sample).
        self.corrector.step(omega, time, self._dt)

        # 5. Apply correction to the raw oscillator phase.
        return self.corrector.corrected_phase(phi_n)


def estimate_gait_phase(
    signal: np.ndarray,
    fs: float,
    time: Optional[np.ndarray] = None,
    config: Optional[GaitPhaseConfig] = None,
) -> GaitPhaseResult:
    """Batch-mode wrapper around `GaitPhaseEstimator`.

    Parameters
    ----------
    signal : np.ndarray
        Raw single-channel ferroelectret recording, in whatever units the
        sensor produces (the AO is scale-invariant after the bandpass).
    fs : float
        Sampling frequency in Hz.
    time : np.ndarray, optional
        Per-sample timestamps in seconds. If omitted, a uniform 1/fs grid
        is generated.
    config : GaitPhaseConfig, optional
        Override any of the four sub-configurations. The default reproduces
        the published evaluation setup.

    Returns
    -------
    GaitPhaseResult
        Diagnostics and the corrected gait phase per sample.
    """
    cfg = config or GaitPhaseConfig(fs=fs)
    if cfg.fs != fs:
        raise ValueError("config.fs must match fs.")

    signal = np.asarray(signal, dtype=float)
    n = signal.shape[0]
    if time is None:
        time = np.arange(n) / fs
    else:
        time = np.asarray(time, dtype=float)

    estimator = GaitPhaseEstimator(cfg)
    bandpass_out = np.empty(n)
    phase_raw = np.empty(n)
    phase_corr = np.empty(n)
    freq_hz = np.empty(n)

    for i in range(n):
        # Inline the step so the per-sample diagnostics can be recorded
        # without an extra method call per array element.
        bp = estimator.bandpass.step(signal[i])
        estimator.oscillator.step(bp, estimator._dt)
        phi_n = estimator.oscillator.phase()
        omega = estimator.oscillator.state.omega
        stride_period = estimator.oscillator.stride_period()

        event_time = estimator.detector.step(bp, time[i], stride_period)
        if event_time is not None:
            estimator.corrector.on_event(phi_n, event_time, omega)
        estimator.corrector.step(omega, time[i], estimator._dt)

        bandpass_out[i] = bp
        phase_raw[i] = phi_n
        phase_corr[i] = estimator.corrector.corrected_phase(phi_n)
        freq_hz[i] = estimator.oscillator.frequency()

    return GaitPhaseResult(
        time=time,
        bandpass=bandpass_out,
        phase_raw=phase_raw,
        phase_corrected=phase_corr,
        frequency_hz=freq_hz,
        event_times=np.asarray(estimator.detector.event_times),
    )
