"""
Causal bandpass filter for ferroelectret force-myography signals.

The raw sensor signal carries a slowly varying DC component from the elastic
preload of the garment, the stride-frequency content (1-3 Hz fundamental plus
a few harmonics), and high-frequency electronic noise. The adaptive oscillator
in oscillator.py expects a zero-mean input that is clean enough to expose the
periodic structure of the gait cycle.

Design choices
--------------
1. Causal (one-pass) filtering. The pipeline is meant for real-time use on
   embedded hardware, so zero-phase methods such as scipy.signal.sosfiltfilt
   that look ahead in time are unavailable. We use scipy.signal.sosfilt, which
   processes samples in order.

2. First-order high-pass at 0.5 Hz. Walking cadence rarely drops below 0.7 Hz,
   so 0.5 Hz removes the DC offset and slow drifts with a small group delay
   in the gait band (~50 ms at 1 Hz). A higher-order high-pass would distort
   the fundamental.

3. Fourth-order low-pass at 15 Hz. Gait harmonics carry useful structure up
   to ~10 Hz; 15 Hz keeps the first six harmonics of a 2.5 Hz fundamental
   intact while rejecting the broadband sensor noise above. Fourth order is
   the smallest that gives sufficient stop-band attenuation.

4. Zero-initial-condition transient handling. A naive sosfilt() output starts
   from zero and converges to the steady-state value over ~5 time constants,
   which corrupts the first 2-3 seconds. scipy.signal.sosfilt_zi computes the
   filter state that would produce a constant output equal to the first
   sample, so the filter begins as if it had been running on that value
   forever. The output is clean from sample one.

Why this matters for the AO
---------------------------
The adaptive oscillator's amplitude-learning equation is sensitive to slow
drifts: a non-zero DC component is absorbed into the alpha_0 term but only
slowly, leading to phase tracking errors during the first few strides if the
high-pass is omitted or starts from zero. The zi-initialized causal bandpass
gives the AO a clean, zero-mean input that lets it lock to the stride
frequency within the first cycle.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal


@dataclass
class BandpassConfig:
    """Cutoff frequencies and filter orders for the causal bandpass.

    The defaults match the values used in the published evaluation:
    1st-order high-pass at 0.5 Hz, 4th-order low-pass at 15 Hz.
    """

    fs: float                # Sampling frequency in Hz.
    f_low: float = 0.5       # High-pass cutoff in Hz; removes DC and drift.
    f_high: float = 15.0     # Low-pass cutoff in Hz; preserves first ~6 harmonics.
    order_hp: int = 1        # First-order high-pass keeps the gait fundamental.
    order_lp: int = 4        # Fourth-order low-pass attenuates >30 dB above 30 Hz.


class CausalBandpass:
    """Sample-by-sample bandpass that initializes its state from the first sample.

    Use this class either online (call `step(sample)` per sample) or offline
    (call `apply(signal_array)` for a full vector). Both routes produce the
    same output because both use the same causal sosfilt.

    Example
    -------
        bp = CausalBandpass(BandpassConfig(fs=330.0))
        # Online:
        for raw_sample in stream:
            filtered_sample = bp.step(raw_sample)
        # Offline (equivalent):
        filtered = bp.apply(raw_signal_array)
    """

    def __init__(self, config: BandpassConfig):
        self.config = config
        nyq = 0.5 * config.fs

        # Second-order-section (SOS) form is numerically more stable than
        # transfer-function (b, a) form for higher-order filters and stacks
        # cleanly when the high-pass and low-pass are applied in sequence.
        self._sos_hp = signal.butter(
            config.order_hp, config.f_low / nyq, btype="high", output="sos"
        )
        self._sos_lp = signal.butter(
            config.order_lp, config.f_high / nyq, btype="low", output="sos"
        )

        # Steady-state initial conditions. sosfilt_zi returns the per-section
        # state for unit DC input; multiply by the actual first-sample value
        # to make the filter believe it has been processing that constant
        # forever. This removes the startup transient entirely.
        self._zi_hp = None
        self._zi_lp = None
        self._initialized = False

    def reset(self) -> None:
        """Forget the filter state. The next `step` call re-initializes from its input."""
        self._zi_hp = None
        self._zi_lp = None
        self._initialized = False

    def step(self, sample: float) -> float:
        """Filter one sample and return the bandpass output.

        On the first call the internal state is set so that a constant
        input equal to `sample` would have produced the same constant as
        output. This eliminates the convergence transient.
        """
        if not self._initialized:
            self._zi_hp = signal.sosfilt_zi(self._sos_hp) * sample
            # The high-pass output for a constant input is zero, so the
            # low-pass starts from zero as well.
            self._zi_lp = signal.sosfilt_zi(self._sos_lp) * 0.0
            self._initialized = True

        x = np.asarray([sample], dtype=float)
        hp_out, self._zi_hp = signal.sosfilt(self._sos_hp, x, zi=self._zi_hp)
        bp_out, self._zi_lp = signal.sosfilt(self._sos_lp, hp_out, zi=self._zi_lp)
        return float(bp_out[0])

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Filter an entire array in one call (still causal).

        Equivalent to calling `step` once per element after a reset. Useful
        for offline processing of a full recording.
        """
        self.reset()
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x

        zi_hp = signal.sosfilt_zi(self._sos_hp) * x[0]
        hp_out, _ = signal.sosfilt(self._sos_hp, x, zi=zi_hp)
        # The high-pass output starts at zero (constant in - constant out = 0),
        # so the low-pass naturally starts from zero state.
        zi_lp = signal.sosfilt_zi(self._sos_lp) * 0.0
        bp_out, _ = signal.sosfilt(self._sos_lp, hp_out, zi=zi_lp)
        self._initialized = True
        return bp_out


def design_bandpass(
    fs: float,
    f_low: float = 0.5,
    f_high: float = 15.0,
    order_hp: int = 1,
    order_lp: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function: return the (sos_hp, sos_lp) section pair.

    Useful when integrating into an existing real-time framework that already
    has its own sample loop and just needs the filter coefficients.
    """
    nyq = 0.5 * fs
    sos_hp = signal.butter(order_hp, f_low / nyq, btype="high", output="sos")
    sos_lp = signal.butter(order_lp, f_high / nyq, btype="low", output="sos")
    return sos_hp, sos_lp
