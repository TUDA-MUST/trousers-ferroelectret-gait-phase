"""
Online event detection on the bandpass-filtered sensor signal.

The phase-correction step in correction.py expects discrete event timestamps,
typically heel-strike instants, where it can compare the adaptive oscillator's
internal phase against an expected reference. We obtain these timestamps by
detecting the dominant negative peaks of the bandpassed gastrocnemius signal,
which line up tightly with heel strike (median offset ~30 ms, circular
consistency R ~ 0.77 in the published evaluation).

Why negative peaks
------------------
The raw gastrocnemius FMG signal has a characteristic shape: a sharp negative
deflection at heel strike (peak loading), followed by a slower positive
swing during stance. After bandpassing, the negative peak is more compact in
time than the positive peak and shows lower stride-to-stride jitter. We pass
the negated signal to a generic maximum detector to keep the code symmetric
in case the wearer flips polarity (different sensor mounting, different
muscle).

Detector behaviour
------------------
A simple three-sample comparison (prev2 < prev1 > current) locates local
maxima. We then filter the candidates:

  1. Settling period. The causal bandpass needs a short startup window
     before its output is trustworthy. Peaks during the first 5 seconds are
     ignored.

  2. Amplitude threshold. The running mean of recently accepted peak heights
     defines a threshold at 50 percent of that mean. This rejects harmonic
     sub-peaks that appear at twice the stride frequency, while still
     accepting the dominant peak even if the wearer's amplitude varies.

  3. Refractory period. Two events cannot fire closer than half a stride
     period apart, where the stride period comes from the oscillator's
     current frequency estimate. This adapts to walking speed.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EventDetectorConfig:
    """Detector hyperparameters.

    Defaults match the bp_minima configuration used in the paper.
    """

    settling_s: float = 5.0             # Ignore peaks during this opening window.
    peak_ema_alpha: float = 0.1         # EMA weight for the running peak amplitude.
    peak_amplitude_fraction: float = 0.5  # Accept peaks above this fraction of the running mean.
    min_interval_s: float = 0.3         # Absolute lower bound on inter-event interval.
    min_stride_fraction: float = 0.5    # Refractory = max(min_interval_s, fraction * stride_period).
    direction: str = "negative"         # "negative" detects troughs, "positive" detects peaks.


class HeelStrikeDetector:
    """Online dominant-peak detector for the bandpassed sensor signal.

    The detector is stateful. Call `step(value, time, stride_period)` once
    per sample. A non-None return value is the timestamp of the detected
    event, typically passed straight into the PhaseErrorLearner.

    Example
    -------
        det = HeelStrikeDetector(EventDetectorConfig())
        for t, v_bp in zip(time_array, bandpassed_signal):
            event_time = det.step(v_bp, t, ao.stride_period())
            if event_time is not None:
                phase_learner.on_event(ao.phase(), event_time, ao.state.omega)
    """

    def __init__(self, config: Optional[EventDetectorConfig] = None):
        self.config = config or EventDetectorConfig()
        if self.config.direction not in ("negative", "positive"):
            raise ValueError("direction must be 'negative' or 'positive'.")
        self._sign = -1.0 if self.config.direction == "negative" else 1.0

        self._t_start: Optional[float] = None
        self._prev2: Optional[float] = None
        self._prev1: Optional[float] = None
        self._prev1_time: Optional[float] = None
        self._peak_ema: Optional[float] = None
        self._last_event_time: float = -float("inf")
        self.event_times: list = []

    def reset(self) -> None:
        """Drop all state. The next call starts a fresh settling window."""
        self.__init__(self.config)

    def step(self, value: float, time: float, stride_period: float) -> Optional[float]:
        """Process one sample. Returns the event time, or None if no event."""
        c = self.config

        # The detector compares signal extrema after flipping the sign so that
        # the "negative" case becomes a maximum search on -value.
        v = self._sign * value

        if self._t_start is None:
            self._t_start = time

        event_time: Optional[float] = None

        if self._prev2 is not None and self._prev1 is not None:
            # Three-sample local-maximum test. The previous sample is a peak
            # when it is strictly greater than both neighbours.
            is_local_max = self._prev1 > self._prev2 and self._prev1 > v

            if is_local_max:
                peak_height = abs(self._prev1)
                in_settling = (
                    self._prev1_time is not None
                    and self._prev1_time < self._t_start + c.settling_s
                )

                if not in_settling:
                    # The running mean of accepted peak heights is the
                    # reference against which sub-peaks are filtered. The
                    # very first valid peak seeds the EMA so that the
                    # threshold immediately becomes meaningful.
                    if self._peak_ema is None:
                        self._peak_ema = peak_height

                    amplitude_ok = peak_height > c.peak_amplitude_fraction * self._peak_ema

                    if amplitude_ok:
                        # Update the running mean with a slow EMA, so a few
                        # noisy peaks cannot pull the threshold drastically.
                        self._peak_ema = (
                            (1.0 - c.peak_ema_alpha) * self._peak_ema
                            + c.peak_ema_alpha * peak_height
                        )

                        # Refractory period uses the oscillator-estimated
                        # stride period so that the detector adapts to
                        # walking speed automatically.
                        min_interval = max(
                            c.min_interval_s,
                            c.min_stride_fraction * stride_period,
                        )
                        if (self._prev1_time - self._last_event_time) > min_interval:
                            event_time = self._prev1_time
                            self._last_event_time = self._prev1_time
                            self.event_times.append(self._prev1_time)

        # Shift the three-sample window forward.
        self._prev2 = self._prev1
        self._prev1 = v
        self._prev1_time = time

        return event_time
