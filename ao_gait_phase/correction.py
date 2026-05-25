"""
Event-triggered phase correction (Yan et al. 2017, Eqs. 8-11).

The adaptive oscillator in oscillator.py tracks the sensor signal but its
fundamental phase phi_1 has an arbitrary offset relative to the gait cycle:
phi_1 = 0 might correspond to mid-stance, late swing, or any other instant,
depending on the sensor signal shape. This module adds a small correction
state phi_e that pulls the oscillator's phase toward zero at heel-strike
events, producing a corrected gait phase phi_corr where 0 = heel strike.

The four-step loop per Yan et al.
---------------------------------

(Eq. 8) On every detected heel-strike event we compute the wrapped phase
error between the oscillator's instantaneous phase phi_n and the expected
reference phi_ref:

    P_e = wrap_to_pi(phi_ref - phi_n)

(Eq. 9) Unwrap P_e across consecutive events so that successive corrections
sit on the same branch of the multivalued atan2 surface:

    P_tilde = P_e + correction_for_wrap_jumps_relative_to_prev_P_tilde

(Eq. 10) An impulsive update to a small phase-correction state phi_e:

    epsilon_phi = kp * (P_tilde - phi_e)        at event time
    dphi_e/dt   = epsilon_phi * omega * exp(-omega * (t - t_event))

The exponential envelope spreads the correction over roughly one stride so
the corrected phase never jumps discontinuously.

(Eq. 11) Output corrected phase:

    phi_corr = wrap_2pi(phi_n + phi_e - phi_ref)

Why learn phi_ref online
------------------------
Yan et al. assumed a fixed phi_ref calibrated from an external sensor
(FSR, IMU). For the textile-integrated FMG setup the relationship between
the bandpass-minimum event and heel strike depends on garment fit, muscle
geometry, and walking speed. Learning phi_ref as the circular mean of phi_n
at detected events removes the calibration step entirely, at the cost of a
short warm-up period (set `warmup_events` to skip the first few corrections
while the mean converges).

Adaptive kp (optional)
----------------------
Setting `adaptive_kp=True` scales the proportional gain by the relative rate
of frequency change |d omega / dt| / omega. The oscillator already tracks
steady-state walking well, so a small kp suffices there; during speed
changes a larger kp catches up faster. This gives the AO+correction system
a flat tracking error across cadence ramps and steady walking.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _wrap_to_pi(angle: float) -> float:
    """Map an angle into [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _wrap_to_2pi(angle: float) -> float:
    """Map an angle into [0, 2*pi)."""
    return angle % (2.0 * np.pi)


@dataclass
class PhaseCorrectorConfig:
    """Hyperparameters for the phase-correction loop.

    Defaults reproduce the paper configuration with online reference-phase
    learning enabled.
    """

    kp: float = 0.5                      # Base proportional gain (Yan: 0.5).
    learn_reference: bool = True         # Learn phi_ref from detected events.
    reference_alpha: float = 0.05        # EMA weight for the circular-mean update.
    warmup_events: int = 0               # Number of initial events to use only for learning phi_ref.
    integrator_clip: float = 2.0 * np.pi  # Saturation on phi_e to keep it bounded.

    adaptive_kp: bool = False            # Enable cadence-aware kp scaling.
    adaptive_kp_gain: float = 5.0        # Sensitivity of kp to |d omega / dt| / omega.
    kp_min: float = 0.1                  # Lower bound on adaptive kp.
    kp_max: float = 1.0                  # Upper bound on adaptive kp.


class PhaseErrorLearner:
    """Implements Eqs. 8-11 plus optional online phi_ref learning.

    Usage
    -----
        learner = PhaseErrorLearner(PhaseCorrectorConfig())
        for t, sample in stream:
            v_hat = ao.step(sample, dt)
            event_time = detector.step(bp_sample, t, ao.stride_period())
            if event_time is not None:
                learner.on_event(ao.phase(), event_time, ao.state.omega)
            learner.step(ao.state.omega, t, dt)
            phi_corr = learner.corrected_phase(ao.phase())
    """

    def __init__(self, config: Optional[PhaseCorrectorConfig] = None):
        self.config = config or PhaseCorrectorConfig()

        self.phi_e: float = 0.0               # Slow correction state (Eq. 10).
        self.epsilon_phi: float = 0.0         # Impulse magnitude set at each event.
        self._P_tilde_prev: float = 0.0       # Previous unwrapped P_e for branch tracking.
        self._last_event_time: Optional[float] = None

        # Circular running mean of phi_n at events, stored as the (cos, sin)
        # pair so the EMA stays well-behaved across the 2*pi wrap.
        self._ref_cos: float = 1.0
        self._ref_sin: float = 0.0
        self._learned_reference: float = 0.0
        self._event_count: int = 0

        # State for the adaptive-kp option.
        self._prev_omega: Optional[float] = None
        self._domega_dt: float = 0.0

    # ----------------------------------------------------------------- helpers

    def _update_reference(self, phi_n: float) -> None:
        """Refresh the EMA circular mean of phi_n with the latest event."""
        a = self.config.reference_alpha
        self._ref_cos = (1.0 - a) * self._ref_cos + a * np.cos(phi_n)
        self._ref_sin = (1.0 - a) * self._ref_sin + a * np.sin(phi_n)
        self._learned_reference = _wrap_to_2pi(np.arctan2(self._ref_sin, self._ref_cos))

    def _current_reference(self) -> float:
        """Return the reference phase used by Eq. 8."""
        return self._learned_reference if self.config.learn_reference else 0.0

    def _kp(self, omega: float) -> float:
        """Return the current proportional gain (possibly adapted to cadence)."""
        c = self.config
        if c.adaptive_kp and omega > 0.0:
            relative_change = abs(self._domega_dt) / omega
            scale = min(1.0, c.adaptive_kp_gain * relative_change)
            return c.kp_min + (c.kp_max - c.kp_min) * scale
        return c.kp

    # ------------------------------------------------------------------ events

    def on_event(self, phi_n: float, time: float, omega: float) -> None:
        """Handle a detected heel-strike event at the current oscillator phase."""
        self._event_count += 1

        if self.config.learn_reference:
            self._update_reference(phi_n)

        if self._event_count <= self.config.warmup_events:
            # During warm-up, accumulate phi_ref information without firing
            # any corrections. This keeps the very first strides clean of
            # impulses driven by an untrained reference.
            self._last_event_time = time
            return

        phi_ref = self._current_reference()

        # Eq. 8: wrapped phase error at the event.
        P_e = _wrap_to_pi(phi_ref - phi_n)

        # Eq. 9: unwrap P_e across event boundaries so consecutive corrections
        # belong to the same branch. The +/- pi/2 thresholds match Yan et al.
        # and trigger only when the wrap discontinuity is unambiguous.
        P_tilde = P_e
        if P_e > 0.5 * np.pi and self._P_tilde_prev < -0.5 * np.pi:
            P_tilde = P_e - 2.0 * np.pi
        elif P_e < -0.5 * np.pi and self._P_tilde_prev > 0.5 * np.pi:
            P_tilde = P_e + 2.0 * np.pi
        self._P_tilde_prev = P_tilde

        # Eq. 10 (event part): set the impulse amplitude that drives phi_e.
        self.epsilon_phi = self._kp(omega) * (P_tilde - self.phi_e)
        self._last_event_time = time

    # ------------------------------------------------------------------ stream

    def step(self, omega: float, time: float, dt: float) -> None:
        """Advance phi_e one Euler step. Call once per sample, after on_event."""
        c = self.config

        # Track |d omega / dt| with a slow EMA so the adaptive kp ignores
        # single-sample noise. tau ~ 0.5 s by default; tune via dt scaling.
        if self._prev_omega is not None:
            raw = (omega - self._prev_omega) / dt
            alpha = min(1.0, dt * 2.0)
            self._domega_dt = (1.0 - alpha) * self._domega_dt + alpha * raw
        self._prev_omega = omega

        if self._last_event_time is None:
            return

        # Eq. 10 (continuous part): exponentially-decaying impulse around
        # the event, spread over roughly one stride period.
        dt_since_event = time - self._last_event_time
        dphi_e = self.epsilon_phi * omega * np.exp(-omega * dt_since_event)
        self.phi_e += dphi_e * dt
        self.phi_e = float(np.clip(self.phi_e, -c.integrator_clip, c.integrator_clip))

    # ------------------------------------------------------------------ output

    def corrected_phase(self, phi_n: float) -> float:
        """Corrected gait phase. 0 corresponds to heel strike after warm-up.

        Eq. 11 plus the reference-phase mapping. The returned value lies in
        [0, 2*pi) and is the quantity an exoskeleton or prosthesis would use
        to schedule assistance.
        """
        return _wrap_to_2pi(phi_n + self.phi_e - self._current_reference())
