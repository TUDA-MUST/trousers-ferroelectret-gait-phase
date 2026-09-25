"""Event-driven phase error learning (paper Section 2.5), after Yan et al. (2017).

At each detected event t_k the wrapped error P_e = (phi_ref - phi_AO(t_k) + pi) mod 2pi - pi
is unwrapped against the previous event, and the correction state phi_e relaxes toward
it with a rate that decays within the stride:

    eps_k        = k_p (P~_e(k) - phi_e(t_k))
    dphi_e/dt    = eps_k omega exp(-omega (t - t_k))

Sign convention of this implementation: phi_e tracks P~_e = phi_ref - phi_AO, so it is
added to the oscillator phase, phi_GP = (phi_AO + phi_e - phi_ref + phi_off) mod 2pi.
phi_e is saturated at +-2pi.
"""
from dataclasses import dataclass

import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class CorrectionParams:
    kp: float = 1.2050152279950668
    phase_offset: float = -0.43542803919770107
    event_ref_phase: float = 0.0

    @classmethod
    def from_params(cls, params):
        p = params['phase_error_learner']
        return cls(kp=p['kp'], phase_offset=p['phase_offset_rad'],
                   event_ref_phase=p['event_ref_phase_rad'])


class PhaseErrorLearner:
    def __init__(self, params=None):
        self.p = params or CorrectionParams()
        self.phi_e = 0.0
        self.epsilon = 0.0
        self.last_event_time = None
        self.p_tilde_prev = 0.0

    def on_event(self, phi_ao, time):
        """Update the correction target at a detected event."""
        p_e = (self.p.event_ref_phase - phi_ao + np.pi) % TWO_PI - np.pi
        p_tilde = p_e
        if p_e > np.pi / 2 and self.p_tilde_prev < -np.pi / 2:
            p_tilde = p_e - 2 * np.pi
        elif p_e < -np.pi / 2 and self.p_tilde_prev > np.pi / 2:
            p_tilde = p_e + 2 * np.pi
        self.p_tilde_prev = p_tilde
        self.epsilon = self.p.kp * (p_tilde - self.phi_e)
        self.last_event_time = time

    def step(self, omega, time, dt):
        """Integrate the correction state over one sample."""
        if self.last_event_time is None:
            return
        dphi_e = self.epsilon * omega * np.exp(-omega * (time - self.last_event_time))
        self.phi_e += dphi_e * dt
        self.phi_e = np.clip(self.phi_e, -2 * np.pi, 2 * np.pi)

    def corrected_phase(self, phi_ao):
        """Gait phase phi_GP in [0, 2*pi)."""
        return (phi_ao + self.phi_e - self.p.event_ref_phase + self.p.phase_offset) % TWO_PI
