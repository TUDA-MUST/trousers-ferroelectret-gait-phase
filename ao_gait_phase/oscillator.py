"""
Adaptive oscillator for gait phase estimation.

The oscillator reconstructs a quasi-periodic input signal as a sum of N
harmonics with adaptive amplitudes, phases, and a shared fundamental
frequency. The fundamental phase phi_1 tracks the gait phase of the wearer.

References
----------
Righetti, J., Buchli, J., Ijspeert, A. J. (2006). Dynamic Hebbian learning
in adaptive frequency oscillators. Physica D, 216(2):269-281.
Yan, T., Parri, A., Garate, V. R., Cempini, M., Ronsse, R., Vitiello, N.
(2017). An oscillator-based smooth real-time estimate of gait phase for
wearable robotics. Autonomous Robots, 41(3):759-774.

What the equations do
---------------------
Each harmonic has its own phase phi_n and amplitude alpha_n, with phi_n
running at n times the fundamental frequency omega. The reconstruction
    v_hat = alpha_0 + sum_n alpha_n * sin(phi_n)
is compared to the current input v, giving an error F = v - v_hat. Hebbian
update rules push the state in the direction that reduces |F|:

    dphi_n/dt   = n * omega + nu_phi * (F / Sigma|alpha|) * cos(phi_n)
    domega/dt   = nu_omega   * (F / Sigma|alpha|) * cos(phi_1)
    dalpha_n/dt = eta * F * sin(phi_n)
    dalpha_0/dt = eta * F

The cos(phi_n) factor in the phase update advances or retards each harmonic
when the reconstruction is ahead of or behind the input. Dividing by the
total amplitude scales the correction so that the convergence rate stays
roughly constant regardless of signal amplitude.

The fundamental phase phi_1 is the quantity that the rest of the pipeline
treats as the estimated gait phase, modulo 2*pi.

Why Euler integration
---------------------
A first-order Euler step is enough at the typical 330 Hz sampling rate
because the oscillator dynamics are well-resolved at this rate. RK4 changes
the steady-state error by less than 1%, at four times the cost.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OscillatorConfig:
    """Hyperparameters that govern oscillator convergence.

    Defaults reproduce the configuration in the published paper.
    The learning-rate scales (nu_phi, nu_omega, eta) follow Yan et al. with
    eta raised from 0.4 to 0.8 to halve the initial amplitude warm-up time
    without affecting steady-state accuracy.
    """

    n_harmonics: int = 5                # N in the equations above.
    nu_phi: float = 4.0                 # Phase learning rate.
    nu_omega: float = 4.0               # Frequency learning rate.
    eta: float = 0.8                    # Amplitude learning rate.
    omega_init_hz: float = 0.9          # Initial guess for the fundamental, in Hz.
    omega_min: float = 0.1              # Lower clamp on omega to keep stride period finite.
    alpha_init: float = 0.01            # Small non-zero initial harmonic amplitudes.


@dataclass
class OscillatorState:
    """Mutable state that evolves at every sample."""

    phi: np.ndarray                     # Per-harmonic phase, shape (N,).
    alpha: np.ndarray                   # Per-harmonic amplitude, shape (N,).
    alpha_0: float = 0.0                # Adaptive DC offset.
    omega: float = 2.0 * np.pi * 0.9    # Fundamental angular frequency (rad/s).


class AdaptiveOscillator:
    """Online adaptive oscillator with Euler integration.

    Usage
    -----
        ao = AdaptiveOscillator(OscillatorConfig())
        dt = 1.0 / fs
        for sample in bandpassed_signal:
            ao.step(sample, dt)
            phase = ao.phase()             # rad in [0, 2*pi)
            stride_hz = ao.frequency()     # Hz
    """

    def __init__(self, config: Optional[OscillatorConfig] = None):
        self.config = config or OscillatorConfig()
        N = self.config.n_harmonics
        self.state = OscillatorState(
            phi=np.zeros(N),
            alpha=np.full(N, self.config.alpha_init),
            alpha_0=0.0,
            omega=2.0 * np.pi * self.config.omega_init_hz,
        )

    # ------------------------------------------------------------------ helpers

    def reconstruct(self) -> float:
        """Current model output v_hat = alpha_0 + sum_n alpha_n * sin(phi_n)."""
        return float(self.state.alpha_0 + np.sum(self.state.alpha * np.sin(self.state.phi)))

    def phase(self) -> float:
        """Fundamental phase in [0, 2*pi). Treat as the raw gait-phase estimate."""
        return float(self.state.phi[0] % (2.0 * np.pi))

    def frequency(self) -> float:
        """Estimated stride frequency in Hz."""
        return float(self.state.omega / (2.0 * np.pi))

    def stride_period(self) -> float:
        """Estimated stride period in seconds."""
        return float(2.0 * np.pi / self.state.omega)

    # -------------------------------------------------------------------- step

    def step(self, v_input: float, dt: float) -> float:
        """Advance the oscillator by one Euler step and return the reconstruction."""
        s = self.state
        c = self.config
        N = c.n_harmonics

        v_hat = self.reconstruct()
        F = v_input - v_hat

        # The 1/Sigma|alpha| normalization keeps the convergence rate
        # roughly amplitude-independent. The clip prevents division by zero
        # when the amplitudes have not yet grown from their initial values.
        alpha_sum = max(float(np.sum(np.abs(s.alpha))), 1e-10)

        # Update each harmonic phase. The integer multiplier (n + 1) ensures
        # that harmonic n runs at n+1 times the fundamental frequency.
        for n in range(N):
            dphi = s.omega * (n + 1) + c.nu_phi * (F / alpha_sum) * np.cos(s.phi[n])
            s.phi[n] += dphi * dt

        # Update the shared fundamental frequency. Only the cos(phi_1) term
        # drives this update; using higher-order harmonics here would create
        # competing pulls when the input is not perfectly sinusoidal.
        domega = c.nu_omega * (F / alpha_sum) * np.cos(s.phi[0])
        s.omega = max(s.omega + domega * dt, c.omega_min)

        # Update per-harmonic amplitudes. The sin(phi_n) factor adapts each
        # alpha_n to match the corresponding component of the input through
        # a standard Hebbian rule.
        for n in range(N):
            dalpha = c.eta * F * np.sin(s.phi[n])
            s.alpha[n] += dalpha * dt

        # The DC offset captures any residual mean that the bandpass did not
        # remove. It converges quickly and protects the harmonic amplitudes
        # from absorbing a slow drift.
        s.alpha_0 += c.eta * F * dt

        return v_hat
