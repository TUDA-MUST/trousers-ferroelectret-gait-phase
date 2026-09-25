"""Adaptive oscillator (paper Section 2.3), after Righetti et al. (2006) and Yan et al. (2017).

The oscillator reconstructs its input as alpha_0 + sum_n alpha_n sin(phi_n) with N
harmonics and adapts phases, fundamental frequency and amplitudes to the reconstruction
error F = V_bp - V_hat:

    dphi_n/dt   = n omega + nu_phi   F / sum|alpha| cos(phi_n)
    domega/dt   =           nu_omega F / sum|alpha| cos(phi_1)
    dalpha_n/dt = eta F sin(phi_n),     dalpha_0/dt = eta F

Integrated with a fourth-order Runge-Kutta step (input held constant over the step);
omega is clamped at omega_min at every stage. The arithmetic follows the code used for
the paper line by line, so that results agree bit for bit.
"""
from dataclasses import dataclass

import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class OscillatorParams:
    N: int = 4
    nu_phi: float = 0.9501658222719294
    nu_omega: float = 0.24370958078477165
    eta: float = 1.592840268715275
    omega_init_hz: float = 0.9
    alpha_init: float = 0.01
    omega_min: float = 0.1

    @classmethod
    def from_params(cls, params):
        p = params['adaptive_oscillator']
        return cls(N=p['N'], nu_phi=p['nu_phi'], nu_omega=p['nu_omega'], eta=p['eta'],
                   omega_init_hz=p['omega_init_hz'], alpha_init=p['alpha_init'],
                   omega_min=p['omega_min_rad_s'])


class AdaptiveOscillator:
    def __init__(self, params=None):
        self.p = params or OscillatorParams()
        self.phi = np.zeros(self.p.N)
        self.alpha = np.ones(self.p.N) * self.p.alpha_init
        self.alpha_0 = 0.0
        self.omega = 2 * np.pi * self.p.omega_init_hz

    def _derivatives(self, phi, omega, alpha, alpha_0, v):
        p = self.p
        v_hat = alpha_0
        for n in range(p.N):
            v_hat += alpha[n] * np.sin(phi[n])
        F = v - v_hat
        alpha_sum = max(np.sum(np.abs(alpha)), 1e-10)
        dphi = np.zeros(p.N)
        dalpha = np.zeros(p.N)
        for n in range(p.N):
            dphi[n] = omega * (n + 1) + p.nu_phi * (F / alpha_sum) * np.cos(phi[n])
            dalpha[n] = p.eta * F * np.sin(phi[n])
        domega = p.nu_omega * (F / alpha_sum) * np.cos(phi[0])
        dalpha_0 = p.eta * F
        return dphi, domega, dalpha, dalpha_0

    def step(self, v, dt):
        """Advance the oscillator by one sample of input v (RK4)."""
        wmin = self.p.omega_min
        phi0, om0, al0, al00 = self.phi.copy(), self.omega, self.alpha.copy(), self.alpha_0

        k1_phi, k1_om, k1_al, k1_al0 = self._derivatives(phi0, om0, al0, al00, v)

        phi1 = phi0 + 0.5 * dt * k1_phi
        om1 = max(om0 + 0.5 * dt * k1_om, wmin)
        al1 = al0 + 0.5 * dt * k1_al
        al01 = al00 + 0.5 * dt * k1_al0
        k2_phi, k2_om, k2_al, k2_al0 = self._derivatives(phi1, om1, al1, al01, v)

        phi2 = phi0 + 0.5 * dt * k2_phi
        om2 = max(om0 + 0.5 * dt * k2_om, wmin)
        al2 = al0 + 0.5 * dt * k2_al
        al02 = al00 + 0.5 * dt * k2_al0
        k3_phi, k3_om, k3_al, k3_al0 = self._derivatives(phi2, om2, al2, al02, v)

        phi3 = phi0 + dt * k3_phi
        om3 = max(om0 + dt * k3_om, wmin)
        al3 = al0 + dt * k3_al
        al03 = al00 + dt * k3_al0
        k4_phi, k4_om, k4_al, k4_al0 = self._derivatives(phi3, om3, al3, al03, v)

        self.phi = phi0 + (dt / 6) * (k1_phi + 2 * k2_phi + 2 * k3_phi + k4_phi)
        self.omega = max(om0 + (dt / 6) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om), wmin)
        self.alpha = al0 + (dt / 6) * (k1_al + 2 * k2_al + 2 * k3_al + k4_al)
        self.alpha_0 = al00 + (dt / 6) * (k1_al0 + 2 * k2_al0 + 2 * k3_al0 + k4_al0)

    @property
    def phase(self):
        """Fundamental phase varphi_AO = phi_1 mod 2*pi."""
        return self.phi[0] % TWO_PI

    @property
    def stride_period(self):
        return 2 * np.pi / self.omega

    @property
    def quality(self):
        """Reconstruction quality q = sum_n |alpha_n|, the weight in the frequency consensus."""
        return np.sum(np.abs(self.alpha))
