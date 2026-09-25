"""Gait event detector (paper Section 2.4): adaptive threshold on the bandpass signal.

Running envelope E, mean mu and variance sigma^2 are exponential moving averages with
smoothing factor alpha_env. An event fires on the first sample at which the signal
crosses theta = -k_env E - k_std sigma (negative direction), provided at least
rho * T has passed since the previous event, with T = 2*pi/omega the AO stride period.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorParams:
    env_alpha: float = 0.0011429996591265945
    k_env: float = 1.0929644559084108
    k_std: float = 0.8046603457722377
    min_stride_fraction: float = 0.7967219244187553
    direction: str = 'neg'

    @classmethod
    def from_params(cls, params):
        p = params['event_detector']
        return cls(env_alpha=p['env_alpha'], k_env=p['k_env'], k_std=p['k_std'],
                   min_stride_fraction=p['min_stride_fraction'], direction=p['direction'])


class EventDetector:
    def __init__(self, params=None):
        self.p = params or DetectorParams()
        self.last_event_time = -np.inf
        self.crossed = False
        self.envelope = None
        self.mean = 0.0
        self.var = 0.0
        self.threshold = None
        self.event_times = []

    def step(self, value, time, stride_period):
        """Process one bandpass sample; return the event time or None."""
        p = self.p
        a = p.env_alpha
        if self.envelope is None:          # first sample initializes envelope and mean
            self.envelope = abs(value)
            self.mean = value
        else:
            self.envelope = (1 - a) * self.envelope + a * abs(value)
        self.mean = (1 - a) * self.mean + a * value
        self.var = (1 - a) * self.var + a * (value - self.mean) ** 2
        std = np.sqrt(self.var)

        if p.direction == 'neg':
            self.threshold = -p.k_env * self.envelope - p.k_std * std
            crossed = value <= self.threshold
        else:
            self.threshold = p.k_env * self.envelope + p.k_std * std
            crossed = value >= self.threshold

        event = None
        if crossed and not self.crossed:
            if (time - self.last_event_time) > p.min_stride_fraction * stride_period:
                event = time
                self.last_event_time = time
                self.event_times.append(time)
        self.crossed = crossed
        return event
