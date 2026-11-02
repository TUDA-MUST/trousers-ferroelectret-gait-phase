"""Continuous gait phase estimation from textile-integrated ferroelectret sensors.

Reference implementation of the four estimator configurations of Seiler et al.,
"Adaptive Continuous Gait Phase Estimation from Textile-Integrated Ferroelectret
Sensors" (Bioinspiration & Biomimetics):

    AO      single-sensor adaptive oscillator with event-driven phase error learning
    C-AO    three coupled oscillators with quality-weighted frequency consensus
    TCN     stand-alone causal dilated TCN (absolute phase)
    AO+TCN  hybrid: C-AO phase corrected and blended by a multi-head TCN

The oscillator path needs numpy and scipy only; the network path (gaitphase.tcn)
also needs torch.
"""
import json
from pathlib import Path

from .data import load_recording, load_registry, find_dataset_root, MUSCLES, FS
from .preprocessing import causal_bandpass, bandpass_from_params, to_volts, fsr_reference
from .estimator import run_estimator

__version__ = '1.0.1'

DEFAULT_PARAMS = Path(__file__).resolve().parents[1] / 'params' / 'estimator_params.json'


def load_params(path=DEFAULT_PARAMS):
    """Parsed params/estimator_params.json (paper Table 1)."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


__all__ = ['load_recording', 'load_registry', 'find_dataset_root', 'MUSCLES', 'FS',
           'causal_bandpass', 'bandpass_from_params', 'to_volts', 'fsr_reference',
           'run_estimator', 'load_params', '__version__']
