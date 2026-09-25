"""Causal dilated TCN, windowed inference and the AO+TCN blend (paper Section 2.8).

Input: the three bandpass signals (G, VM, RF, in volts), downsampled from 330 Hz to
110 Hz by taking every third sample. Each prediction reads a causal window of 167
samples (1.52 s), zero-padded at the start of a recording. Input GroupNorm(1, 3) pools
statistics over channels and time within the window. Seven causal convolutions
(48 filters, kernel 5, dilations 1, 2, 4, 8, 16, 64, 128), each followed by BatchNorm,
ReLU and dropout, the last six with an identity skip; the features of the last time
step feed linear heads:

    head_abs  -> [sin, cos] of the absolute phase phi_GT       (stand-alone TCN: only this head)
    head_add  -> [sin, cos] of the additive correction dphi to the C-AO phase
    head_conf -> confidence c in [0, 1] (sigmoid)

Blend (Eq. blend):  phi_out = atan2(c sin(phi_AO + dphi) + (1 - c) sin(phi_abs),
                                    c cos(phi_AO + dphi) + (1 - c) cos(phi_abs)) mod 2pi
with phi_AO the corrected C-AO phase at 110 Hz. The 110 Hz output is held for three
samples to return to 330 Hz. The network never feeds back into the oscillators.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import as_strided

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'weights' / 'tcn_config.json'
WINDOW = 167
DS_FACTOR = 3


class CausalConv1d(nn.Module):
    def __init__(self, n_in, n_out, k, d=1):
        super().__init__()
        self.p = (k - 1) * d
        self.c = nn.Conv1d(n_in, n_out, k, dilation=d)

    def forward(self, x):
        return self.c(nn.functional.pad(x, (self.p, 0)))


class _Backbone(nn.Module):
    def __init__(self, n_in=3, nf=48, dilations=(1, 2, 4, 8, 16, 64, 128), k=5, dropout=0.2):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_in, affine=True)
        self.convs = nn.ModuleList([CausalConv1d(n_in, nf, k, dilations[0])])
        self.bns = nn.ModuleList([nn.BatchNorm1d(nf)])
        for d in dilations[1:]:
            self.convs.append(CausalConv1d(nf, nf, k, d))
            self.bns.append(nn.BatchNorm1d(nf))
        self.drop = nn.Dropout(dropout)

    def features(self, x):
        x = self.norm(x)
        h = self.drop(torch.relu(self.bns[0](self.convs[0](x))))
        for i in range(1, len(self.convs)):
            h = self.drop(torch.relu(self.bns[i](self.convs[i](h)))) + h
        return h[:, :, -1]


class MultiHeadTCN(_Backbone):
    """Hybrid network: absolute phase, additive correction and confidence heads."""

    def __init__(self, nf=48, **kw):
        super().__init__(nf=nf, **kw)
        self.head_abs = nn.Linear(nf, 2)
        self.head_add = nn.Linear(nf, 2)
        self.head_conf = nn.Linear(nf, 1)

    def forward(self, x):
        f = self.features(x)
        return self.head_abs(f), self.head_add(f), torch.sigmoid(self.head_conf(f))


class AbsoluteTCN(_Backbone):
    """Stand-alone network: absolute phase head only."""

    def __init__(self, nf=48, **kw):
        super().__init__(nf=nf, **kw)
        self.head_abs = nn.Linear(nf, 2)

    def forward(self, x):
        return self.head_abs(self.features(x))


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_model(kind, config=None, device='cpu'):
    """Load a paper network. kind: 'hybrid' (AO+TCN) or 'standalone' (TCN)."""
    config = config or load_config()
    arch = config['architecture']
    spec = config['models'][kind]
    cls = MultiHeadTCN if spec['class'] == 'MultiHeadTCN' else AbsoluteTCN
    model = cls(nf=arch['n_filters'], n_in=arch['n_inputs'], dilations=tuple(arch['dilations']),
                k=arch['kernel_size'], dropout=arch['dropout'])
    path = DEFAULT_CONFIG.parent / spec['file']
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def make_windows(x, wl=WINDOW):
    """Causal windows of an (n, channels) array: (n, channels, wl), zero-padded at the start."""
    n, nc = x.shape
    p = np.vstack([np.zeros((wl - 1, nc), dtype=x.dtype), x])
    s0, s1 = p.strides
    return np.ascontiguousarray(as_strided(p, shape=(n, wl, nc), strides=(s0, s0, s1)).transpose(0, 2, 1))


def network_input(bp_g, bp_vm, bp_rf, factor=DS_FACTOR):
    """Stack the three bandpass signals (float32), downsample and window them."""
    bp3 = np.column_stack([bp_g, bp_vm, bp_rf]).astype(np.float32)
    return make_windows(bp3[::factor])


def run_heads(model, windows, device='cpu', batch=2048):
    """Evaluate a network on all windows; returns the list of head outputs as arrays."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(windows), batch):
            y = model(torch.tensor(windows[s:s + batch], device=device))
            outs.append([t.cpu().numpy() for t in (y if isinstance(y, tuple) else (y,))])
    return [np.concatenate([o[j] for o in outs]) for j in range(len(outs[0]))]


def absolute_phase(pa):
    return np.arctan2(pa[:, 0], pa[:, 1]) % (2 * np.pi)


def blend(phi_ao_ds, pa, pd, pc):
    """Confidence-weighted circular blend of the corrected AO phase and the absolute head."""
    pc = pc.squeeze()
    ph_abs = absolute_phase(pa)
    ph_delta = np.arctan2(pd[:, 0], pd[:, 1])
    add_corr = (phi_ao_ds + ph_delta) % (2 * np.pi)
    sb = pc * np.sin(add_corr) + (1 - pc) * np.sin(ph_abs)
    cb = pc * np.cos(add_corr) + (1 - pc) * np.cos(ph_abs)
    return np.arctan2(sb, cb) % (2 * np.pi)


def hold_upsample(x, n, factor=DS_FACTOR):
    """Return a 110 Hz sequence to n samples at 330 Hz by holding each value."""
    up = np.repeat(x, factor)[:n]
    if len(up) < n:
        up = np.concatenate([up, np.full(n - len(up), up[-1])])
    return up


def predict_hybrid(model, windows, phi_cao, device='cpu'):
    """AO+TCN phase at the full rate, given the C-AO corrected phase at the full rate."""
    pa, pd, pc = run_heads(model, windows, device)
    return hold_upsample(blend(phi_cao[::DS_FACTOR], pa, pd, pc), len(phi_cao))


def predict_standalone(model, windows, n, device='cpu'):
    """Stand-alone TCN phase at the full rate (n samples)."""
    (pa,) = run_heads(model, windows, device)
    return hold_upsample(absolute_phase(pa), n)
