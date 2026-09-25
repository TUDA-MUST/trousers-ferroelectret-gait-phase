"""Load the public recordings (TUdatalib, DOI 10.48328/tudatalib-2246) at a common 330 Hz.

The two sessions were sampled at about 333 Hz (S1) and 392 Hz (S2). Every recording is
brought to a uniform 330 Hz grid with anti-aliased polyphase resampling right after
loading, then trimmed to the analysis window of the paper. The order matters for exact
reproduction: resample the full recording first, trim afterwards.
"""
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

FS = 330.0
MUSCLES = ('G', 'VM', 'RF')          # M. Gastrocnemius (primary), M. Vastus Medialis, M. Rectus Femoris
SIGNAL_KEYS = ('ferro_1', 'ferro_2', 'ferro_3', 'ferro_4', 'fsr')
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS = REPO_ROOT / 'params' / 'recordings.json'


def load_registry(path=DEFAULT_RECORDINGS):
    """Return the recording registry (params/recordings.json) as a dict."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def find_dataset_root(path):
    """Locate the dataset root, i.e. the folder that contains data/session_01.

    Accepts the root itself or a folder up to two levels above it (for example the
    folder into which the TUdatalib archive was extracted).
    """
    p = Path(path).expanduser().resolve()
    candidates = [p] + sorted(q for q in p.glob('*') if q.is_dir()) \
        + sorted(q for q in p.glob('*/*') if q.is_dir())
    for c in candidates:
        if (c / 'data' / 'session_01').is_dir():
            return c
    raise FileNotFoundError(f'no data/session_01 folder found in or below {p}')


def read_csv(path, drop_first_sample=False):
    """Read one sensor-unit CSV (time_s, ferro_1..ferro_4 in mV, fsr in ADC counts).

    Returns a dict with 'time' (s, zeroed), 'ferro_1'..'ferro_4' (mV), 'fsr' (V) and
    'fs' (mean sampling rate, Hz).
    """
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    if drop_first_sample:
        data = data[1:]
    time_s = data[:, 0]
    time_s = time_s - time_s[0]
    fs = 1.0 / np.mean(np.diff(time_s))
    fsr = data[:, 5]
    if np.max(fsr) > 100:                 # 16-bit ADC counts against a 3.3 V reference
        fsr = fsr * (3.3 / 2 ** 16)
    return {'time': time_s, 'ferro_1': data[:, 1], 'ferro_2': data[:, 2],
            'ferro_3': data[:, 3], 'ferro_4': data[:, 4], 'fsr': fsr, 'fs': fs}


def resample(d, target_fs=FS):
    """Polyphase-resample every channel of a loaded recording to a uniform target_fs grid.

    The rational up/down ratio is derived from the mean native rate; all channels share
    it and are cut to a common length. The time vector is rebuilt as arange(n)/target_fs.
    """
    fs = float(d['fs'])
    r = Fraction(target_fs / fs).limit_denominator(2000)
    up, down = r.numerator, r.denominator
    out = dict(d)
    ys = {k: signal.resample_poly(np.asarray(d[k], dtype=float), up, down).astype(float)
          for k in SIGNAL_KEYS}
    n = min(len(y) for y in ys.values())
    for k in SIGNAL_KEYS:
        out[k] = ys[k][:n]
    out['time'] = np.arange(n) / target_fs
    out['fs'] = float(target_fs)
    return out


def trim(d, start_s, end_s):
    """Cut start_s seconds from the start and end_s seconds from the end; re-zero time."""
    keys = ('time',) + SIGNAL_KEYS
    if start_s > 0:
        i = int(start_s * d['fs'])
        for k in keys:
            d[k] = d[k][i:]
        d['time'] = d['time'] - d['time'][0]
    if end_s > 0:
        i = len(d['time']) - int(end_s * d['fs'])
        for k in keys:
            d[k] = d[k][:i]
    return d


def load_recording(dataset_root, name, registry=None, drop_first_sample=None):
    """Load one evaluated recording at 330 Hz, trimmed to the paper's analysis window.

    Returns the signal dict of read_csv (at 330 Hz) plus 'name', 'meta' (the registry
    entry) and 'muscles' ({'G', 'VM', 'RF'} -> raw ferroelectret signal in mV).
    drop_first_sample=None uses the registry setting (see params/recordings.json).
    """
    registry = registry or load_registry()
    meta = registry['recordings'][name]
    drop = meta.get('drop_first_sample', False) if drop_first_sample is None else drop_first_sample
    d = read_csv(Path(dataset_root) / meta['file'], drop_first_sample=drop)
    d = trim(resample(d), meta['trim_start_s'], meta['trim_end_s'])
    d['name'] = name
    d['meta'] = meta
    d['muscles'] = {m: d[meta['channels'][m]] for m in MUSCLES}
    return d
