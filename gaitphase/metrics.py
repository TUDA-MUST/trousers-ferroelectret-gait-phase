"""Evaluation metrics of the paper (Section 2.10).

All phase errors are wrapped differences dphi = (phi_est - phi_GT + pi) mod 2pi - pi.

continuous_errors  continuous MAE and RMSE over every sample of a recording
direct_mae         |phi_est| at the FSR heel strikes, after the first three
stride_time_errors stride-time phase error 2pi |T_true - T_est| / T_true, estimated strides
                   from phase wraps matched to FSR strides within 0.5 s (first three skipped)
per_stride_mae     mean |dphi| within each reference stride (strides with >= 10 samples)
temporal_errors    signed timing offset of each estimated heel strike (phase wrap 2pi -> 0),
                   greedy 1:1 matched to FSR heel strikes within +-40 % of the local stride,
                   in seconds and in percent of the local stride duration
event_consistency  Rayleigh resultant length R of the reference phase at detected events
protocol_segments  convergence / steady / transition windows from the protocol timing
"""
import numpy as np

TWO_PI = 2 * np.pi


def wrap(x):
    return (x + np.pi) % TWO_PI - np.pi


def continuous_errors(phase_est, phase_gt):
    """Continuous MAE and RMSE (rad) over all samples."""
    err = (phase_est - phase_gt + np.pi) % TWO_PI - np.pi
    v = ~np.isnan(err)
    return float(np.mean(np.abs(err[v]))), float(np.sqrt(np.mean(err[v] ** 2)))


def direct_mae(phase_est, hs_idx, skip=3):
    """Mean |wrapped estimate| at the FSR heel strikes, skipping the first `skip`."""
    if len(hs_idx) <= skip:
        return float('nan')
    e = (phase_est[hs_idx[skip:]] + np.pi) % TWO_PI - np.pi
    return float(np.mean(np.abs(e)))


def stride_time_errors(time, phase_est, hs_idx, true_stride_times, skip=3, max_offset_s=0.5):
    """Signed stride-time phase errors (rad) for FSR strides matched to estimated strides."""
    wraps = np.where(np.diff(phase_est) < -np.pi)[0]
    if len(wraps) < 2:
        return np.array([])
    est_T = np.diff(time[wraps])
    est_start = time[wraps[:-1]]
    true_start = time[hs_idx[:-1]]
    est, true = [], []
    for i, t in enumerate(true_start):
        if i < skip:
            continue
        dist = np.abs(est_start - t)
        j = np.argmin(dist)
        if dist[j] < max_offset_s:
            est.append(est_T[j])
            true.append(true_stride_times[i])
    if not est:
        return np.array([])
    est, true = np.array(est), np.array(true)
    return ((true - est) / true) * TWO_PI


def st_mae(time, phase_est, hs_idx, true_stride_times):
    """Stride-time MAE (rad); errors above 2*pi are excluded as matching artifacts."""
    st = np.abs(stride_time_errors(time, phase_est, hs_idx, true_stride_times))
    st = st[st <= TWO_PI]
    return float(np.mean(st)) if len(st) else float('nan')


def stride_bounds(phase_gt):
    """Stride boundaries of the reference phase: indices where it wraps from 2*pi to 0."""
    return np.where(np.diff(phase_gt) < -np.pi)[0].tolist()


def per_stride_mae(phase_est, phase_gt, min_samples=10):
    """List of (stride index, start, end, MAE rad) over the reference strides."""
    err = (phase_est - phase_gt + np.pi) % TWO_PI - np.pi
    b = stride_bounds(phase_gt)
    out = []
    if len(b) < 3:
        return out
    for k in range(len(b) - 1):
        s, e = b[k], b[k + 1]
        if e > len(phase_gt):
            break
        sl = err[s:e]
        v = ~np.isnan(sl)
        if np.sum(v) < min_samples:
            continue
        out.append((k, s, e, float(np.mean(np.abs(sl[v])))))
    return out


def phase_wrap_times(phase_est, time):
    """Estimated heel strikes: the sample after each wrap of the phase from 2*pi to 0."""
    hs = np.where(np.diff(phase_est) < -np.pi)[0] + 1
    hs = hs[hs < len(time)]
    return time[hs]


def match_events(event_times, hs_times, max_fraction=0.4):
    """Greedy 1:1 matching of events to heel strikes within max_fraction of the local stride.

    Heel strikes are visited in order; each takes the closest unused event inside its
    window. Returns offsets (event - heel strike, s), the matched heel-strike indices and
    the local stride period (interval to the next heel strike) of each match.
    """
    ev = np.asarray(event_times)
    hs = np.asarray(hs_times)
    per = np.empty(len(hs))
    iv = np.diff(hs)
    per[:-1] = iv
    per[-1] = iv[-1] if len(iv) > 0 else 1.0
    used, off, idx, sp = set(), [], [], []
    for i, t in enumerate(hs):
        d = ev - t
        cand = [c for c in np.where(np.abs(d) <= max_fraction * per[i])[0] if c not in used]
        if not cand:
            continue
        best = cand[np.argmin(np.abs(d[cand]))]
        used.add(best)
        off.append(d[best])
        idx.append(i)
        sp.append(per[i])
    return {'offset_s': np.array(off), 'hs_idx': np.array(idx), 'stride_period_s': np.array(sp),
            'n_missed': len(hs) - len(idx), 'n_false': len(ev) - len(used)}


def temporal_errors(phase_est, time, hs_times, min_count=5):
    """Temporal error of the estimated heel strikes against the FSR heel strikes.

    Returns None if fewer than min_count heel strikes, estimated strikes or matches exist,
    otherwise a dict with 'offset_s', 'offset_pct' (signed, % of local stride),
    'hs_idx' and 'stride_period_s'.
    """
    if len(hs_times) < min_count:
        return None
    est = phase_wrap_times(phase_est, time)
    if len(est) < min_count:
        return None
    m = match_events(est, hs_times)
    if len(m['offset_s']) < min_count:
        return None
    m['offset_pct'] = m['offset_s'] / m['stride_period_s'] * 100.0
    return m


def event_consistency(event_times, time, phase_gt):
    """Rayleigh resultant length of the reference phase at the detected events."""
    if len(event_times) == 0:
        return float('nan')
    g = np.array([phase_gt[np.argmin(np.abs(time - t))] for t in event_times])
    return float(np.sqrt(np.mean(np.sin(g)) ** 2 + np.mean(np.cos(g)) ** 2))


def protocol_segments(meta, convergence_s=10.0, transition_s=10.0, steady_margin_s=10.0):
    """Analysis segments from the protocol timing (times after trimming).

    Speed step i of a ramp starts at step_onset_s + i * speed_duration_s, where step_onset_s
    is the measured start of the first step in the trimmed time base (params/recordings.json).
    Steady segments skip steady_margin_s at both ends of a step and start after the
    convergence window; the last step runs to the end of the recording.
    Returns a list of dicts with 'condition' ('convergence', 'steady', 'transition'),
    'start_s', 'end_s' and 'speed' (km/h as a string; 'a->b' for transitions).
    """
    speeds = meta['speeds_kmh']
    segs = [{'condition': 'convergence', 'start_s': 0.0, 'end_s': convergence_s, 'speed': str(speeds[0])}]
    if meta['protocol'] == 'ramp':
        dur = meta['speed_duration_s']
        onset = meta.get('step_onset_s', 0.0)
        for i, v in enumerate(speeds):
            a, b = onset + i * dur, onset + (i + 1) * dur
            start = max(a + steady_margin_s, convergence_s)
            end = float('inf') if i == len(speeds) - 1 else b - steady_margin_s
            min_len = 0.0 if (i == 0 and onset < 0) else 5.0
            if end > start + min_len:
                segs.append({'condition': 'steady', 'start_s': start, 'end_s': end, 'speed': str(v)})
            if i < len(speeds) - 1:
                segs.append({'condition': 'transition', 'start_s': max(b - transition_s, 0),
                             'end_s': b + transition_s, 'speed': f'{v}->{speeds[i + 1]}'})
    else:
        segs.append({'condition': 'steady', 'start_s': convergence_s,
                     'end_s': meta['speed_duration_s'], 'speed': str(speeds[0])})
    return segs


def segment_of(t, segments):
    """First segment that contains time t, or None."""
    for s in segments:
        if s['start_s'] <= t < s['end_s']:
            return s
    return None
