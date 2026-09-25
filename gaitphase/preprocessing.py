"""Causal bandpass preprocessing (paper Section 2.2) and the FSR heel-strike reference (2.10)."""
import numpy as np
from scipy import signal


def to_volts(x):
    """Ferroelectret readout in volts. Values above 100 are taken as millivolts."""
    x = np.asarray(x, dtype=float).copy()
    if np.mean(x) > 100:
        x = x * 1e-3
    return x


def causal_bandpass(x, fs, highpass_hz=0.5, highpass_order=1, lowpass_hz=15.0, lowpass_order=4):
    """First-order Butterworth highpass followed by a fourth-order Butterworth lowpass.

    Both stages are causal (sample-by-sample realizable) and start from their
    steady-state response to the first sample, which removes the start-up transient.
    """
    sos_hp = signal.butter(highpass_order, highpass_hz / (fs / 2), btype='high', output='sos')
    zi_hp = signal.sosfilt_zi(sos_hp) * x[0]
    x_hp, _ = signal.sosfilt(sos_hp, x, zi=zi_hp)
    sos_lp = signal.butter(lowpass_order, lowpass_hz / (fs / 2), btype='low', output='sos')
    zi_lp = signal.sosfilt_zi(sos_lp) * x_hp[0]
    x_bp, _ = signal.sosfilt(sos_lp, x_hp, zi=zi_lp)
    return x_bp


def bandpass_from_params(x_mv, fs, params):
    """to_volts + causal_bandpass with the corner frequencies of params['bandpass']."""
    bp = params['bandpass']
    return causal_bandpass(to_volts(x_mv), fs, bp['highpass_hz'], bp['highpass_order'],
                           bp['lowpass_hz'], bp['lowpass_order'])


def fsr_reference(time, fsr_v, fs, threshold, lowpass_hz=10.0, order=6):
    """Ground-truth gait phase from the heel FSR.

    The FSR voltage is lowpass filtered with a zero-phase sixth-order Butterworth filter
    at 10 Hz; heel strikes are rising edges above the per-recording threshold; the phase
    rises linearly from 0 to 2*pi between consecutive heel strikes (0 before the first).

    Returns a dict with 'phase', 'hs_idx', 'hs_times', 'stride_times', 'fsr_filtered'.
    """
    b, a = signal.butter(order, lowpass_hz / (fs / 2))
    fsr_f = signal.filtfilt(b, a, fsr_v)
    above = (fsr_f > threshold).astype(float)
    hs_idx = np.where(np.diff(above) > 0)[0]
    hs_times = time[hs_idx]
    phase = np.zeros_like(time)
    for i in range(1, len(hs_idx)):
        i0, i1 = hs_idx[i - 1], hs_idx[i]
        t0, t1 = time[i0], time[i1]
        phase[i0:i1 + 1] = (time[i0:i1 + 1] - t0) * (2 * np.pi / (t1 - t0))
    return {'phase': phase, 'hs_idx': hs_idx, 'hs_times': hs_times,
            'stride_times': np.diff(hs_times), 'fsr_filtered': fsr_f}
