"""Sample-by-sample AO and C-AO estimators (paper Sections 2.3 to 2.7).

AO:   one oscillator on the primary (M. Gastrocnemius) bandpass signal, the event
      detector on the same signal, and the phase error learner.
C-AO: one oscillator per channel with the same parameters. After every oscillator step
      the frequencies are pulled toward their quality-weighted mean,

          omega_bar = sum_j q_j omega_j / sum_j q_j,   q_j = sum_n |alpha_n,j|
          omega_j  <- omega_j + kappa (omega_bar - omega_j) dt,

      applied to each channel in turn with omega_bar fixed for the sample. Event detection
      and phase correction stay single-channel on the primary signal, so the auxiliary
      channels act only through this frequency consensus.
"""
import numpy as np

from .correction import CorrectionParams, PhaseErrorLearner
from .events import DetectorParams, EventDetector
from .oscillator import AdaptiveOscillator, OscillatorParams


def run_estimator(bp_signals, time, fs, params=None, kappa=0.0, quality_weighted=True):
    """Run the AO (one signal) or C-AO (several signals, kappa > 0) over a recording.

    Parameters
    ----------
    bp_signals : sequence of 1-D arrays
        Bandpass-filtered signals in volts, primary channel first.
    time : 1-D array
        Sample times in seconds.
    fs : float
        Sampling rate in Hz.
    params : dict or None
        Parsed params/estimator_params.json; None uses the paper values.
    kappa : float
        Coupling strength of the frequency consensus (0.25 in the paper; 0 = uncoupled).

    Returns
    -------
    dict with 'phase' (corrected gait phase, rad, per sample), 'omega' (primary AO
    frequency after coupling, rad/s), 'phi_ao' (primary AO phase) and 'event_times' (s).
    """
    if params is None:
        osc_p, det_p, cor_p = OscillatorParams(), DetectorParams(), CorrectionParams()
    else:
        osc_p = OscillatorParams.from_params(params)
        det_p = DetectorParams.from_params(params)
        cor_p = CorrectionParams.from_params(params)

    signals = [np.asarray(s) for s in bp_signals]
    n = len(time)
    dt = 1.0 / fs
    aos = [AdaptiveOscillator(osc_p) for _ in signals]
    detector = EventDetector(det_p)
    learner = PhaseErrorLearner(cor_p)
    coupled = kappa > 0 and len(aos) > 1
    wmin = osc_p.omega_min

    phase = np.zeros(n)
    omega_out = np.zeros(n)
    phi_ao = np.zeros(n)
    primary = aos[0]
    x0 = signals[0]
    for i in range(n):
        for ao, x in zip(aos, signals):
            ao.step(x[i], dt)

        if coupled:
            if quality_weighted:
                q = [ao.quality for ao in aos]
                q_total = sum(q)
                if q_total > 1e-10:
                    omega_bar = sum(qj * ao.omega for qj, ao in zip(q, aos)) / q_total
                else:
                    omega_bar = np.mean([ao.omega for ao in aos])
            else:
                omega_bar = np.mean([ao.omega for ao in aos])
            for ao in aos:
                ao.omega += kappa * (omega_bar - ao.omega) * dt
                ao.omega = max(ao.omega, wmin)

        phi = primary.phase
        omega = primary.omega
        if detector.step(x0[i], time[i], primary.stride_period) is not None:
            learner.on_event(phi, time[i])
        learner.step(omega, time[i], dt)
        phase[i] = learner.corrected_phase(phi)
        omega_out[i] = omega
        phi_ao[i] = phi

    return {'phase': phase, 'omega': omega_out, 'phi_ao': phi_ao,
            'event_times': np.asarray(detector.event_times)}
