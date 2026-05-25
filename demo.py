"""End-to-end demo: synthesize a gait-like signal and recover the phase.

Running this script produces a single figure with four panels:

    1. The synthetic raw and bandpassed signal.
    2. Estimated stride frequency over time, with the ground truth.
    3. Raw and corrected gait phase (sawtooth) over the same window.
    4. Phase error (corrected estimate minus ground truth), wrapped to +/- pi.

The synthetic signal mimics a typical bandpassed gastrocnemius FMG recording:
a slow drift, a stride-frequency ramp from 1.0 Hz to 2.0 Hz over 60 seconds,
and added white noise. The corrected phase should track the ground-truth
sawtooth within ~0.15 rad (about 2.5 percent of stride) once the oscillator
has locked, which usually takes 3-5 strides.
"""

import numpy as np
import matplotlib.pyplot as plt

from ao_gait_phase import GaitPhaseConfig, estimate_gait_phase


def synthesize_recording(duration_s: float, fs: float, seed: int = 0):
    """Build a gait-like signal with a known phase trajectory.

    The fundamental frequency ramps linearly from 1.0 Hz to 2.0 Hz, which
    spans a realistic walking range. Two harmonics give the signal the same
    asymmetric shape as a real bandpassed gastrocnemius trace.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    f_inst = 1.0 + (1.0 / duration_s) * t        # Linear ramp 1.0 -> 2.0 Hz.
    phase_true = 2.0 * np.pi * np.cumsum(f_inst) / fs

    # Asymmetric stride shape: fundamental + small second harmonic + phase
    # shift on the harmonic so the negative peak is sharper than the
    # positive peak, matching a bandpassed FMG signal.
    raw = (
        -np.cos(phase_true)
        - 0.35 * np.cos(2.0 * phase_true - 0.4)
        + 0.10 * np.sin(0.3 * t)                  # slow drift
        + 0.05 * rng.standard_normal(n)           # broadband noise
    )

    return t, raw, phase_true % (2.0 * np.pi), f_inst


def main() -> None:
    fs = 330.0
    duration_s = 60.0
    time, raw, phase_true, freq_true = synthesize_recording(duration_s, fs)

    result = estimate_gait_phase(raw, fs=fs, time=time, config=GaitPhaseConfig(fs=fs))

    # Wrap the phase error into [-pi, pi) so it is centered on zero.
    err = (result.phase_corrected - phase_true + np.pi) % (2.0 * np.pi) - np.pi

    # Skip the first 5 seconds when reporting the steady-state error. The
    # oscillator needs a few strides to lock and the event detector skips
    # peaks during the same window.
    settle = time > 5.0
    mae = float(np.mean(np.abs(err[settle])))

    print(f"Detected {len(result.event_times)} events over {duration_s:.0f} s.")
    print(f"Steady-state mean absolute phase error: {mae:.3f} rad")
    print(f"  (~{mae / (2.0 * np.pi) * 100:.1f} percent of stride)")

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, raw, color="0.6", lw=0.5, label="raw")
    axes[0].plot(time, result.bandpass, color="C0", lw=0.7, label="bandpass")
    axes[0].set_ylabel("signal (a.u.)")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].plot(time, freq_true, color="0.4", lw=1.0, label="true")
    axes[1].plot(time, result.frequency_hz, color="C1", lw=0.8, label="estimated")
    axes[1].set_ylabel("frequency (Hz)")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(time, phase_true, color="0.4", lw=0.8, label="true")
    axes[2].plot(time, result.phase_corrected, color="C2", lw=0.6, label="corrected")
    axes[2].set_ylabel("gait phase (rad)")
    axes[2].legend(loc="upper right", fontsize=9)
    for t_ev in result.event_times:
        axes[2].axvline(t_ev, color="0.85", lw=0.5, zorder=0)

    axes[3].plot(time, err, color="C3", lw=0.5)
    axes[3].axhline(0.0, color="0.6", lw=0.4)
    axes[3].set_ylabel("error (rad)")
    axes[3].set_xlabel("time (s)")
    axes[3].set_ylim(-np.pi, np.pi)

    plt.tight_layout()
    plt.savefig("demo_output.png", dpi=140)
    print("Saved figure to demo_output.png")


if __name__ == "__main__":
    main()
