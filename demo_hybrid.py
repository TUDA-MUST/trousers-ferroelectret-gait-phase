"""End-to-end demo of the hybrid AO + TCN gait phase estimator.

This script generates a synthetic three-channel walking signal, runs the
single-sensor AO on the primary channel, loads the published TCN
checkpoint, and combines the two into the hybrid output described in the
paper. A four-panel figure with raw signals, AO and hybrid phase traces,
and the per-sample TCN confidence is written to `demo_hybrid_output.png`.

Note on accuracy
----------------
The TCN was trained on real ferroelectret recordings from textile-integrated
sensors. A synthetic three-channel signal will not match that training
distribution exactly, so the steady-state error reported here is larger
than the 0.106 rad per-stride MAE achieved on real data. The purpose of
this demo is to exercise the model loading, the sliding-window inference,
and the hybrid blending end-to-end rather than to reproduce paper accuracy.
For meaningful accuracy on your own data, replace `synthesize_recording`
with a loader for real bandpass-filtered three-channel input.
"""

import numpy as np
import matplotlib.pyplot as plt

from ao_gait_phase import (
    BandpassConfig,
    CausalBandpass,
    GaitPhaseConfig,
    estimate_gait_phase,
)
from tcn_hybrid import estimate_hybrid_phase, load_pretrained


def synthesize_three_channel_recording(duration_s: float, fs: float, seed: int = 0):
    """Build a three-channel gait-like signal with a known phase.

    The three channels imitate Gastrocnemius, Vastus Medialis, and Rectus
    Femoris by reusing the same fundamental phase trajectory with
    different harmonic mixes and phase offsets, plus independent noise.
    The exact shape is not meant to match a real recording; it exists to
    produce a coherent three-channel input that the TCN can ingest.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    # Cadence ramp from 1.0 to 2.0 Hz over the recording.
    f_inst = 1.0 + (1.0 / duration_s) * t
    phase_true = 2.0 * np.pi * np.cumsum(f_inst) / fs

    channels = [
        # Gastrocnemius-like: sharp negative peak around heel strike.
        -np.cos(phase_true) - 0.35 * np.cos(2.0 * phase_true - 0.4),
        # Vastus Medialis-like: shifted by a quarter cycle, lower harmonic content.
        -np.cos(phase_true - 0.5 * np.pi) - 0.20 * np.cos(2.0 * phase_true + 0.6),
        # Rectus Femoris-like: shifted again with a different harmonic balance.
        -np.cos(phase_true + 0.4 * np.pi) - 0.25 * np.cos(2.0 * phase_true + 1.2),
    ]
    signal = np.column_stack(channels)
    # Slow drift on every channel plus broadband noise.
    signal += 0.10 * np.sin(0.3 * t)[:, None]
    signal += 0.05 * rng.standard_normal(signal.shape)
    return t, signal, phase_true % (2.0 * np.pi)


def main() -> None:
    fs = 330.0
    duration_s = 60.0
    time, raw_3ch, phase_true = synthesize_three_channel_recording(duration_s, fs)

    # Step 1: bandpass each of the three channels with the same causal filter.
    bp = CausalBandpass(BandpassConfig(fs=fs))
    bp_3ch = np.column_stack([bp.apply(raw_3ch[:, i]) for i in range(3)])

    # Step 2: run the single-sensor AO on the primary channel for the
    # phase prior. The hybrid combines this with the TCN's correction.
    ao_result = estimate_gait_phase(
        raw_3ch[:, 0], fs=fs, time=time, config=GaitPhaseConfig(fs=fs),
    )

    # Step 3: load the bundled checkpoint and run hybrid inference.
    print("Loading pretrained TCN weights...")
    model = load_pretrained()
    print(f"  Loaded {sum(p.numel() for p in model.parameters())} parameters.")

    phase_hybrid = estimate_hybrid_phase(
        bandpass_3ch=bp_3ch,
        phi_ao=ao_result.phase_corrected,
        model=model,
    )

    # Compare both estimates against the ground truth, ignoring the first
    # five seconds where the AO is still locking on.
    settle = time > 5.0
    err_ao = (ao_result.phase_corrected - phase_true + np.pi) % (2.0 * np.pi) - np.pi
    err_hybrid = (phase_hybrid - phase_true + np.pi) % (2.0 * np.pi) - np.pi

    mae_ao = float(np.mean(np.abs(err_ao[settle])))
    mae_hybrid = float(np.mean(np.abs(err_hybrid[settle])))
    print(f"AO-only steady-state MAE     : {mae_ao:.3f} rad")
    print(f"Hybrid (AO+TCN) steady-state : {mae_hybrid:.3f} rad")

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(time, raw_3ch[:, 0], lw=0.5, color="C0", label="G")
    axes[0].plot(time, raw_3ch[:, 1], lw=0.5, color="C1", label="VM")
    axes[0].plot(time, raw_3ch[:, 2], lw=0.5, color="C2", label="RF")
    axes[0].set_ylabel("raw (a.u.)")
    axes[0].legend(loc="upper right", fontsize=9, ncol=3)

    axes[1].plot(time, phase_true, color="0.4", lw=0.8, label="true")
    axes[1].plot(time, ao_result.phase_corrected, color="C0", lw=0.6, label="AO only")
    axes[1].set_ylabel("phase (rad)")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(time, phase_true, color="0.4", lw=0.8, label="true")
    axes[2].plot(time, phase_hybrid, color="C3", lw=0.6, label="hybrid AO+TCN")
    axes[2].set_ylabel("phase (rad)")
    axes[2].legend(loc="upper right", fontsize=9)

    axes[3].plot(time, err_ao, color="C0", lw=0.4, label="AO only")
    axes[3].plot(time, err_hybrid, color="C3", lw=0.4, label="hybrid")
    axes[3].axhline(0.0, color="0.6", lw=0.4)
    axes[3].set_ylim(-np.pi, np.pi)
    axes[3].set_ylabel("error (rad)")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("demo_hybrid_output.png", dpi=140)
    print("Saved figure to demo_hybrid_output.png")


if __name__ == "__main__":
    main()
