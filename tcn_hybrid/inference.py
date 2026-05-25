"""
Hybrid AO + TCN inference for the published gait phase estimator.

This module loads the trained TCN weights, runs sliding-window inference on
a three-channel bandpass-filtered signal, and combines the result with the
adaptive oscillator's phase to produce the hybrid output described in the
paper.

Pipeline summary
----------------
1. Bandpass each of the three sensor channels with the same causal filter
   used by the single-sensor AO (see `ao_gait_phase.bandpass`).
2. Run an adaptive oscillator on the primary channel (M.~Gastrocnemius) to
   obtain the phase prior phi_AO at the native sampling rate.
3. Downsample the three-channel bandpassed signal by a factor of 3 to
   110 Hz. Build a sliding 167-sample window at every output sample.
4. Pass each window through the TCN and read the three heads:
     - sin/cos of an absolute phase prediction (the fallback during AO warm-up)
     - sin/cos of an additive correction delta_phi to be applied to phi_AO
     - a confidence weight c in [0, 1].
5. Blend the additive correction and the absolute fallback through the
   confidence:
        phi_corrected = c * (phi_AO + delta_phi) + (1 - c) * phi_abs,
   computed in (sin, cos) space and atan2-recovered to avoid wrap issues.
6. Upsample the result back to the AO's sampling rate.

The blend exploits two complementary modes of operation. During steady
walking the AO already tracks accurately, so the additive correction is
small and reliable — confidence is high. During the first few strides
after start, when the AO has not locked yet, the TCN's absolute prediction
serves as a fallback — confidence is low and the absolute head takes over.

Why the TCN expects three channels
----------------------------------
The TCN was trained on M.~Gastrocnemius, M.~Vastus Medialis, and M.~Rectus
Femoris signals because the relative amplitude and timing between these
three muscles encode stride frequency and convergence state more robustly
than any single channel. Running the network on the primary sensor alone
is unsupported.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .model import MultiHeadTCN


# Defaults that match the configuration of the released checkpoint. Change
# these only if you have retrained the network with a different setup.
DEFAULT_FS_HZ = 330.0
DEFAULT_DOWNSAMPLE_FACTOR = 3
DEFAULT_WINDOW_SAMPLES = 167


@dataclass
class HybridConfig:
    """Constants used by the hybrid inference loop."""

    fs_hz: float = DEFAULT_FS_HZ                   # Native sampling rate of the input.
    downsample_factor: int = DEFAULT_DOWNSAMPLE_FACTOR  # 3 -> 110 Hz network input rate.
    window_samples: int = DEFAULT_WINDOW_SAMPLES   # 167 samples = 1.52 s at 110 Hz.
    batch_size: int = 2048                         # GPU batch for sliding-window forward.
    device: str = "auto"                           # "auto", "cpu", or "cuda".


def _resolve_device(name: str) -> torch.device:
    """Choose CPU or GPU based on the requested name."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _default_weights_path() -> Path:
    """Locate the bundled `.pt` checkpoint regardless of how the package is installed."""
    return Path(__file__).parent / "weights" / "tcn_final_7L48f.pt"


def load_pretrained(
    weights_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> MultiHeadTCN:
    """Load the published MultiHeadTCN checkpoint into evaluation mode.

    Parameters
    ----------
    weights_path : str or Path, optional
        Path to a `.pt` file produced by the training script. If omitted,
        the bundled `tcn_final_7L48f.pt` is loaded.
    device : torch.device, optional
        Target device. Defaults to CUDA when available, otherwise CPU.
    """
    if weights_path is None:
        weights_path = _default_weights_path()
    device = device or _resolve_device("auto")

    model = MultiHeadTCN().to(device)
    # `weights_only=True` becomes the PyTorch default in a future release.
    # Setting it explicitly silences the deprecation warning and avoids
    # arbitrary pickle execution; the bundled checkpoint contains only
    # tensors so the strict loader works.
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def build_windows(signal_3ch: np.ndarray, window_samples: int) -> np.ndarray:
    """Build right-aligned sliding windows of shape (n, 3, window_samples).

    Sample i in the output uses inputs [i - window_samples + 1, ..., i] of
    the source. Earlier indices are zero-padded so the output length equals
    the input length, matching the convention used during training.
    """
    n, n_ch = signal_3ch.shape
    if n_ch != 3:
        raise ValueError(f"Expected 3-channel input, got {n_ch} channels.")
    padded = np.vstack(
        [np.zeros((window_samples - 1, n_ch), dtype=signal_3ch.dtype), signal_3ch]
    )
    # Use stride tricks to construct the windowed view without copying.
    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n, window_samples, n_ch),
        strides=(s0, s0, s1),
    )
    # Reorder to (n, channels, window_samples) so it matches PyTorch's
    # Conv1d expectation, then make a contiguous copy.
    return np.ascontiguousarray(windows.transpose(0, 2, 1))


def _upsample_to_length(arr: np.ndarray, target_length: int, factor: int) -> np.ndarray:
    """Repeat each downsampled sample `factor` times and pad/truncate to length.

    The TCN runs on 110 Hz output; each prediction therefore corresponds to
    three samples at the AO's 330 Hz rate. Repeating is appropriate because
    gait phase varies smoothly within 9 ms.
    """
    out = np.repeat(arr, factor)
    if out.shape[0] < target_length:
        # If the input length is not an exact multiple of factor, pad the
        # final value to match the AO trace.
        pad = np.full(target_length - out.shape[0], out[-1] if out.size > 0 else 0.0)
        out = np.concatenate([out, pad])
    return out[:target_length]


def run_tcn_inference(
    model: MultiHeadTCN,
    bandpass_3ch: np.ndarray,
    config: Optional[HybridConfig] = None,
) -> dict:
    """Run sliding-window TCN inference on bandpass-filtered multi-sensor input.

    Parameters
    ----------
    model : MultiHeadTCN
        Loaded checkpoint.
    bandpass_3ch : np.ndarray of shape (n_samples, 3)
        Bandpass-filtered signals at the native sampling rate (typically
        330 Hz) for the three sensors in order: Gastrocnemius, Vastus
        Medialis, Rectus Femoris.
    config : HybridConfig, optional
        Override the default sampling/window settings.

    Returns
    -------
    dict with keys:
        phi_abs    : (n_samples,) absolute phase prediction in [0, 2*pi)
        delta_phi  : (n_samples,) additive correction in [-pi, pi)
        confidence : (n_samples,) blend weight in [0, 1]
    """
    cfg = config or HybridConfig()
    device = next(model.parameters()).device

    n = bandpass_3ch.shape[0]

    # Downsample for the TCN input rate (110 Hz). The AO and its phase
    # trace continue at the native 330 Hz; only the TCN side is reduced.
    bp_ds = bandpass_3ch[:: cfg.downsample_factor].astype(np.float32)
    windows = build_windows(bp_ds, cfg.window_samples)

    # Forward pass in mini-batches to fit on small GPUs.
    abs_chunks = []
    add_chunks = []
    conf_chunks = []
    with torch.no_grad():
        for start in range(0, len(windows), cfg.batch_size):
            stop = min(start + cfg.batch_size, len(windows))
            batch = torch.from_numpy(windows[start:stop]).to(device)
            abs_out, add_out, conf_out = model(batch)
            abs_chunks.append(abs_out.cpu().numpy())
            add_chunks.append(add_out.cpu().numpy())
            conf_chunks.append(conf_out.cpu().numpy())

    abs_sincos = np.vstack(abs_chunks)
    add_sincos = np.vstack(add_chunks)
    confidence_ds = np.concatenate(conf_chunks).squeeze(-1)

    # Recover phase angles via atan2 to avoid the 2*pi wrap problem.
    phi_abs_ds = np.arctan2(abs_sincos[:, 0], abs_sincos[:, 1]) % (2.0 * np.pi)
    delta_phi_ds = np.arctan2(add_sincos[:, 0], add_sincos[:, 1])

    return {
        "phi_abs": _upsample_to_length(phi_abs_ds, n, cfg.downsample_factor),
        "delta_phi": _upsample_to_length(delta_phi_ds, n, cfg.downsample_factor),
        "confidence": _upsample_to_length(confidence_ds, n, cfg.downsample_factor),
    }


def blend_hybrid(phi_ao: np.ndarray, tcn_outputs: dict) -> np.ndarray:
    """Combine the AO phase with the TCN heads into the final hybrid output.

    The blend is computed in (sin, cos) space so the wrap discontinuity at
    2*pi does not introduce spikes when the confidence weight transitions.
    """
    phi_abs = tcn_outputs["phi_abs"]
    delta_phi = tcn_outputs["delta_phi"]
    confidence = tcn_outputs["confidence"]

    # Additive correction applied to the AO phase.
    phi_add = (phi_ao + delta_phi) % (2.0 * np.pi)

    # Confidence-weighted circular blend between the additive correction
    # and the absolute fallback.
    sin_b = confidence * np.sin(phi_add) + (1.0 - confidence) * np.sin(phi_abs)
    cos_b = confidence * np.cos(phi_add) + (1.0 - confidence) * np.cos(phi_abs)
    return np.arctan2(sin_b, cos_b) % (2.0 * np.pi)


def estimate_hybrid_phase(
    bandpass_3ch: np.ndarray,
    phi_ao: np.ndarray,
    model: Optional[MultiHeadTCN] = None,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[HybridConfig] = None,
) -> np.ndarray:
    """End-to-end convenience wrapper.

    Parameters
    ----------
    bandpass_3ch : np.ndarray of shape (n_samples, 3)
        Bandpassed multi-sensor input at the native 330 Hz rate.
    phi_ao : np.ndarray of shape (n_samples,)
        Adaptive oscillator phase computed on the primary (Gastrocnemius)
        channel, in radians, native sample rate.
    model : MultiHeadTCN, optional
        Pre-loaded model. If omitted, `load_pretrained(weights_path)` is
        called with the bundled checkpoint.
    weights_path : str or Path, optional
        Override the default checkpoint path.
    config : HybridConfig, optional
        Override the default sampling/window settings.

    Returns
    -------
    np.ndarray
        Corrected gait phase in [0, 2*pi), one value per input sample.
    """
    if model is None:
        model = load_pretrained(weights_path)
    tcn_out = run_tcn_inference(model, bandpass_3ch, config)
    return blend_hybrid(phi_ao, tcn_out)
