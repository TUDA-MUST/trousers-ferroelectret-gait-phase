# Adaptive Oscillator + TCN Gait Phase Estimator

Reference implementation of the gait phase estimator described in:

> Seiler, J., Suppelt, M., Wilhelm, R., Beckerle, P., Kupnik, M.
> "Real-Time Gait Phase Estimation Based on Textile Integrated
Ferroelectrets, Adaptive Oscillators and Temporal Convolutional Networks." (2026).

The repository contains two Python packages:

- **`ao_gait_phase/`** — single-sensor adaptive-oscillator estimator. Pure
  NumPy / SciPy, sample-by-sample causal, designed for direct integration
  into a real-time control loop on embedded hardware. This block alone
  reproduces the single-sensor AO baseline reported in the paper.
- **`tcn_hybrid/`** — multi-sensor TCN residual-correction extension. Loads
  the published 7-layer, 48-filter multi-head TCN checkpoint and combines
  its prediction with the AO output through a confidence-weighted blend.
  Required to reproduce the hybrid AO+TCN headline result.

The full research code base, the evaluation harness, the ablations, and
the training scripts are kept out of this repository on purpose. They are
referenced in the paper's supplementary material and remain available
upon request.

## Repository layout

```text
ao_gait_phase/
├── __init__.py
├── bandpass.py         # Causal bandpass with steady-state initialization.
├── oscillator.py       # Yan / Righetti adaptive oscillator.
├── events.py           # Online dominant-peak detector for heel strike.
├── correction.py       # Event-triggered phase correction (Yan Eqs. 8-11).
└── pipeline.py         # High-level driver that wires the four together.

tcn_hybrid/
├── __init__.py
├── model.py            # MultiHeadTCN: 7 layers, 48 filters, 71 099 parameters.
├── inference.py        # Weight loading, sliding-window inference, AO+TCN blend.
└── weights/
    └── tcn_final_7L48f.pt  # Pretrained checkpoint (305 kB).

demo.py                 # Single-sensor AO demo with a synthetic signal.
demo_hybrid.py          # Hybrid AO+TCN demo on synthetic three-channel data.
requirements.txt
CITATION.cff
LICENSE                 # MIT
```

Each module is documented inline with the *why* of every design choice
rather than the *what*. The five blocks of `ao_gait_phase/` correspond
directly to the five stages described in the methods section of the
paper. The architecture of the bundled TCN matches the bolded
configuration in supplementary Section S4.

## Install and run the demos

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Single-sensor AO baseline
python demo.py

# Hybrid AO+TCN (requires PyTorch from requirements.txt)
python demo_hybrid.py
```

`demo.py` synthesizes a 60 s walking-like signal with a 1.0-2.0 Hz cadence
ramp, runs the single-sensor estimator on it, and saves
`demo_output.png`. Steady-state phase error after the first 5 s is below
0.15 rad on real data; on synthetic input the error is larger but the
visual output confirms phase lock and event detection.

`demo_hybrid.py` synthesizes a coherent three-channel signal, runs the
single-sensor AO on the primary channel, loads the published TCN weights,
and writes `demo_hybrid_output.png` showing the AO trace, the hybrid
trace, and the per-sample error. The TCN was trained on real
ferroelectret recordings; running it on synthetic data does not reproduce
paper accuracy. The demo's purpose is to exercise the inference path
rather than to benchmark.

## Single-sensor real-time integration

```python
from ao_gait_phase import GaitPhaseConfig, GaitPhaseEstimator

cfg = GaitPhaseConfig(fs=330.0)          # 330 Hz default.
estimator = GaitPhaseEstimator(cfg)

for sample, time_s in sensor_stream:
    gait_phase = estimator.step(sample, time_s)
    # gait_phase is in [0, 2*pi); 0 corresponds to heel strike after warm-up.
```

Replace any of the four sub-blocks by passing a custom configuration:

```python
from ao_gait_phase import (
    BandpassConfig, OscillatorConfig, EventDetectorConfig, PhaseCorrectorConfig,
    GaitPhaseConfig, GaitPhaseEstimator,
)

cfg = GaitPhaseConfig(
    fs=500.0,
    bandpass=BandpassConfig(fs=500.0, f_low=0.7, f_high=20.0),
    oscillator=OscillatorConfig(n_harmonics=4, eta=0.6),
    detector=EventDetectorConfig(direction="positive"),
    corrector=PhaseCorrectorConfig(adaptive_kp=True),
)
```

## Hybrid AO+TCN inference

```python
import numpy as np
from ao_gait_phase import BandpassConfig, CausalBandpass, estimate_gait_phase
from tcn_hybrid import estimate_hybrid_phase, load_pretrained

# raw_3ch: (n_samples, 3) — M.Gastrocnemius, M.Vastus Medialis, M.Rectus Femoris
fs = 330.0
bp = CausalBandpass(BandpassConfig(fs=fs))
bandpass_3ch = np.column_stack([bp.apply(raw_3ch[:, i]) for i in range(3)])

ao_result = estimate_gait_phase(raw_3ch[:, 0], fs=fs)
phi_hybrid = estimate_hybrid_phase(
    bandpass_3ch=bandpass_3ch,
    phi_ao=ao_result.phase_corrected,
    model=load_pretrained(),
)
```

The bundled checkpoint expects the channel order Gastrocnemius, Vastus
Medialis, Rectus Femoris and a native sampling rate of 330 Hz. The
inference path internally downsamples the bandpass signal to 110 Hz for
the TCN; the AO and the final output stay at 330 Hz.

## License and citation

This code is released under the MIT License (see `LICENSE`). If you use it
in academic work, please cite the paper above. A `CITATION.cff` is
included for tools that read citation metadata directly. The data used
for the paper's evaluation are archived separately on TUdatalib; see the
supplementary material for the DOI.

## Contact

Open a GitHub issue or contact the corresponding author at
`julian.seiler@tu-darmstadt.de`.
