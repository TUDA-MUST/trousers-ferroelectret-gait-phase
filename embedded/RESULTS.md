# ESP32 port: measured results (2026-09-04)

Board: ESP32-D0WDQ6 revision 1 (Xtensa LX6, two cores, 240 MHz, single-precision FPU),
4 MB flash at 80 MHz QIO, Arduino framework via PlatformIO `espressif32@6.7.0`,
GCC 8.4 `-O2`. Timing is the CPU cycle counter around the estimator call with the
serial driver and interrupts running as in deployment. Reference outputs come from the
paper's Python pipeline on the identical float32 input (`tools/export_params.py`,
`tools/export_multi.py`); the five evaluation recordings are the paper's.

## 1. Single-sensor AO (bandpass + oscillator + event detector + phase correction)

Per-sample cost at 330 Hz, single precision (`f32`), from `reference/hil_f32.csv`:

| recording | samples | mean µs | median µs | p99 µs | max µs | load, mean | events dev/ref | max abs Δφ vs Python |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ramp_60s (S1) | 106,045 | 45.3 | 44.6 | 67.0 | 90.6 | 1.49 % | 310/310 | 6.4e-5 rad |
| Step_6.7_4.6 (S1) | 75,706 | 45.3 | 44.6 | 67.5 | 90.0 | 1.49 % | 250/250 | 3.1 rad, see below |
| Const_4.5 (S1) | 75,715 | 45.4 | 44.8 | 67.2 | 90.6 | 1.50 % | 201/201 | 8.4e-5 rad |
| Ramp_60s_v2 (S2) | 114,737 | 45.4 | 44.8 | 67.0 | 91.2 | 1.50 % | 298/299 | 9.4e-2 rad, see below |
| Ramp_30s (S2) | 54,341 | 45.4 | 44.7 | 67.1 | 90.7 | 1.50 % | 153/153 | 6.5e-5 rad |

Sample period 3,030 µs; the estimator uses 1.5 % of one core. State size 492 bytes.
The double-precision build (`f64`, software floating point) reproduces the Python
reference to 2.4e-7 rad with identical events on all five recordings at 468 µs per
sample (15 % load), which establishes that the port is exact and that the two
single-precision deviations below are rounding effects:

- Ramp_60s_v2: one of 299 events is not detected on the device (threshold crossing
  decided differently at float32 rounding); the phase re-anchors at the next event
  after a 4 s transient; MAE against the FSR reference 0.1282 rad (device) vs 0.1286
  (Python).
- Step_6.7_4.6: identical events, but during the recording's initial transient (20–40 s,
  where the correction state swings across the ±π unwrap decision) one decision flips,
  the two trajectories separate by up to π for ~2 s and reconverge; after 60 s they
  agree to 0.006 rad. MAE against the FSR reference 0.3369 vs 0.3341 rad.

The Python reference itself moves by up to 9.5e-4 rad on Step_6.7_4.6 when its input is
quantised to float32, i.e. this recording is sensitive to rounding independently of the
port.

## 2. Coupled AO (three channels, Kuramoto frequency consensus)

`reference/hil_multi_f32_h11.csv` (TCN at 10 Hz in the same loop):

| recording | C-AO mean µs | p99 µs | load | events | max abs Δφ vs Python | MAE vs FSR dev / py |
| --- | --- | --- | --- | --- | --- | --- |
| Ramp_60s | 141.9 | 509 | 4.7 % | 310/310 | 7.9e-4 rad | 0.1165 / 0.1165 |
| Step_6.7_4.6 | 142.2 | 527 | 4.7 % | 250/250 | 2.6e-4 rad | 0.3035 / 0.3035 |
| Const_4.5 | 141.7 | 504 | 4.7 % | 201/201 | 3.1e-5 rad | 0.0769 / 0.0769 |
| Ramp_60s_v2 | 142.1 | 525 | 4.7 % | 298/299 | 9.5e-2 rad | 0.1271 / 0.1275 |
| Ramp_30s | 142.0 | 523 | 4.7 % | 153/153 | 3.2e-5 rad | 0.1242 / 0.1242 |

State size 1,308 bytes. The p99 is set by the samples that follow a TCN inference on
the same core (the inference evicts the oscillator code from the flash cache); on the
dual-core layout below the oscillator core is not disturbed.

## 3. TCN, exact windowed inference (7 layers, 48 filters, 167-sample window)

- Agreement with PyTorch on five real windows: max |Δ| 1.8e-7 over the five head outputs.
- Work per inference with dependency masks (only the positions the last output needs):
  966,192 MAC instead of 9,779,520 for the full window (positions per layer
  84 / 42 / 21 / 11 / 3 / 2 / 1).
- Time per inference on one core, weights read from memory-mapped flash at 40 MHz DIO:
  75 ms → 64 ms with four accumulators; at 80 MHz QIO with 128 kB of the 230 kB
  weights staged in SRAM: **34.3 ms (29 Hz)**. The remaining cost is the LX6's
  floating-point issue rate, not memory.
- Memory: weights 230 kB (flash, 128 kB of it mirrored in SRAM), activations 66 kB
  (two 167 × 48 planes + normalised input), free heap after everything 102 kB.

## 4. Hybrid AO+TCN, dual-core layout

Sample-rate estimator on core 1, TCN task on core 0 (`D` command):

| | AO median µs | AO p99 µs | AO max µs | TCN on core 0 |
| --- | --- | --- | --- | --- |
| AO alone | 44.7 | 55.9 | 161 | idle |
| AO while core 0 runs the TCN back to back | 44.7 | 55.9 | 216 | 34.4 ms each, 29 Hz |

End-to-end hybrid output (C-AO phase blended with the TCN heads, heads held between
updates) against the host reference: max 1.3e-5 rad with the TCN at 110 Hz on 12 s;
≤8e-4 rad (rms ≤7e-5) on the full recordings with the TCN at 10 Hz and at 27.5 Hz,
identical MAE against the FSR reference to four decimals on four of five recordings
(the fifth inherits the C-AO event flip above).

Device-measured hybrid MAE against the FSR reference with the TCN at the deployable
27.5 Hz (`reference/hil_multi_f32_h4.csv`, 35,548 inferences at 34.2 ms each):

| recording | C-AO alone | hybrid, TCN 27.5 Hz | hybrid, Python at the same rate |
| --- | --- | --- | --- |
| Ramp_60s (S1) | 0.1165 | 0.0733 | 0.0733 |
| Step_6.7_4.6 (S1) | 0.3035 | 0.2093 | 0.2093 |
| Const_4.5 (S1) | 0.0769 | 0.0965 | 0.0965 |
| Ramp_60s_v2 (S2) | 0.1271 | 0.1058 | 0.1062 |
| Ramp_30s (S2) | 0.1242 | 0.1337 | 0.1337 |
| mean | 0.1496 | 0.1237 | 0.1238 |

## 5. Accuracy of the windowed TCN at reduced update rate (offline, Python)

Mean MAE against the FSR reference over the five recordings, exact windowed inference
whose head outputs are held for k samples at 110 Hz while the blend uses the current
C-AO phase (`tools/tcn_streaming_check.py`, `reference/tcn_streaming_check.json`):

| TCN rate | 110 Hz | 55 Hz | 37 Hz | 27.5 Hz | 22 Hz | 10 Hz | 5 Hz | C-AO alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAE, rad | 0.1198 | 0.1211 | 0.1225 | 0.1238 | 0.1255 | 0.1354 | 0.1544 | 0.1583 |

A ring-buffer "streaming" implementation (58,560 MAC per sample) is **not** equivalent
to the windowed model: GroupNorm statistics are taken over the window and the deeper
layers were trained on zero-padded windows. Simulated offline it deviates from the
windowed output by up to π per sample (rms 0.08–0.15 rad) and its mean MAE is 0.1256.
It was therefore not ported.

## 6. Teensy 4.1 (i.MX RT1062, Cortex-M7 at 600 MHz, FPU with fused multiply-add)

Same sources, environment `teensy41`; all 230 kB of TCN weights and the work planes in
RAM (176 kB heap left). Boot self-tests:

| | Teensy 4.1 | ESP32 (for comparison) |
| --- | --- | --- |
| single AO, median / p99 / max per sample | 9.9 / 10.1 / 10.2 µs (0.32 % at 330 Hz) | 44.7 / 55.9 / 161 µs |
| single AO vs Python, excerpt | max 1.6e-5 rad, 17/17 events | max 1.4e-5 rad, 17/17 |
| exact TCN inference | **6.66 ms → 150 Hz** on one core | 34.3 ms → 29 Hz |
| TCN vs PyTorch | max 1.8e-7 | max 1.8e-7 |

Hardware-in-the-loop over the five recordings (`reference/hil_teensy41.csv`,
`reference/hil_multi_teensy41_h1.csv`), single precision:

| recording | AO µs/sample mean / max | AO events | C-AO µs/sample | TCN ms, 110 Hz | hybrid MAE vs FSR dev / py | hybrid max abs Δφ vs ref |
| --- | --- | --- | --- | --- | --- | --- |
| Ramp_60s (S1) | 9.7 / 10.3 | 310/310 | 28.0 | 6.66 | 0.0695 / 0.0695 | 1.2e-3 rad |
| Step_6.7_4.6 (S1) | 9.7 / 10.3 | 250/250 | 28.0 | 6.66 | 0.1998 / 0.1998 | 1.2e-3 rad |
| Const_4.5 (S1) | 9.8 / 10.3 | 201/201 | 28.0 | 6.66 | 0.0982 / 0.0982 | 2.8e-5 rad |
| Ramp_60s_v2 (S2) | 9.8 / 10.3 | 299/299 | 28.0 | 6.66 | 0.1025 / 0.1025 | 1.7e-3 rad |
| Ramp_30s (S2) | 9.8 / 10.3 | 153/153 | 28.0 | 6.66 | 0.1290 / 0.1290 | 1.1e-4 rad |

142,184 exact inferences at 6.66 ms each, one per 110 Hz sample. The hybrid output
equals the paper's AO+TCN per-recording MAE to four decimals on every recording (mean
0.1198 rad); events are identical on all five, including the S2 ramp on which the ESP32
lost one to rounding. The single-AO run on the step-change recording shows the same
±π transient as the ESP32 (MAE 0.3373 vs 0.3341), i.e. that sensitivity belongs to the
recording, not to a particular FPU. Load on the single M7 core with the TCN at 110 Hz:
0.92 % (C-AO) + 73 % (TCN) ≈ 74 %; the Teensy runs the model exactly as evaluated,
without a reduced update rate.

## What the paper can state

- The single-sensor estimator runs at 45 µs per 330 Hz sample on a 240 MHz ESP32
  (1.5 % of one core) and at 9.8 µs on a 600 MHz Cortex-M7 (0.3 %); the three-channel
  coupled estimator at 142 µs (4.7 %) and 28 µs (0.9 %).
- The exact TCN inference takes 34 ms on the ESP32 (learned correction at up to 29 Hz
  on the second core, oscillator output every 3 ms; the offline curve gives the
  accuracy cost) and 6.7 ms on the Cortex-M7, where the complete hybrid runs at the
  evaluated 110 Hz update rate at 74 % of one core with per-recording MAE identical to
  the offline evaluation.
- A high-level controller such as a Raspberry Pi 5 has an order of magnitude more
  throughput than the M7; there the sample-synchronous stage should stay on the
  acquisition microcontroller and the learned correction can run on the controller.
  `native/` measures it (untested here, no compiler on this PC).
- ESP32-S3 (the charge-amplifier host mentioned in §2.8) is a different core (LX7 with
  SIMD extensions); numbers here are for the classic ESP32 and are conservative.
