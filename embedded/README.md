# ESP32 port of the gait phase estimator

> **Scope of this copy.** This folder holds the microcontroller port as validated on
> hardware in September 2026, without build artifacts. Three points differ from the
> description below:
>
> 1. `include/tcn_weights_gen.h` and the expected outputs of the TCN self-test in
>    `include/tcn_test_gen.h` come from an earlier checkpoint of the hybrid network that was
>    trained on all six ramp speeds, before the paper networks were retrained with two
>    windows of each ramp held out (`../weights/`). The architecture is the same, so the
>    timing and memory figures in [RESULTS.md](RESULTS.md) hold; its accuracy figures refer
>    to the earlier checkpoint. To deploy the paper network, regenerate the header with
>    `python tools/export_weights_c.py ../weights/tcn_hybrid_7L48f.pt include/tcn_weights_gen.h`.
>    The TCN boot self-test then reports a mismatch until its expected head outputs are
>    recomputed: evaluate the five windows stored in `include/tcn_test_gen.h` with
>    `gaitphase.tcn.run_heads` and the new weights, and write the results in the same layout.
>    The oscillator self-test and all timing measurements are unaffected.
> 2. The per-recording reference vectors `reference/*.npz` (about 120 MB), which
>    `tools/hil_validate.py` streams to the board, are not included, and neither are the
>    scripts that produced them from the internal analysis code (`tools/export_params.py`,
>    `tools/export_multi.py`, `tools/tcn_streaming_check.py`). The Python package of this
>    repository computes the same signals from the TUdatalib dataset. The measured tables
>    `reference/hil_*.csv` are included.
> 3. `include/excerpt_gen.h` contains the first 20 s of the S1 speed ramp from the TUdatalib
>    dataset (CC BY 4.0) for the boot self-test. The weight exporter referred to below as
>    `../export_weights_c.py` is `tools/export_weights_c.py` here.

PlatformIO project that runs the paper's estimators on an ESP32 and measures their
execution time: the single-sensor AO (causal bandpass, adaptive oscillator with RK4,
adaptive envelope-threshold event detector, Yan phase correction), the coupled
three-channel C-AO, and the hybrid AO+TCN with exact windowed inference of the
7-layer, 48-filter multi-head TCN. Everything algorithmic is plain C99 in
`lib/gaitphase/` with no dependency on Arduino or ESP-IDF; `src/main.cpp` is the
board-specific harness (self-tests at boot, binary serial protocol for validation).

Boards: ESP32-D0WDQ6 (Xtensa LX6, 2 cores, 240 MHz, single-precision FPU, 4 MB flash)
on a CP2102 devkit (`board = esp32dev`, environments `f32` / `f64`), and Teensy 4.1
(i.MX RT1062, Cortex-M7 at 600 MHz, 1 MB RAM; environments `teensy41` /
`teensy41_f64`). `src/board.h` holds the few board-specific services (cycle counter,
clock, heap, identification). `native/` builds the same self-tests as a Linux program
for a Raspberry Pi or a PC.

## Layout

| path | role |
| --- | --- |
| `lib/gaitphase/gp_*.c/h` | portable estimator: types, bandpass, AO, events, correction, single-sensor estimator, coupled AO, TCN |
| `include/params_gen.h` | generated: filter coefficients + `sosfilt_zi`, AO/event/correction parameters |
| `include/excerpt_gen.h` | generated: 20 s excerpt of one recording + Python reference phase (boot self-test) |
| `include/tcn_weights_gen.h` | generated: headline checkpoint `_tcn_final_7L48f_330hz.pt`, BatchNorm folded, dead taps pruned |
| `include/tcn_test_gen.h` | generated: five real 110 Hz windows + PyTorch head outputs (boot self-test) |
| `reference/*.npz` | generated: per-recording device inputs and Python reference outputs (single AO, C-AO, TCN heads) |
| `tools/export_params.py` | single-sensor configuration, excerpt, and references |
| `tools/export_multi.py` | three-channel references: C-AO output, bandpass signals, TCN heads at 110 Hz |
| `../export_weights_c.py` | TCN weight export (`python export_weights_c.py <ckpt> <out.h>`) |
| `tools/export_tcn_test.py` | self-test windows from `reference/tcn_ref_*.npz` |
| `tools/tcn_streaming_check.py` | offline: streaming vs windowed TCN, and windowed accuracy vs update rate |
| `tools/hil_validate.py` | streams recordings through the board, compares every sample, collects timing (`--multi` for the hybrid) |
| `tools/boot_report.py` | resets the board and prints the boot self-tests |
| `src/main.cpp` | boot self-tests + binary serial protocol |
| `src/board.h` | board-specific services for ESP32 and Teensy 4.x |
| `native/` | `bench_native.c` + Makefile: the self-tests as a Linux program (Raspberry Pi, PC) |

## Results

Measured numbers and their interpretation are in [RESULTS.md](RESULTS.md); the raw
per-recording tables are `reference/hil_*.csv`.

## Build environments

`f32` (default) is the deployment configuration: `gp_real = float`, the ESP32's FPU.
`f64` builds the same sources with `gp_real = double` (software floating point on the
ESP32) and exists to show the port reproduces the Python float64 reference to rounding,
so that any deviation of `f32` is attributable to single precision alone.

```text
pio run -e f32 -t upload            # ESP32: build + flash
pio run -e f64 -t upload
python tools/boot_report.py --cmd D # boot self-tests (AO, TCN) + dual-core benchmark
python tools/hil_validate.py --tag f32                      # single AO, all recordings
python tools/hil_validate.py --multi --hold 4 --tag f32     # hybrid C-AO+TCN, TCN every 4th 110 Hz sample

pio run -e teensy41 -t upload       # Teensy 4.1 (Teensy Loader opens; port enumerates as USB serial)
python tools/boot_report.py --port COM30 --timeout 8 --cmd I --cmd B --cmd T
python tools/hil_validate.py --port COM30 --tag teensy41 --cpu-mhz 600 --boot-timeout 3
python tools/hil_validate.py --port COM30 --multi --hold 1 --tag teensy41 --cpu-mhz 600 --boot-timeout 3
```

Native (Raspberry Pi Zero 2 W, 64-bit OS): copy the folder over and

```text
cd native && make CFLAGS="-O2 -mcpu=cortex-a53" && ./bench_native
```

Regenerating the inputs after a change of parameters or checkpoint:

```text
python tools/export_params.py                       # params_gen.h, excerpt_gen.h, reference/*.npz
python tools/export_multi.py                        # reference/multi_*.npz
python tools/tcn_streaming_check.py                 # reference/tcn_ref_*.npz + accuracy vs rate
python tools/export_tcn_test.py                     # tcn_test_gen.h
python ../export_weights_c.py <ckpt.pt> include/tcn_weights_gen.h
```

The PlatformIO CLI installed by the VS Code extension is
`%USERPROFILE%\.platformio\penv\Scripts\pio.exe`.

## Exactness

The reference outputs are computed on the float32-quantised input the device receives,
so `f64` differences are arithmetic only. `export_params.py` prints the effect of that
quantisation (float64 raw vs float32 raw through Python) for the record. One deliberate
difference: the port wraps each oscillator phase to [0, 2π) after every step. The
dynamics only read sin/cos of the phases, so this is exact in real arithmetic and keeps
single-precision resolution constant over long recordings.

Time is kept as a sample index on the device (`t = n / fs`), which is what the 330 Hz
reference uses.

## Timing

`gp_estimator_step()` is bracketed with the CPU cycle counter (`xthal_get_ccount`), with
interrupts and the serial driver running as they would in deployment. The boot self-test
reports min / median / mean / p99 / max cycles and microseconds per sample and the
implied CPU load at 330 Hz; `hil_validate.py` reports the same over whole recordings.
