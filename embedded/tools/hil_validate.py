# -*- coding: utf-8 -*-
"""Hardware-in-the-loop validation of the ESP32 port.

Streams every evaluation recording (reference/<name>.npz, from export_params.py) through
the device over USB serial, sample by sample in chunks, and compares each returned output
with the Python reference computed on the identical float32 input. Also collects the
per-sample CPU cycle count the firmware measures around gp_estimator_step().

    python hil_validate.py [--port COM28] [--baud 921600] [--only Ramp_60s] [--tag f32]

Writes reference/hil_<tag>.csv (one row per recording) and prints a summary table.
"""
import os
import sys
import csv
import json
import time
import struct
import argparse
import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, '..'))
REF = os.path.join(PROJ, 'reference')
CHUNK = 256   # samples per request; reply is CHUNK*20 bytes, kept well below driver buffers
REC = 20      # bytes per output record
RETRIES = 3   # a whole recording is restarted from a device reset on any framing error


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def wait_ready(ser, timeout=15.0):
    """Collect the boot report up to READY. Boards with native USB (Teensy) do not reboot
    when the port opens and print nothing; after `timeout` we simply carry on, since
    every command below is self-synchronising."""
    t0 = time.time()
    lines = []
    while time.time() - t0 < timeout:
        ln = ser.readline().decode('utf-8', 'replace').rstrip()
        if ln:
            lines.append(ln)
            print('  [dev] ' + ln)
        if ln.startswith('READY'):
            return lines
    print('  (no boot report within %.0f s; assuming the board is already running)' % timeout)
    ser.reset_input_buffer()
    return lines


def cmd_info(ser):
    ser.write(b'I')
    ln = ser.readline().decode('utf-8', 'replace').rstrip()
    print('  [dev] ' + ln)
    return ln


def cmd_reset(ser):
    ser.reset_input_buffer()
    ser.write(b'Z')
    ln = ser.readline().decode('utf-8', 'replace').rstrip()
    if ln != 'OK':
        raise RuntimeError('reset failed: %r' % ln)


REPLY_DT = np.dtype([('ph', '<f4'), ('om', '<f4'), ('fl', '<f4'), ('cy', '<u4'), ('ev', '<u4')])


def read_reply(ser, k):
    """Return the payload bytes of one 'R' reply, or None if it arrived damaged."""
    hdr = ser.read(3)
    if len(hdr) == 0:
        raise RuntimeError('device silent')          # request lost on the way in
    if len(hdr) != 3 or hdr[0:1] != b'R' or struct.unpack('<H', hdr[1:3])[0] != k:
        return None
    payload = ser.read(k * REC)
    return payload if len(payload) == k * REC else None


def stream(ser, raw32):
    n = len(raw32)
    phase = np.empty(n, np.float32)
    omega = np.empty(n, np.float32)
    filt = np.empty(n, np.float32)
    cyc = np.empty(n, np.uint32)
    evf = np.empty(n, np.uint32)
    n_resend = 0
    t0 = time.time()
    for s in range(0, n, CHUNK):
        e = min(s + CHUNK, n)
        k = e - s
        ser.write(b'S' + struct.pack('<H', k) + raw32[s:e].astype('<f4').tobytes())
        payload = read_reply(ser, k)
        if payload is None:
            # bytes lost on the way back: the device state already advanced, ask again
            time.sleep(0.1)
            ser.reset_input_buffer()
            ser.write(b'A')
            payload = read_reply(ser, k)
            n_resend += 1
            if payload is None:
                raise RuntimeError('reply damaged twice at sample %d' % s)
        a = np.frombuffer(payload, dtype=REPLY_DT)
        phase[s:e] = a['ph']; omega[s:e] = a['om']; filt[s:e] = a['fl']
        cyc[s:e] = a['cy']; evf[s:e] = a['ev']
        if (s // CHUNK) % 40 == 0:
            print('    %6d/%d samples  %.1f s' % (e, n, time.time() - t0), end='\r')
    print('    %6d/%d samples  %.1f s  (%d chunk resends)' % (n, n, time.time() - t0, n_resend))
    return phase, omega, filt, cyc, evf


MULTI_DT = np.dtype([('ph', '<f4'), ('hy', '<f4'), ('bp', '<f4'), ('om', '<f4'),
                     ('cy', '<u4'), ('cyt', '<u4'), ('fl', '<u4')])
MREC = 28


def read_reply_m(ser, k):
    hdr = ser.read(3)
    if len(hdr) == 0:
        raise RuntimeError('device silent')
    if len(hdr) != 3 or hdr[0:1] != b'R' or struct.unpack('<H', hdr[1:3])[0] != k:
        return None
    payload = ser.read(k * MREC)
    return payload if len(payload) == k * MREC else None


def stream_multi(ser, raw3, chunk):
    n = len(raw3)
    rec = np.empty(n, MULTI_DT)
    n_resend = 0
    t0 = time.time()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        k = e - s
        ser.write(b'M' + struct.pack('<H', k) + np.ascontiguousarray(raw3[s:e], dtype='<f4').tobytes())
        payload = read_reply_m(ser, k)
        if payload is None:
            time.sleep(0.1)
            ser.reset_input_buffer()
            ser.write(b'A')
            payload = read_reply_m(ser, k)
            n_resend += 1
            if payload is None:
                raise RuntimeError('reply damaged twice at sample %d' % s)
        rec[s:e] = np.frombuffer(payload, dtype=MULTI_DT)
        if (s // chunk) % 20 == 0:
            print('    %6d/%d samples  %.1f s' % (e, n, time.time() - t0), end='\r')
    print('    %6d/%d samples  %.1f s  (%d chunk resends)' % (n, n, time.time() - t0, n_resend))
    return rec


def blend_np(pa, pd, pc, phi):
    ph_abs = np.arctan2(pa[:, 0], pa[:, 1]) % (2 * np.pi)
    ph_delta = np.arctan2(pd[:, 0], pd[:, 1])
    add = (phi + ph_delta) % (2 * np.pi)
    sb = pc * np.sin(add) + (1 - pc) * np.sin(ph_abs)
    cb = pc * np.cos(add) + (1 - pc) * np.cos(ph_abs)
    return np.arctan2(sb, cb) % (2 * np.pi)


def hybrid_reference(z, hold, n):
    """Host-side hybrid phase at 330 Hz for a TCN updated every `hold` 110 Hz samples."""
    pa, pd, pc = z['heads_pa'], z['heads_pd'], z['heads_pc']
    n_ds = len(pc)
    idx = (np.arange(n_ds) // hold) * hold
    ao_ds = z['ref_gp_c'][::3][:n_ds]
    hy_ds = blend_np(pa[idx], pd[idx], pc[idx], ao_ds)
    up = np.repeat(hy_ds, 3)[:n]
    if len(up) < n:
        up = np.concatenate([up, np.full(n - len(up), up[-1])])
    return up


def run_multi(ser, args, cpu_mhz):
    with open(os.path.join(REF, 'multi_manifest.json')) as f:
        manifest = json.load(f)
    if args.only:
        wanted = set(args.only.split(','))
        manifest = [m for m in manifest if m['name'] in wanted]
    ser.write(b'H' + struct.pack('<H', args.hold))
    if ser.readline().decode('utf-8', 'replace').rstrip() != 'OK':
        raise RuntimeError('hold command failed')
    print('TCN hold = %d 110 Hz samples (%.1f Hz updates)' % (args.hold, 110.0 / args.hold))
    rows = []
    for m in manifest:
        name = m['name']
        z = np.load(os.path.join(REF, 'multi_%s.npz' % name))
        raw3 = z['raw3']
        if args.seconds:
            raw3 = raw3[:int(args.seconds * m['fs'])]
        n = len(raw3)
        fs = float(m['fs'])
        period_us = 1e6 / fs
        print('%s (%s, n=%d, %.0f s)' % (name, m['session'], n, n / fs))
        for attempt in range(RETRIES):
            try:
                cmd_reset(ser)
                rec = stream_multi(ser, raw3, args.chunk)
                break
            except RuntimeError as exc:
                print('    serial framing error (%s); restarting recording (attempt %d/%d)'
                      % (exc, attempt + 2, RETRIES))
                # anything the device printed (a crash backtrace, a boot banner) is diagnostic
                time.sleep(3.0)
                dump = ser.read(ser.in_waiting or 0).decode('utf-8', 'replace')
                for ln in dump.splitlines():
                    if ln.strip() and ln.isprintable():
                        print('      [dev] ' + ln)
                ser.reset_input_buffer()
        else:
            raise RuntimeError('recording %s failed %d times' % (name, RETRIES))

        ref_c = z['ref_gp_c'][:n]
        d_c = wrap(rec['ph'].astype(np.float64) - ref_c)
        ref_h = hybrid_reference(z, args.hold, len(z['ref_gp_c']))[:n]
        d_h = wrap(rec['hy'].astype(np.float64) - ref_h)
        dev_ev = np.flatnonzero(rec['fl'] & 1)
        ref_ev = z['ref_events']; ref_ev = ref_ev[ref_ev < n]
        ev_same = len(dev_ev) == len(ref_ev) and np.array_equal(dev_ev, ref_ev)
        gt = z['gt_phase'][:n]; v = ~np.isnan(gt)
        us_c = rec['cy'].astype(np.float64) / cpu_mhz
        tcn_mask = (rec['fl'] & 2) != 0
        ms_t = rec['cyt'][tcn_mask].astype(np.float64) / cpu_mhz / 1000.0
        row = {
            'recording': name, 'session': m['session'], 'n': n, 'tag': args.tag, 'hold': args.hold,
            'cao_max_abs_dphi_rad': float(np.max(np.abs(d_c))),
            'cao_rms_dphi_rad': float(np.sqrt(np.mean(d_c ** 2))),
            'cao_events_device': int(len(dev_ev)), 'cao_events_ref': int(len(ref_ev)),
            'cao_events_identical': bool(ev_same),
            'cao_bp_max_diff_V': float(np.max(np.abs(rec['bp'].astype(np.float64) - z['ref_bp3'][:n, 0]))),
            'hyb_max_abs_dphi_rad': float(np.max(np.abs(d_h))),
            'hyb_rms_dphi_rad': float(np.sqrt(np.mean(d_h ** 2))),
            'hyb_p99_abs_dphi_rad': float(np.percentile(np.abs(d_h), 99)),
            'mae_gt_cao_device': float(np.mean(np.abs(wrap(rec['ph'][v].astype(np.float64) - gt[v])))),
            'mae_gt_cao_python': float(np.mean(np.abs(wrap(ref_c[v] - gt[v])))),
            'mae_gt_hyb_device': float(np.mean(np.abs(wrap(rec['hy'][v].astype(np.float64) - gt[v])))),
            'mae_gt_hyb_python': float(np.mean(np.abs(wrap(ref_h[v] - gt[v])))),
            'cao_us_mean': float(np.mean(us_c)), 'cao_us_p99': float(np.percentile(us_c, 99)),
            'cao_us_max': float(np.max(us_c)),
            'cao_load_mean_pct': float(100 * np.mean(us_c) / period_us),
            'tcn_n_inferences': int(tcn_mask.sum()),
            'tcn_ms_mean': float(np.mean(ms_t)) if len(ms_t) else float('nan'),
            'tcn_ms_max': float(np.max(ms_t)) if len(ms_t) else float('nan'),
        }
        rows.append(row)
        print('    C-AO vs ref: max %.3e rms %.3e rad | bp max diff %.2e | events %d/%d %s'
              % (row['cao_max_abs_dphi_rad'], row['cao_rms_dphi_rad'], row['cao_bp_max_diff_V'],
                 len(dev_ev), len(ref_ev), 'identical' if ev_same else 'DIFFERENT'))
        print('    hybrid vs ref (hold %d): max %.3e rms %.3e p99 %.3e rad'
              % (args.hold, row['hyb_max_abs_dphi_rad'], row['hyb_rms_dphi_rad'], row['hyb_p99_abs_dphi_rad']))
        print('    MAE vs FSR: C-AO device %.4f / python %.4f | hybrid device %.4f / python %.4f'
              % (row['mae_gt_cao_device'], row['mae_gt_cao_python'], row['mae_gt_hyb_device'], row['mae_gt_hyb_python']))
        print('    C-AO time/sample: mean %.1f p99 %.1f max %.1f us (load %.2f %%) | TCN: %d inferences, mean %.2f ms, max %.2f ms'
              % (row['cao_us_mean'], row['cao_us_p99'], row['cao_us_max'], row['cao_load_mean_pct'],
                 row['tcn_n_inferences'], row['tcn_ms_mean'], row['tcn_ms_max']))
        np.savez(os.path.join(REF, 'device_multi_%s_h%d_%s.npz' % (args.tag, args.hold, name)), rec=rec)
    out = os.path.join(REF, 'hil_multi_%s_h%d.csv' % (args.tag, args.hold))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('\nwrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM28')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--only', default=None)
    ap.add_argument('--tag', default='f32')
    ap.add_argument('--cpu-mhz', type=float, default=240.0)
    ap.add_argument('--multi', action='store_true', help='hybrid C-AO + TCN protocol')
    ap.add_argument('--hold', type=int, default=1, help='TCN update every N 110 Hz samples')
    ap.add_argument('--seconds', type=float, default=0.0, help='limit each recording (0 = all)')
    ap.add_argument('--chunk', type=int, default=128)
    ap.add_argument('--boot-timeout', type=float, default=15.0,
                    help='seconds to wait for the boot report (use ~3 for native-USB boards)')
    args = ap.parse_args()

    with open(os.path.join(REF, 'manifest.json')) as f:
        manifest = json.load(f)
    if args.only:
        wanted = set(args.only.split(','))
        manifest = [m for m in manifest if m['name'] in wanted]

    ser = serial.Serial(args.port, args.baud, timeout=10)
    try:
        ser.set_buffer_size(rx_size=1 << 18, tx_size=1 << 16)   # Windows driver buffers
    except Exception:
        pass
    # opening the port toggles DTR/RTS and resets the board: catch the boot report
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.setDTR(False); ser.setRTS(True); time.sleep(0.1); ser.setRTS(False)
    boot = wait_ready(ser, timeout=args.boot_timeout)
    info = cmd_info(ser)

    if args.multi:
        run_multi(ser, args, args.cpu_mhz)
        ser.close()
        return

    rows = []
    for m in manifest:
        name = m['name']
        z = np.load(os.path.join(REF, name + '.npz'))
        raw32 = z['raw32']
        fs = float(m['fs'])
        period_us = 1e6 / fs
        print('%s (%s, n=%d, %.0f s)' % (name, m['session'], len(raw32), len(raw32) / fs))
        for attempt in range(RETRIES):
            try:
                cmd_reset(ser)
                ph, om, fl, cy, ev = stream(ser, raw32)
                break
            except RuntimeError as exc:
                print('    serial framing error (%s); restarting recording (attempt %d/%d)'
                      % (exc, attempt + 2, RETRIES))
                time.sleep(1.5)            # > device's 1 s partial-frame timeout
                ser.reset_input_buffer()
        else:
            raise RuntimeError('recording %s failed %d times' % (name, RETRIES))

        ref = z['ref_phase']
        d = wrap(ph.astype(np.float64) - ref)
        gt = z['gt_phase']
        v = ~np.isnan(gt)
        mae_dev = np.mean(np.abs(wrap(ph[v].astype(np.float64) - gt[v])))
        mae_ref = np.mean(np.abs(wrap(ref[v] - gt[v])))
        dev_ev = np.flatnonzero(ev)
        ref_ev = z['ref_events']
        ev_same = len(dev_ev) == len(ref_ev) and np.array_equal(dev_ev, ref_ev)
        ev_shift = ''
        if not ev_same and len(dev_ev) == len(ref_ev):
            ev_shift = 'max shift %d samples' % int(np.max(np.abs(dev_ev - ref_ev)))
        us = cy.astype(np.float64) / args.cpu_mhz
        row = {
            'recording': name, 'session': m['session'], 'n': len(raw32), 'tag': args.tag,
            'max_abs_dphi_rad': float(np.max(np.abs(d))),
            'rms_dphi_rad': float(np.sqrt(np.mean(d ** 2))),
            'p99_abs_dphi_rad': float(np.percentile(np.abs(d), 99)),
            'max_abs_domega_rad_s': float(np.max(np.abs(om.astype(np.float64) - z['ref_omega']))),
            'max_abs_dfilt_V': float(np.max(np.abs(fl.astype(np.float64) - z['ref_filtered']))),
            'events_device': int(len(dev_ev)), 'events_ref': int(len(ref_ev)),
            'events_identical': bool(ev_same),
            'mae_vs_gt_device_rad': float(mae_dev), 'mae_vs_gt_python_rad': float(mae_ref),
            'us_mean': float(np.mean(us)), 'us_median': float(np.median(us)),
            'us_p99': float(np.percentile(us, 99)), 'us_max': float(np.max(us)),
            'load_mean_pct': float(100 * np.mean(us) / period_us),
            'load_max_pct': float(100 * np.max(us) / period_us),
        }
        rows.append(row)
        print('    dphi vs ref: max %.3e  rms %.3e  p99 %.3e rad | domega max %.2e | dfilt max %.2e'
              % (row['max_abs_dphi_rad'], row['rms_dphi_rad'], row['p99_abs_dphi_rad'],
                 row['max_abs_domega_rad_s'], row['max_abs_dfilt_V']))
        print('    events: device %d, ref %d, %s %s' % (len(dev_ev), len(ref_ev),
              'identical' if ev_same else 'DIFFERENT', ev_shift))
        print('    MAE vs FSR truth: device %.4f rad, python %.4f rad' % (mae_dev, mae_ref))
        print('    time/sample: mean %.2f  median %.2f  p99 %.2f  max %.2f us  -> load %.2f %% mean, %.2f %% max'
              % (row['us_mean'], row['us_median'], row['us_p99'], row['us_max'],
                 row['load_mean_pct'], row['load_max_pct']))
        np.savez(os.path.join(REF, 'device_%s_%s.npz' % (args.tag, name)),
                 phase=ph, omega=om, filtered=fl, cycles=cy, event=ev)

    out = os.path.join(REF, 'hil_%s.csv' % args.tag)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(REF, 'hil_%s_info.txt' % args.tag), 'w',
              encoding='utf-8', errors='replace') as f:
        # the first boot line is ROM-bootloader output at 115200 baud, i.e. garbage here
        f.write('\n'.join(ln for ln in boot + [info] if ln.isprintable()) + '\n')
    print('\nwrote', out)
    ser.close()


if __name__ == '__main__':
    main()
