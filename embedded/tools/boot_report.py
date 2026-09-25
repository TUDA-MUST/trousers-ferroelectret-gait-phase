# -*- coding: utf-8 -*-
"""Reset the board, print its boot report (self-tests) and optionally run more commands.

    python boot_report.py [--port COM28] [--cmd T] [--cmd B]
"""
import sys
import time
import argparse
import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM28')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--cmd', action='append', default=[])
    ap.add_argument('--timeout', type=float, default=60.0)
    args = ap.parse_args()
    ser = serial.Serial(args.port, args.baud, timeout=2)
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.setDTR(False); ser.setRTS(True); time.sleep(0.1); ser.setRTS(False)
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        ln = ser.readline().decode('utf-8', 'replace').rstrip()
        if ln and ln.isprintable():
            print(ln)
        if ln.startswith('READY'):
            break
    for c in args.cmd:
        ser.write(c.encode('ascii'))
        t0 = time.time()
        while time.time() - t0 < args.timeout:
            ln = ser.readline().decode('utf-8', 'replace').rstrip()
            if ln:
                print(ln)
            if ln.startswith('END') or ln.startswith('INFO'):
                break
    ser.close()


if __name__ == '__main__':
    main()
