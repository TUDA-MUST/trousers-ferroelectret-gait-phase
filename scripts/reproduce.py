"""Reproduce the paper's results from the public recordings.

Runs the four configurations (AO, C-AO, TCN, AO+TCN) on the five evaluated recordings
and prints the release values next to the values printed in the paper.

    python scripts/reproduce.py --data PATH/TO/Ferroelectret_FMG_Dataset
    python scripts/reproduce.py --data ... --out results      # also write per-stride CSVs

PATH is the extracted TUdatalib dataset (the folder containing data/, or a parent of it).
"""
import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaitphase as gp  # noqa: E402
from gaitphase import metrics as M  # noqa: E402
from gaitphase import tcn as T  # noqa: E402

METHODS = ('AO', 'C-AO', 'AO+TCN', 'TCN')
TWO_PI = 2 * np.pi


def evaluate_recording(root, name, registry, params, models, device, drop_first_sample):
    """Run all four configurations on one recording; return phases and reference."""
    rec = gp.load_recording(root, name, registry, drop_first_sample=drop_first_sample)
    fs, t = rec['fs'], rec['time']
    bp = {m: gp.bandpass_from_params(rec['muscles'][m], fs, params) for m in gp.MUSCLES}
    fr = params['fsr_reference']
    ref = gp.fsr_reference(t, rec['fsr'], fs, rec['meta']['fsr_threshold_v'], fr['lowpass_hz'], fr['order'])
    ao = gp.run_estimator([bp['G']], t, fs, params)
    cp = params['coupling']
    cao = gp.run_estimator([bp['G'], bp['VM'], bp['RF']], t, fs, params,
                           kappa=cp['kappa'], quality_weighted=cp['quality_weighted'])
    windows = T.network_input(bp['G'], bp['VM'], bp['RF'])
    phases = {
        'AO': ao['phase'],
        'C-AO': cao['phase'],
        'AO+TCN': T.predict_hybrid(models['hybrid'], windows, cao['phase'], device),
        'TCN': T.predict_standalone(models['standalone'], windows, len(t), device),
    }
    return rec, ref, ao, phases


def collect(root, registry, params, models, device, drop_first_sample, verbose=True):
    """Evaluate every recording and gather per-stride, temporal and per-recording results."""
    holdout = set(registry['holdout_speeds_kmh'])
    strides, temporal, per_rec = [], [], []
    for name, meta in registry['recordings'].items():
        t0 = time.time()
        rec, ref, ao, phases = evaluate_recording(root, name, registry, params, models, device, drop_first_sample)
        t, gt = rec['time'], ref['phase']
        segs = M.protocol_segments(meta)
        row = {'recording': name, 'session': meta['session'], 'training': meta['tcn_training']}
        for m in METHODS:
            ph = phases[m]
            cmae, crmse = M.continuous_errors(ph, gt)
            row[f'{m} cont_mae'] = cmae
            row[f'{m} cont_rmse'] = crmse
            for k, s, e, mae in M.per_stride_mae(ph, gt):
                seg = M.segment_of(0.5 * (float(t[s]) + float(t[e - 1])), segs)
                cond = seg['condition'] if seg else 'unknown'
                speed = seg['speed'] if seg else ''
                # held-out speeds: steady strides at 4.5 or 5.5 km/h of the two training ramps
                held = (meta['tcn_training'] and cond == 'steady' and '->' not in speed
                        and float(speed) in holdout)
                strides.append({'recording': name, 'session': meta['session'],
                                'training': meta['tcn_training'], 'stride': k, 'condition': cond,
                                'speed_kmh': speed, 'held_out_speed': held, 'method': m, 'mae_rad': mae})
            te = M.temporal_errors(ph, t, ref['hs_times'])
            if te is not None:
                for j in range(len(te['offset_s'])):
                    temporal.append({'recording': name, 'method': m, 'hs': int(te['hs_idx'][j]),
                                     'offset_ms': 1000 * te['offset_s'][j], 'offset_pct': te['offset_pct'][j]})
        # single-sensor AO metrics of Table 2
        row['AO dir_mae'] = M.direct_mae(phases['AO'], ref['hs_idx'])
        row['AO st_mae'] = M.st_mae(t, phases['AO'], ref['hs_idx'], ref['stride_times'])
        row['AO event_R'] = M.event_consistency(ao['event_times'], t, gt)
        per_rec.append(row)
        if verbose:
            print(f"  {name:<13} {meta['label']:<30} {len(t):6d} samples  "
                  f"{len(ref['hs_idx']):4d} heel strikes  ({time.time() - t0:.0f} s)", flush=True)
    return strides, temporal, per_rec


# ---------------------------------------------------------------------------- report
def mean_of(rows, **cond):
    v = [r['mae_rad'] for r in rows if all(r[k] == x for k, x in cond.items())]
    return (float(np.mean(v)) if v else float('nan')), len(v)


def line(label, paper, value, n='', dec=3, unit=''):
    ok = 'ok' if f'{value:.{dec}f}' == f'{paper:.{dec}f}' else 'DIFF'
    print(f"  {label:<48} {paper:>8.{dec}f}{unit} {value:>9.{dec}f}{unit} {str(n):>6}  {ok}")
    return ok == 'ok'


def report(strides, temporal, per_rec):
    checks = []
    hdr = f"  {'':<48} {'paper':>8} {'release':>10} {'n':>6}"

    print('\nHeadline numbers')
    print(hdr)
    paper_ps = {'AO': 0.163, 'C-AO': 0.156, 'AO+TCN': 0.108, 'TCN': 0.171}
    for m in METHODS:
        v, n = mean_of(strides, method=m)
        checks.append(line(f'per-stride MAE, pooled (rad)       {m}', paper_ps[m], v, n))
    for m, pm, pr in (('AO', 0.158, 0.290), ('AO+TCN', 0.112, 0.220)):
        checks.append(line(f'continuous MAE, recording mean (rad) {m}', pm,
                           np.mean([r[f'{m} cont_mae'] for r in per_rec]), len(per_rec)))
        checks.append(line(f'continuous RMSE, recording mean (rad) {m}', pr,
                           np.mean([r[f'{m} cont_rmse'] for r in per_rec]), len(per_rec)))
    for m, p in (('AO', 2.19), ('AO+TCN', 1.50)):
        v = [abs(r['offset_pct']) for r in temporal if r['method'] == m]
        checks.append(line(f'temporal MAE, pooled (% stride)    {m}', p, float(np.mean(v)), len(v), dec=2))
    for m, p in (('AO', 0.081), ('AO+TCN', 0.063), ('TCN', 0.049)):
        v, n = mean_of(strides, method=m, held_out_speed=True, training=True)
        checks.append(line(f'held-out speeds, per-stride MAE (rad) {m}', p, v, n))

    print('\nTable 2: single-sensor AO, mean +- SD across recordings (rad)')
    paper_t2 = {'S1': [(0.179, 0.111), (0.291, 0.147), (0.183, 0.126), (0.096, 0.019)],
                'S2': [(0.127, 0.001), (0.287, 0.005), (0.136, 0.004), (0.167, 0.007)],
                'All': [(0.158, 0.089), (0.290, 0.114), (0.164, 0.100), (0.124, 0.038)]}
    keys = [('AO cont_mae', 'Cont MAE'), ('AO cont_rmse', 'Cont RMSE'), ('AO dir_mae', 'Dir MAE'),
            ('AO st_mae', 'ST-MAE')]
    # Known deviation: S2 Dir MAE SD is 0.00344; the paper's 0.004 is consistent with rounding
    # a four-decimal intermediate value of 0.0035.
    known = {('S2', 'Dir MAE'): 'SD 0.00344; paper value consistent with rounding 0.0035'}
    for grp, sel in (('S1', lambda r: r['session'] == 'S1'), ('S2', lambda r: r['session'] == 'S2'),
                     ('All', lambda r: True)):
        rs = [r for r in per_rec if sel(r)]
        for (k, lab), (pm, psd) in zip(keys, paper_t2[grp]):
            v = np.array([r[k] for r in rs])
            ok = f'{v.mean():.3f}' == f'{pm:.3f}' and f'{v.std():.3f}' == f'{psd:.3f}'
            note = '' if ok else known.get((grp, lab), '')
            checks.append('known' if note else ok)
            print(f"  {grp:<4} {lab:<10} paper {pm:.3f} +- {psd:.3f}   release {v.mean():.3f} +- "
                  f"{v.std():.3f}  (n={len(rs)})  {'ok' if ok else 'DIFF'}"
                  + (f'  ({note})' if note else ''))

    print('\nTable 4: per-stride MAE by training exposure (rad)')
    paper_t4 = [('training recordings', dict(training=True), (0.127, 0.069, 0.051)),
                ('  held-out speeds', dict(training=True, held_out_speed=True), (0.081, 0.063, 0.049)),
                ('evaluation recordings', dict(training=False), (0.203, 0.150, 0.298)),
                ('S1, evaluation', dict(training=False, session='S1'), (0.216, 0.151, 0.353)),
                ('S2, evaluation', dict(training=False, session='S2'), (0.165, 0.147, 0.144))]
    print(f"  {'split':<22} {'n':>5}   {'AO':>13}   {'AO+TCN':>13}   {'TCN':>13}   (paper / release)")
    for lab, cond, paper in paper_t4:
        cells, n = [], 0
        for m, p in zip(('AO', 'AO+TCN', 'TCN'), paper):
            v, n = mean_of(strides, method=m, **cond)
            ok = f'{v:.3f}' == f'{p:.3f}'
            checks.append(ok)
            cells.append(f"{p:.3f} / {v:.3f}{'' if ok else '*'}")
        print(f"  {lab:<22} {n:>5}   " + '   '.join(f'{c:>13}' for c in cells))

    print('\nOther pooled values (not all printed in the paper)')
    for m in METHODS:
        v = [abs(r['offset_pct']) for r in temporal if r['method'] == m]
        s = [r['offset_pct'] for r in temporal if r['method'] == m]
        print(f"  {m:<7} temporal MAE {np.mean(v):.2f} %  (signed mean {np.mean(s):+.2f} %, SD {np.std(s):.2f} %, "
              f"n={len(v)} matched heel strikes)")
    for r in per_rec:
        print(f"  {r['recording']:<13} AO event consistency R = {r['AO event_R']:.3f}")

    n_ok = sum(c is True for c in checks)
    n_known = sum(c == 'known' for c in checks)
    print(f'\n{n_ok} of {len(checks)} compared values agree with the paper at the printed precision'
          + (f'; {n_known} differ{"s" if n_known == 1 else ""} for the reason printed next to '
             f'{"it" if n_known == 1 else "them"}.' if n_known else '.'))
    return n_ok + n_known == len(checks)


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='TUdatalib dataset folder (contains data/ or a parent of it)')
    ap.add_argument('--device', default='auto', help="torch device for the networks: auto, cpu or cuda")
    ap.add_argument('--keep-first-sample', action='store_true',
                    help='use the complete S2 files (the paper evaluation skipped their first sample)')
    ap.add_argument('--out', default=None, help='optional folder for per-stride and per-recording CSVs')
    args = ap.parse_args()

    import torch
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    root = gp.find_dataset_root(args.data)
    registry = gp.load_registry()
    params = gp.load_params()
    models = {k: T.load_model(k, device=device) for k in ('hybrid', 'standalone')}
    print(f'gaitphase {gp.__version__} | Python {platform.python_version()} | numpy {np.__version__} | '
          f'scipy {__import__("scipy").__version__} | torch {torch.__version__} | device {device}')
    print(f'dataset: {root}')
    if args.keep_first_sample:
        print('complete S2 files (first sample kept): values can differ slightly from the paper')
    t0 = time.time()
    strides, temporal, per_rec = collect(root, registry, params, models, device,
                                         False if args.keep_first_sample else None)
    print(f'evaluated in {time.time() - t0:.0f} s')
    all_ok = report(strides, temporal, per_rec)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / 'per_stride_mae.csv', strides)
        write_csv(out / 'temporal_errors.csv', temporal)
        write_csv(out / 'per_recording.csv', per_rec)
        print(f'wrote CSVs to {out}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
