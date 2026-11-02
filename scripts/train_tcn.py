"""Retrain a paper network from the TUdatalib recordings (paper Section 2.8, Table 1).

Training data: the two 60 s speed ramps (S1 and S2). Two 70 s windows of each ramp (55 to
125 s and 175 to 245 s after the trim, on the nominal protocol grid) are removed from the
training set (not only zero-weighted), which leaves 42,795 windows at 110 Hz. On the measured
speed steps each window spans parts of two adjacent steps, so every speed also occurs in
training. The hybrid network learns the absolute phase, the
residual of the C-AO phase and a confidence; the stand-alone network only the absolute
phase.

    python scripts/train_tcn.py --data PATH --kind hybrid --out my_hybrid.pt
    python scripts/train_tcn.py --data PATH --kind standalone --out my_standalone.pt

Seed 0 uses the settings of the paper models (torch seed 0, augmentation seed 42).
GPU training is not bit-reproducible across hardware and library versions; retrained
networks therefore differ slightly from weights/*.pt (three training seeds of the
hybrid gave pooled per-stride MAEs of 0.100 to 0.108 rad). 300 epochs took about two
hours per network on the authors' workstation GPU.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaitphase as gp  # noqa: E402
from gaitphase import tcn as T  # noqa: E402

TWO_PI = 2 * np.pi


def holdout_mask(time_ds, meta, holdout_speeds, guard_s):
    """True for 110 Hz samples outside the two windows per ramp removed from training.

    The windows lie on the nominal protocol grid (step i at i * speed_duration_s after the
    trim), as for the paper networks. Because the trims cut into the first speed step
    (step_onset_s in params/recordings.json), they do not coincide with the 4.5 and
    5.5 km/h steps, and every speed also occurs in training."""
    keep = np.ones(len(time_ds), dtype=bool)
    if meta['protocol'] != 'ramp':
        return keep
    for i, v in enumerate(meta['speeds_kmh']):
        if v in holdout_speeds:
            a = i * meta['speed_duration_s'] - guard_s
            b = (i + 1) * meta['speed_duration_s'] + guard_s
            keep &= ~((time_ds >= a) & (time_ds < b))
    return keep


def build_training_set(root, registry, params, use_holdout=True):
    """Windows, targets and sample weights of the training recordings."""
    wins, gts, aos, valid, keep = [], [], [], [], []
    holdout = set(registry['holdout_speeds_kmh'])
    cp = params['coupling']
    for name, meta in registry['recordings'].items():
        if not meta['tcn_training']:
            continue
        rec = gp.load_recording(root, name, registry)
        fs, t = rec['fs'], rec['time']
        bp = {m: gp.bandpass_from_params(rec['muscles'][m], fs, params) for m in gp.MUSCLES}
        ref = gp.fsr_reference(t, rec['fsr'], fs, meta['fsr_threshold_v'])
        cao = gp.run_estimator([bp['G'], bp['VM'], bp['RF']], t, fs, params,
                               kappa=cp['kappa'], quality_weighted=cp['quality_weighted'])
        w = T.network_input(bp['G'], bp['VM'], bp['RF'])
        n = len(w)
        gt_ds = ref['phase'][::T.DS_FACTOR]
        v = (~np.isnan(gt_ds) & (gt_ds > 0)).astype(np.float32)
        v[:10] = 0                                    # warm-up of each recording
        wins.append(w)
        gts.append(gt_ds)
        aos.append(cao['phase'][::T.DS_FACTOR])
        valid.append(v)
        keep.append(holdout_mask(t[::T.DS_FACTOR][:n], meta, holdout, registry['holdout_guard_s'])
                    if use_holdout else np.ones(n, dtype=bool))
        print(f'  {name}: {n} windows, {keep[-1].sum()} kept', flush=True)
    k = np.concatenate(keep)
    return (np.concatenate(wins)[k], np.concatenate(gts)[k], np.concatenate(aos)[k],
            np.concatenate(valid)[k])


def train(model, wins, gt, ao, valid, multi_head, seed=0, epochs=300, device='cpu'):
    delta = (gt - ao + np.pi) % TWO_PI - np.pi
    y_abs = np.column_stack([np.sin(gt), np.cos(gt)]).astype(np.float32)
    y_add = np.column_stack([np.sin(delta), np.cos(delta)]).astype(np.float32)
    y_conf = (1.0 / (1.0 + np.abs(delta))).astype(np.float32)
    n = len(gt)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs,
                                                pct_start=0.1, anneal_strategy='cos')
    rng = np.random.RandomState(seed if seed != 0 else 42)      # augmentation RNG
    best_loss, best_state, t0 = np.inf, None, time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        total, nb = 0.0, 0
        for s in range(0, n, 1024):
            idx = perm[s:min(s + 1024, n)]
            xb = wins[idx].copy()
            for c in range(3):                                   # amplitude scaling, channel dropout
                xb[:, c, :] *= rng.uniform(0.05, 5.0)
                if rng.rand() < 0.1:
                    xb[:, c, :] = 0
            xb += (rng.randn(*xb.shape) * 0.01).astype(np.float32)
            x = torch.tensor(xb, device=device)
            w = torch.tensor(valid[idx], device=device)
            ya = torch.tensor(y_abs[idx], device=device)
            if multi_head:
                pa, pd, pc = model(x)
                yd = torch.tensor(y_add[idx], device=device)
                yc = torch.tensor(y_conf[idx], device=device)
                loss = (w[:, None] * (pa - ya) ** 2).sum() / max(w.sum(), 1) + \
                       (w[:, None] * (pd - yd) ** 2).sum() / max(w.sum(), 1) + \
                       0.5 * (w * (pc.squeeze() - yc) ** 2).sum() / max(w.sum(), 1)
            else:
                pa = model(x)
                loss = (w[:, None] * (pa - ya) ** 2).sum() / max(w.sum(), 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        mean_loss = total / max(nb, 1)
        if mean_loss < best_loss:                                # keep the lowest training loss
            best_loss = mean_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch {ep + 1}/{epochs}  loss {mean_loss:.6f}  ({time.time() - t0:.0f} s)', flush=True)
    model.load_state_dict(best_state)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='TUdatalib dataset folder')
    ap.add_argument('--kind', choices=('hybrid', 'standalone'), default='hybrid')
    ap.add_argument('--out', required=True, help='output state_dict (.pt)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--no-holdout', action='store_true', help='train on the complete ramps')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    root = gp.find_dataset_root(args.data)
    registry, params = gp.load_registry(), gp.load_params()
    print('building training set ...', flush=True)
    wins, gt, ao, valid = build_training_set(root, registry, params, use_holdout=not args.no_holdout)
    print(f'{len(gt)} training samples at 110 Hz', flush=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    arch = T.load_config()['architecture']
    cls = T.MultiHeadTCN if args.kind == 'hybrid' else T.AbsoluteTCN
    model = cls(nf=arch['n_filters'], n_in=arch['n_inputs'], dilations=tuple(arch['dilations']),
                k=arch['kernel_size'], dropout=arch['dropout']).to(device)
    print(f'{args.kind}: {sum(p.numel() for p in model.parameters())} parameters, device {device}', flush=True)
    model = train(model, wins, gt, ao, valid, args.kind == 'hybrid', args.seed, args.epochs, device)
    torch.save(model.state_dict(), args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
