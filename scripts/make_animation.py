"""Render the README animation: from ferroelectret signals to continuous gait phase.

Runs the single-sensor AO from a cold start on the first seconds of one recording and
draws, sample by sample, the three raw sensor signals, the bandpass M. Gastrocnemius
signal with the oscillator's reconstruction and the detected events, and the corrected
gait phase against the FSR reference, next to a phase dial.

    python scripts/make_animation.py --data PATH/TO/Ferroelectret_FMG_Dataset
    python scripts/make_animation.py --data ... --recording Ramp_30s --seconds 12 --out docs/media

Writes ferro_to_phase_light.gif and ferro_to_phase_dark.gif. Needs matplotlib and
Pillow in addition to requirements.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaitphase as gp  # noqa: E402
from gaitphase.correction import CorrectionParams, PhaseErrorLearner  # noqa: E402
from gaitphase.events import DetectorParams, EventDetector  # noqa: E402
from gaitphase.oscillator import AdaptiveOscillator, OscillatorParams  # noqa: E402

TWO_PI = 2 * np.pi
TOE_OFF = 0.6  # approximate toe-off as fraction of the gait cycle, for the dial only

THEMES = {
    'light': dict(bg='#ffffff', fg='#1f2328', muted='#6e7781', faint='#d0d7de', grid='#eaeef2',
                  est='#0a4a8c', osc='#8db6d2', ref='#b51212', signal='#57606a',
                  stance='#c9d6e3', swing='#eef2f6'),
    'dark': dict(bg='#0d1117', fg='#e6edf3', muted='#8b949e', faint='#30363d', grid='#21262d',
                 est='#58a6ff', osc='#2b5a86', ref='#ff7b72', signal='#a6aeb7',
                 stance='#2d4560', swing='#1c2430'),
}


def run_ao_traced(x, time, fs, params):
    """Single-sensor AO as in gaitphase.run_estimator, also recording the oscillator's
    reconstruction of its input and the detector threshold."""
    ao = AdaptiveOscillator(OscillatorParams.from_params(params))
    detector = EventDetector(DetectorParams.from_params(params))
    learner = PhaseErrorLearner(CorrectionParams.from_params(params))
    n, dt = len(time), 1.0 / fs
    out = {k: np.zeros(n) for k in ('phase', 'phi_ao', 'omega', 'v_hat', 'threshold')}
    for i in range(n):
        ao.step(x[i], dt)
        if detector.step(x[i], time[i], ao.stride_period) is not None:
            learner.on_event(ao.phase, time[i])
        learner.step(ao.omega, time[i], dt)
        out['phase'][i] = learner.corrected_phase(ao.phase)
        out['phi_ao'][i] = ao.phase
        out['omega'][i] = ao.omega
        out['v_hat'][i] = ao.alpha_0 + np.sum(ao.alpha * np.sin(ao.phi))
        out['threshold'][i] = detector.threshold
    out['event_times'] = np.asarray(detector.event_times)
    return out


def break_wraps(phase):
    """Copy of a phase trace with NaN after each wrap, so lines do not jump back to 0."""
    p = phase.copy()
    p[1:][np.diff(phase) < -np.pi] = np.nan
    return p


def normalize(x):
    lo, hi = np.percentile(x, [1, 99])
    return (x - 0.5 * (lo + hi)) / (hi - lo)


def context_label(meta, seconds):
    """Walking speed shown in the animation, if the shown part has a single speed."""
    first = meta['speed_duration_s'] + meta.get('step_onset_s', 0.0)
    if meta['protocol'] == 'const' or seconds <= first:
        return f"treadmill walking at {meta['speeds_kmh'][0]:.1f} km/h, estimator starts cold"
    return 'treadmill walking, estimator starts cold'


def prepare(root, name, seconds, params):
    rec = gp.load_recording(root, name)
    fs = rec['fs']
    n = int(round(seconds * fs))
    t = rec['time'][:n] - rec['time'][0]
    fr = params['fsr_reference']
    ref = gp.fsr_reference(rec['time'], rec['fsr'], fs, rec['meta']['fsr_threshold_v'],
                           fr['lowpass_hz'], fr['order'])
    bp = {m: gp.bandpass_from_params(rec['muscles'][m], fs, params)[:n] for m in gp.MUSCLES}
    ao = run_ao_traced(bp['G'], t, fs, params)
    check = gp.run_estimator([bp['G']], t, fs, params)
    assert np.array_equal(check['phase'], ao['phase']), 'traced AO differs from run_estimator'
    return dict(t=t, fs=fs, raw={m: normalize(rec['muscles'][m][:n]) for m in gp.MUSCLES},
                bp=bp['G'], ref=ref['phase'][:n], ao=ao, context=context_label(rec['meta'], seconds))


def to_gif_frames(frames, hex_colors):
    """Quantize RGB frames to one shared palette. The theme colors get exact palette
    entries, and pixels of exactly those colors are mapped to them (Pillow's palette
    lookup is approximate, which would tint the background)."""
    from matplotlib.colors import to_rgb
    from PIL import Image
    palette = frames[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    pal = np.array(palette.getpalette(), dtype=float).reshape(-1, 3)
    exact = []
    for h in hex_colors:
        rgb = np.round(255 * np.array(to_rgb(h)))
        k = int(np.argmin(((pal - rgb) ** 2).sum(1)))
        pal[k] = rgb
        exact.append((rgb.astype(np.uint8), k))
    palette.putpalette(pal.astype(np.uint8).ravel().tolist())
    out = []
    for fr in frames:
        q = fr.quantize(palette=palette, dither=Image.Dither.NONE)
        rgb, idx = np.asarray(fr), np.array(q)
        for col, k in exact:
            idx[(rgb == col).all(axis=2)] = k
        im = Image.fromarray(idx, mode='P')
        im.putpalette(palette.getpalette())
        out.append(im)
    return out


def render(d, theme, path, fps=15, hold_s=2.5, dpi=100):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Wedge
    from PIL import Image

    c = THEMES[theme]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ('Segoe UI', 'Helvetica Neue', 'Arial') if f in installed] + ['DejaVu Sans']
    plt.rcParams.update({
        'font.family': families, 'font.size': 9.5,
        'text.color': c['fg'], 'axes.labelcolor': c['muted'], 'xtick.color': c['muted'],
        'ytick.color': c['muted'], 'axes.edgecolor': c['faint'], 'figure.facecolor': c['bg'],
        'axes.facecolor': c['bg'], 'savefig.facecolor': c['bg'], 'mathtext.fontset': 'dejavusans',
    })
    t, ao = d['t'], d['ao']
    T = t[-1]
    est, osc, ref = break_wraps(ao['phase']), break_wraps(ao['phi_ao']), break_wraps(d['ref'])

    fig = plt.figure(figsize=(9.6, 5.2), dpi=dpi)
    left, right, h = 0.155, 0.655, 0.205
    ax_raw = fig.add_axes([left, 0.715, right - left, h])
    ax_bp = fig.add_axes([left, 0.435, right - left, h])
    ax_ph = fig.add_axes([left, 0.155, right - left, h])
    ax_dial = fig.add_axes([0.695, 0.26, 0.29, 0.56])
    for ax in (ax_raw, ax_bp, ax_ph):
        ax.set_xlim(0, T)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=3, width=0.6, labelsize=8.5)
        ax.grid(axis='x', color=c['grid'], lw=0.6)
        ax.set_axisbelow(True)
    for ax in (ax_raw, ax_bp):
        ax.tick_params(labelbottom=False)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
    for ax, s in ((ax_raw, '1  Sensor signals'), (ax_bp, '2  Oscillator and events'),
                  (ax_ph, '3  Gait phase')):
        ax.text(0, 1.07, s, transform=ax.transAxes, fontsize=10.5, fontweight='semibold',
                color=c['fg'], va='bottom')
    legend_kw = dict(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False, fontsize=8.5,
                     handlelength=1.5, columnspacing=1.1, handletextpad=0.5, labelcolor=c['muted'],
                     borderaxespad=0)

    # 1: raw signals, stacked, M. Gastrocnemius highlighted as the oscillator input
    channels = (('G', 'M. Gastrocnemius'), ('VM', 'M. Vastus Medialis'),
                ('RF', 'M. Rectus Femoris'))
    raw_lines = []
    for k, (m, label) in enumerate(channels):
        col = c['est'] if m == 'G' else c['muted']
        raw_lines.append(ax_raw.plot([], [], color=col, lw=1.2 if m == 'G' else 0.9)[0])
        ax_raw.text(-0.015, -1.3 * k, label, transform=ax_raw.get_yaxis_transform(), ha='right',
                    va='center', fontsize=8.5, color=col)
    ax_raw.set_ylim(-2.6 - 0.6, 0.6)

    # 2: bandpass signal, oscillator reconstruction, events
    bp_line, = ax_bp.plot([], [], color=c['signal'], lw=1.0, label='bandpass signal')
    vh_line, = ax_bp.plot([], [], color=c['est'], lw=1.7, label='oscillator fit')
    ev_t = ao['event_times']
    ev_y = np.interp(ev_t, t, d['bp'])
    ev_pts, = ax_bp.plot([], [], 'v', color=c['fg'], ms=6.5, mec=c['bg'], mew=0.7, label='gait event')
    lo, hi = np.percentile(d['bp'], [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    ax_bp.set_ylim(lo - 1.5 * pad, hi + pad)
    ax_bp.legend(**legend_kw)
    ax_bp.text(-0.015, 0.5, channels[0][1], transform=ax_bp.transAxes, ha='right', va='center',
               fontsize=8.5, color=c['est'])

    # 3: phase
    ref_line, = ax_ph.plot([], [], color=c['ref'], lw=1.3, ls=(0, (3, 2)), label='FSR reference')
    osc_line, = ax_ph.plot([], [], color=c['osc'], lw=1.0, label='oscillator phase')
    est_line, = ax_ph.plot([], [], color=c['est'], lw=1.9, label='estimate')
    ax_ph.set_ylim(-0.25, TWO_PI + 0.25)
    ax_ph.set_yticks([0, np.pi, TWO_PI], ['0', r'$\pi$', r'$2\pi$'])
    ax_ph.set_ylabel('rad', labelpad=4)
    ax_ph.set_xlabel('time (s)', labelpad=3)
    ax_ph.legend(**legend_kw)

    cursors = [ax.axvline(0, color=c['faint'], lw=1.0, zorder=0) for ax in (ax_raw, ax_bp, ax_ph)]

    # dial: heel strike at the top, phase runs clockwise
    ax_dial.set_xlim(-1.5, 1.5)
    ax_dial.set_ylim(-1.5, 1.5)
    ax_dial.set_aspect('equal')
    ax_dial.axis('off')
    ang = lambda p: np.pi / 2 - p  # noqa: E731
    deg = lambda p: np.degrees(ang(p))  # noqa: E731
    ax_dial.add_patch(Wedge((0, 0), 1.0, deg(TOE_OFF * TWO_PI), deg(0), width=0.2,
                            facecolor=c['stance'], edgecolor='none'))
    ax_dial.add_patch(Wedge((0, 0), 1.0, deg(TWO_PI), deg(TOE_OFF * TWO_PI), width=0.2,
                            facecolor=c['swing'], edgecolor='none'))
    for frac in (0.0, TOE_OFF):
        a = ang(frac * TWO_PI)
        ax_dial.plot([0.74 * np.cos(a), 1.06 * np.cos(a)], [0.74 * np.sin(a), 1.06 * np.sin(a)],
                     color=c['muted'], lw=1.1)
    ax_dial.text(0, 1.12, 'heel strike', ha='center', va='bottom', fontsize=8.5, color=c['muted'])
    a = ang(TOE_OFF * TWO_PI)
    ax_dial.text(1.12 * np.cos(a), 1.12 * np.sin(a), 'toe-off', ha='right', va='top', fontsize=8.5,
                 color=c['muted'])
    for frac, label, ha in ((TOE_OFF / 2, 'stance', 'left'), ((1 + TOE_OFF) / 2, 'swing', 'right')):
        a = ang(frac * TWO_PI)
        ax_dial.text(1.1 * np.cos(a), 1.1 * np.sin(a), label, ha=ha, va='center', fontsize=8.5,
                     color=c['muted'])
    ref_hand, = ax_dial.plot([], [], 'o', ms=10, mfc='none', mec=c['ref'], mew=2.0, zorder=4)
    est_hand, = ax_dial.plot([], [], color=c['est'], lw=3.2, solid_capstyle='round', zorder=5)
    est_tip, = ax_dial.plot([], [], 'o', ms=7.5, color=c['est'], zorder=6)
    ax_dial.plot(0, 0, 'o', ms=5.5, color=c['est'], zorder=6)
    clock = fig.text(0.84, 0.9, '', ha='center', va='bottom', fontsize=10.5, fontweight='semibold',
                     color=c['fg'])
    fig.text(0.84, 0.875, d['context'], ha='center', va='top', fontsize=8, color=c['muted'])
    info = fig.text(0.84, 0.215, '', ha='center', va='top', fontsize=9, color=c['muted'], linespacing=1.55)

    n_frames = int(np.floor(T * fps)) + 1
    frames = []
    for f in range(n_frames):
        tf = min(f / fps, T)
        i = int(np.searchsorted(t, tf, side='right'))
        for k, (line, (m, _)) in enumerate(zip(raw_lines, channels)):
            line.set_data(t[:i], d['raw'][m][:i] - 1.3 * k)
        bp_line.set_data(t[:i], d['bp'][:i])
        vh_line.set_data(t[:i], ao['v_hat'][:i])
        shown = ev_t <= tf
        ev_pts.set_data(ev_t[shown], ev_y[shown])
        ref_line.set_data(t[:i], ref[:i])
        osc_line.set_data(t[:i], osc[:i])
        est_line.set_data(t[:i], est[:i])
        for cur in cursors:
            cur.set_xdata([tf, tf])
        j = max(i - 1, 0)
        pe, pr = ao['phase'][j], d['ref'][j]
        a_e, a_r = ang(pe), ang(pr)
        est_hand.set_data([0, 0.8 * np.cos(a_e)], [0, 0.8 * np.sin(a_e)])
        est_tip.set_data([0.8 * np.cos(a_e)], [0.8 * np.sin(a_e)])
        ref_hand.set_data([0.9 * np.cos(a_r)], [0.9 * np.sin(a_r)])
        err = abs((pe - pr + np.pi) % TWO_PI - np.pi)
        info.set_text(f'estimate {pe:.2f} rad, {100 * pe / TWO_PI:.0f} % of stride\n'
                      f'error to FSR reference {err:.2f} rad\n'
                      f'stride frequency {ao["omega"][j] / TWO_PI:.2f} Hz')
        clock.set_text(f't = {tf:.1f} s')
        fig.canvas.draw()
        frames.append(Image.frombuffer('RGBA', fig.canvas.get_width_height(),
                                       fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1).convert('RGB'))
    plt.close(fig)

    gif = to_gif_frames(frames, [c[k] for k in ('bg', 'fg', 'muted', 'est', 'ref', 'osc', 'signal')])
    durations = [int(round(1000 / fps))] * len(gif)
    durations[-1] = int(hold_s * 1000)
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='extracted TUdatalib dataset')
    ap.add_argument('--recording', default='Ramp_30s', help='name in params/recordings.json')
    ap.add_argument('--seconds', type=float, default=12.0, help='length from the start of the recording')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'docs' / 'media'))
    args = ap.parse_args()

    params = gp.load_params()
    d = prepare(gp.find_dataset_root(args.data), args.recording, args.seconds, params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        p = render(d, theme, out / f'ferro_to_phase_{theme}.gif', fps=args.fps)
        print(f'wrote {p} ({p.stat().st_size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
