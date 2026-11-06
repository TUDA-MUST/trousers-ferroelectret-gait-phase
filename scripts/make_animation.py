"""Render the README animations: from ferroelectret signals to continuous gait phase.

cold start: the single-sensor AO from a cold start on the first seconds of one recording.
    Draws, sample by sample, the three raw sensor signals, the bandpass M. Gastrocnemius
    signal with the oscillator's reconstruction and the detected events, and the corrected
    gait phase against the FSR reference.
hybrid: AO+TCN on a later window of the same recording. Draws the three bandpass signals
    with the network's causal input window, the correction the network adds to the phase
    of the coupled oscillators, and the phase error of both against the FSR reference.

Both show the instrumented trousers (docs/media/trousers_side.png) and a phase dial.

    python scripts/make_animation.py --data PATH/TO/Ferroelectret_FMG_Dataset
    python scripts/make_animation.py --data ... --which hybrid --hybrid-start 114 --seconds 12

Writes ferro_to_phase_{light,dark}.gif and hybrid_{light,dark}.gif. Needs matplotlib and
Pillow in addition to requirements.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gaitphase as gp  # noqa: E402
from gaitphase import tcn as T  # noqa: E402
from gaitphase.correction import CorrectionParams, PhaseErrorLearner  # noqa: E402
from gaitphase.events import DetectorParams, EventDetector  # noqa: E402
from gaitphase.metrics import wrap  # noqa: E402
from gaitphase.oscillator import AdaptiveOscillator, OscillatorParams  # noqa: E402

TWO_PI = 2 * np.pi
TOE_OFF = 0.6  # approximate toe-off as fraction of the gait cycle, for the dial only
MEDIA = Path(__file__).resolve().parents[1] / 'docs' / 'media'
PHOTO = MEDIA / 'trousers_side.png'
NAMES = {'G': 'M. Gastrocnemius', 'VM': 'M. Vastus Medialis', 'RF': 'M. Rectus Femoris'}
# sensor sites in the photo as (x, y) fractions of its width and height; the
# M. Vastus Medialis sensor sits on the inner side of the thigh, hidden in this view
SITES = {'RF': (0.741, 0.229), 'VM': (0.745, 0.347), 'G': (0.653, 0.623)}

THEMES = {
    'light': dict(bg='#ffffff', fg='#1f2328', muted='#6e7781', faint='#d0d7de', grid='#eaeef2',
                  est='#0a4a8c', osc='#8db6d2', ref='#b51212', signal='#57606a',
                  stance='#c9d6e3', swing='#eef2f6', window='#dbe7f3'),
    'dark': dict(bg='#0d1117', fg='#e6edf3', muted='#8b949e', faint='#30363d', grid='#21262d',
                 est='#58a6ff', osc='#2b5a86', ref='#ff7b72', signal='#a6aeb7',
                 stance='#2d4560', swing='#1c2430', window='#17304d'),
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


def speed_during(meta, t0, t1):
    """Treadmill speed if [t0, t1] lies within one speed step of the protocol, else None."""
    if meta['protocol'] == 'const':
        return meta['speeds_kmh'][0]
    onset, dur = meta.get('step_onset_s', 0.0), meta['speed_duration_s']
    k0, k1 = int((t0 - onset) // dur), int((t1 - onset) // dur)
    return meta['speeds_kmh'][k0] if k0 == k1 and 0 <= k0 < len(meta['speeds_kmh']) else None


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
    v = speed_during(rec['meta'], t[0], t[-1])
    context = (f'treadmill walking at {v:.1f} km/h, estimator starts cold' if v is not None
               else 'treadmill walking, estimator starts cold')
    return dict(t=t, fs=fs, raw={m: normalize(rec['muscles'][m][:n]) for m in gp.MUSCLES},
                bp=bp['G'], ref=ref['phase'][:n], ao=ao, context=context)


def prepare_hybrid(root, name, start, seconds, params):
    """AO+TCN as in scripts/reproduce.py, run from the start of the recording and cut to
    [start, start + seconds]."""
    rec = gp.load_recording(root, name)
    fs = rec['fs']
    n = int(round((start + seconds) * fs))
    t = rec['time'][:n] - rec['time'][0]
    fr = params['fsr_reference']
    ref = gp.fsr_reference(rec['time'], rec['fsr'], fs, rec['meta']['fsr_threshold_v'],
                           fr['lowpass_hz'], fr['order'])['phase'][:n]
    bp = [gp.bandpass_from_params(rec['muscles'][m], fs, params)[:n] for m in gp.MUSCLES]
    cp = params['coupling']
    cao = gp.run_estimator(bp, t, fs, params, kappa=cp['kappa'], quality_weighted=cp['quality_weighted'])
    model = T.load_model('hybrid')
    windows = T.network_input(*bp)
    pa, pd, pc = T.run_heads(model, windows)
    hyb_ds = T.blend(cao['phase'][::T.DS_FACTOR], pa, pd, pc)
    hyb = T.hold_upsample(hyb_ds, n)
    assert np.array_equal(hyb, T.predict_hybrid(model, windows, cao['phase'])), 'hybrid differs'
    corr = T.hold_upsample(wrap(hyb_ds - cao['phase'][::T.DS_FACTOR]), n)
    conf = T.hold_upsample(pc.squeeze(), n)
    err_c, err_h = np.abs(wrap(cao['phase'] - ref)), np.abs(wrap(hyb - ref))
    k = int(round(fs))  # mean absolute error over the last second, for the info text
    roll = lambda e: np.convolve(e, np.ones(k) / k)[:n]  # noqa: E731
    s = slice(int(round(start * fs)), n)
    v = speed_during(rec['meta'], t[s][0], t[s][-1])
    context = (f'treadmill walking at {v:.1f} km/h, {start:.0f} s into the recording' if v is not None
               else f'treadmill walking, {start:.0f} s into the recording')
    return dict(t=t[s], fs=fs, bp={m: normalize(x[s]) for m, x in zip(gp.MUSCLES, bp)},
                ref=ref[s], cao=cao['phase'][s], hyb=hyb[s], corr=corr[s], conf=conf[s],
                mae_c=roll(err_c)[s], mae_h=roll(err_h)[s], context=context)


def to_gif_frames(frames, hex_colors, dither_box=None):
    """Quantize RGB frames to one shared palette. The theme colors get exact palette
    entries, and pixels of exactly those colors are mapped to them (Pillow's palette
    lookup is approximate, which would tint the background). The photo inside
    dither_box is dithered instead."""
    from matplotlib.colors import to_rgb
    from PIL import Image
    palette = frames[-1].quantize(colors=192, method=Image.Quantize.MEDIANCUT)
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
        if dither_box is not None:
            x0, y0, x1, y1 = dither_box
            idx[y0:y1, x0:x1] = np.array(fr.crop(dither_box).quantize(palette=palette,
                                                                      dither=Image.Dither.FLOYDSTEINBERG))
        for col, k in exact:
            idx[(rgb == col).all(axis=2)] = k
        im = Image.fromarray(idx, mode='P')
        im.putpalette(palette.getpalette())
        out.append(im)
    return out


class Scene:
    """Shared layout: trousers photo, three stacked time-series rows and a phase dial."""

    def __init__(self, theme, t, titles, active, dpi=100):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        self.plt = plt
        self.c = c = THEMES[theme]
        installed = {f.name for f in font_manager.fontManager.ttflist}
        families = [f for f in ('Segoe UI', 'Helvetica Neue', 'Arial') if f in installed] + ['DejaVu Sans']
        plt.rcParams.update({
            'font.family': families, 'font.size': 9.5,
            'text.color': c['fg'], 'axes.labelcolor': c['muted'], 'xtick.color': c['muted'],
            'ytick.color': c['muted'], 'axes.edgecolor': c['faint'], 'figure.facecolor': c['bg'],
            'axes.facecolor': c['bg'], 'savefig.facecolor': c['bg'], 'mathtext.fontset': 'dejavusans',
        })
        self.fig = fig = plt.figure(figsize=(11.6, 5.2), dpi=dpi)
        left, right, h = 0.335, 0.735, 0.205
        self.rows = [fig.add_axes([left, y, right - left, h]) for y in (0.715, 0.435, 0.155)]
        for ax, title in zip(self.rows, titles):
            ax.set_xlim(t[0], t[-1])
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.tick_params(length=3, width=0.6, labelsize=8.5)
            ax.grid(axis='x', color=c['grid'], lw=0.6)
            ax.set_axisbelow(True)
            ax.text(0, 1.07, title, transform=ax.transAxes, fontsize=10.5, fontweight='semibold',
                    color=c['fg'], va='bottom')
        for ax in self.rows[:2]:
            ax.tick_params(labelbottom=False)
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        self.rows[2].set_xlabel('time (s)', labelpad=3)
        self.cursors = [ax.axvline(t[0], color=c['faint'], lw=1.0, zorder=0) for ax in self.rows]
        self.legend_kw = dict(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=3, frameon=False,
                              fontsize=8.5, handlelength=1.5, columnspacing=1.1, handletextpad=0.5,
                              labelcolor=c['muted'], borderaxespad=0)
        self._photo(fig.add_axes([0.0, 0.02, 0.215, 0.96]), active)
        self._dial(fig.add_axes([0.755, 0.26, 0.235, 0.56]))
        self.clock = fig.text(0.8725, 0.9, '', ha='center', va='bottom', fontsize=10.5,
                              fontweight='semibold', color=c['fg'])
        self.context = fig.text(0.8725, 0.875, '', ha='center', va='top', fontsize=8, color=c['muted'])
        self.info = fig.text(0.8725, 0.215, '', ha='center', va='top', fontsize=9, color=c['muted'],
                             linespacing=1.55)
        self.frames = []

    def channel_labels(self, ax, offsets, colors):
        for m, y in offsets.items():
            ax.text(-0.015, y, NAMES[m], transform=ax.get_yaxis_transform(), ha='right', va='center',
                    fontsize=8.5, color=colors[m])

    def _photo(self, ax, active):
        from matplotlib.patches import Circle
        from PIL import Image
        c = self.c
        img = np.asarray(Image.open(PHOTO).convert('RGBA')).astype(float) / 255
        h, w = img.shape[:2]
        rows = np.arange(h)
        img[..., 3] *= np.clip(np.minimum(rows / (0.1 * h), (h - 1 - rows) / (0.22 * h)), 0, 1)[:, None]
        ax.imshow(img, interpolation='antialiased')
        ax.axis('off')
        self.photo_ax = ax
        self.flash = {}
        for m, (fx, fy) in SITES.items():
            x, y, r = fx * w, fy * h, 0.065 * w
            col = c['est'] if m in active else c['muted']
            self.flash[m] = ax.add_patch(Circle((x, y), r, facecolor=c['est'], edgecolor='none', alpha=0))
            ax.add_patch(Circle((x, y), r, fill=False, edgecolor=col, lw=1.8, ls=(0, (2, 1.5)) if m == 'VM' else '-'))
            ax.plot([x + r, 0.97 * w], [y, y], color=col, lw=0.7, clip_on=False)
            label = NAMES[m] + ('\n(inner side)' if m == 'VM' else '')
            ax.text(w, y, label, ha='left', va='center', fontsize=8, color=col, clip_on=False,
                    linespacing=1.2)

    def _dial(self, ax):
        """Heel strike at the top, phase runs clockwise."""
        from matplotlib.patches import Wedge
        c = self.c
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ang = lambda p: np.pi / 2 - p  # noqa: E731
        deg = lambda p: np.degrees(ang(p))  # noqa: E731
        ax.add_patch(Wedge((0, 0), 1.0, deg(TOE_OFF * TWO_PI), deg(0), width=0.2,
                           facecolor=c['stance'], edgecolor='none'))
        ax.add_patch(Wedge((0, 0), 1.0, deg(TWO_PI), deg(TOE_OFF * TWO_PI), width=0.2,
                           facecolor=c['swing'], edgecolor='none'))
        for frac in (0.0, TOE_OFF):
            a = ang(frac * TWO_PI)
            ax.plot([0.74 * np.cos(a), 1.06 * np.cos(a)], [0.74 * np.sin(a), 1.06 * np.sin(a)],
                    color=c['muted'], lw=1.1)
        ax.text(0, 1.12, 'heel strike', ha='center', va='bottom', fontsize=8.5, color=c['muted'])
        a = ang(TOE_OFF * TWO_PI)
        ax.text(1.12 * np.cos(a), 1.12 * np.sin(a), 'toe-off', ha='right', va='top', fontsize=8.5,
                color=c['muted'])
        for frac, label, ha in ((TOE_OFF / 2, 'stance', 'left'), ((1 + TOE_OFF) / 2, 'swing', 'right')):
            a = ang(frac * TWO_PI)
            ax.text(1.1 * np.cos(a), 1.1 * np.sin(a), label, ha=ha, va='center', fontsize=8.5,
                    color=c['muted'])
        self.dial, self.ang = ax, ang
        self.ref_mark, = ax.plot([], [], 'o', ms=10, mfc='none', mec=c['ref'], mew=2.0, zorder=4)

    def hand(self, color, lw, ms, z):
        """A dial hand; returns a setter taking the phase."""
        line, = self.dial.plot([], [], color=color, lw=lw, solid_capstyle='round', zorder=z)
        tip, = self.dial.plot([], [], 'o', ms=ms, color=color, zorder=z + 1)
        self.dial.plot(0, 0, 'o', ms=5.5, color=color, zorder=z + 1)

        def set_phase(p, r=0.8):
            a = self.ang(p)
            line.set_data([0, r * np.cos(a)], [0, r * np.sin(a)])
            tip.set_data([r * np.cos(a)], [r * np.sin(a)])
        return set_phase

    def set_ref(self, p):
        a = self.ang(p)
        self.ref_mark.set_data([0.9 * np.cos(a)], [0.9 * np.sin(a)])

    def grab(self, tf):
        from PIL import Image
        for cur in self.cursors:
            cur.set_xdata([tf, tf])
        self.clock.set_text(f't = {tf:.1f} s')
        self.fig.canvas.draw()
        self.frames.append(Image.frombuffer('RGBA', self.fig.canvas.get_width_height(),
                                            self.fig.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1).convert('RGB'))

    def save(self, path, fps, hold_s=2.5):
        H = self.fig.canvas.get_width_height()[1]
        x0, y0, x1, y1 = self.photo_ax.get_window_extent().extents
        box = (int(x0), int(H - y1), int(np.ceil(x1)), int(np.ceil(H - y0)))
        self.plt.close(self.fig)
        gif = to_gif_frames(self.frames, [self.c[k] for k in ('bg', 'fg', 'muted', 'est', 'ref', 'osc', 'signal')],
                            dither_box=box)
        durations = [int(round(1000 / fps))] * len(gif)
        durations[-1] = int(hold_s * 1000)
        gif[0].save(path, save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
        return path


def render(d, theme, path, fps=15):
    """Cold start of the single-sensor AO."""
    sc = Scene(theme, d['t'], ('1  Sensor signals', '2  Oscillator and events', '3  Gait phase'),
               active={'G'})
    c = sc.c
    ax_raw, ax_bp, ax_ph = sc.rows
    t, ao = d['t'], d['ao']
    est, osc, ref = break_wraps(ao['phase']), break_wraps(ao['phi_ao']), break_wraps(d['ref'])

    # 1: raw signals, stacked, M. Gastrocnemius highlighted as the oscillator input
    offsets = {'G': 0.0, 'VM': -1.3, 'RF': -2.6}
    colors = {m: c['est'] if m == 'G' else c['muted'] for m in offsets}
    raw_lines = {m: ax_raw.plot([], [], color=colors[m], lw=1.2 if m == 'G' else 0.9)[0] for m in offsets}
    sc.channel_labels(ax_raw, offsets, colors)
    ax_raw.set_ylim(-3.2, 0.6)

    # 2: bandpass signal, oscillator reconstruction, events
    bp_line, = ax_bp.plot([], [], color=c['signal'], lw=1.0, label='bandpass signal')
    vh_line, = ax_bp.plot([], [], color=c['est'], lw=1.7, label='oscillator fit')
    ev_t = ao['event_times']
    ev_y = np.interp(ev_t, t, d['bp'])
    ev_pts, = ax_bp.plot([], [], 'v', color=c['fg'], ms=6.5, mec=c['bg'], mew=0.7, label='gait event')
    lo, hi = np.percentile(d['bp'], [0.5, 99.5])
    pad = 0.1 * (hi - lo)
    ax_bp.set_ylim(lo - 1.5 * pad, hi + pad)
    ax_bp.legend(**sc.legend_kw)
    ax_bp.text(-0.015, 0.5, NAMES['G'], transform=ax_bp.transAxes, ha='right', va='center',
               fontsize=8.5, color=c['est'])

    # 3: phase
    ref_line, = ax_ph.plot([], [], color=c['ref'], lw=1.3, ls=(0, (3, 2)), label='FSR reference')
    osc_line, = ax_ph.plot([], [], color=c['osc'], lw=1.0, label='oscillator phase')
    est_line, = ax_ph.plot([], [], color=c['est'], lw=1.9, label='estimate')
    ax_ph.set_ylim(-0.25, TWO_PI + 0.25)
    ax_ph.set_yticks([0, np.pi, TWO_PI], ['0', r'$\pi$', r'$2\pi$'])
    ax_ph.set_ylabel('rad', labelpad=4)
    ax_ph.legend(**sc.legend_kw)

    est_hand = sc.hand(c['est'], 3.2, 7.5, 5)
    sc.context.set_text(d['context'])
    for f in range(int(np.floor(t[-1] * fps)) + 1):
        tf = min(f / fps, t[-1])
        i = int(np.searchsorted(t, tf, side='right'))
        for m, line in raw_lines.items():
            line.set_data(t[:i], d['raw'][m][:i] + offsets[m])
        bp_line.set_data(t[:i], d['bp'][:i])
        vh_line.set_data(t[:i], ao['v_hat'][:i])
        shown = ev_t <= tf
        ev_pts.set_data(ev_t[shown], ev_y[shown])
        since = tf - ev_t[shown][-1] if shown.any() else np.inf
        sc.flash['G'].set_alpha(0.75 * max(0.0, 1 - since / 0.4))
        ref_line.set_data(t[:i], ref[:i])
        osc_line.set_data(t[:i], osc[:i])
        est_line.set_data(t[:i], est[:i])
        j = max(i - 1, 0)
        pe, pr = ao['phase'][j], d['ref'][j]
        est_hand(pe)
        sc.set_ref(pr)
        err = abs(wrap(pe - pr))
        sc.info.set_text(f'estimate {pe:.2f} rad, {100 * pe / TWO_PI:.0f} % of stride\n'
                         f'error to FSR reference {err:.2f} rad\n'
                         f'stride frequency {ao["omega"][j] / TWO_PI:.2f} Hz')
        sc.grab(tf)
    return sc.save(path, fps)


def render_hybrid(d, theme, path, fps=15):
    """AO+TCN: network input window, correction and phase error."""
    sc = Scene(theme, d['t'], ('1  Network input', '2  Network correction', '3  Phase error'),
               active=set(gp.MUSCLES))
    c = sc.c
    ax_in, ax_corr, ax_err = sc.rows
    t = d['t']
    t0 = t[0]
    win_s = T.WINDOW * T.DS_FACTOR / d['fs']

    # 1: the three bandpass signals and the causal window the network reads
    offsets = {'G': 0.0, 'VM': -1.3, 'RF': -2.6}
    sc.channel_labels(ax_in, offsets, {m: c['est'] for m in offsets})
    in_lines = {m: ax_in.plot([], [], color=c['est'], lw=1.0)[0] for m in offsets}
    ax_in.set_ylim(-3.2, 0.75)
    from matplotlib.patches import Rectangle
    box = ax_in.add_patch(Rectangle((t0, -3.15), 0, 3.85, facecolor=c['window'], edgecolor=c['est'],
                                    lw=0.8, zorder=0))
    box_label = ax_in.text(t0, 0.72, f'{win_s:.2f} s window', ha='right', va='bottom', fontsize=8,
                           color=c['est'])

    # 2: correction the network adds to the coupled-oscillator phase
    corr_line, = ax_corr.plot([], [], color=c['est'], lw=1.7, label='added to the oscillator phase')
    ax_corr.axhline(0, color=c['faint'], lw=0.8, zorder=0)
    lo, hi = min(0.0, np.percentile(d['corr'], 1)), np.percentile(d['corr'], 99)
    ax_corr.set_ylim(lo - 0.15 * (hi - lo) - 0.05, hi + 0.25 * (hi - lo))
    ax_corr.spines['left'].set_visible(True)
    ax_corr.set_yticks([0, 0.1])
    ax_corr.tick_params(labelleft=True)
    ax_corr.set_ylabel('rad', labelpad=4)
    ax_corr.legend(**sc.legend_kw)

    # 3: phase error to the FSR reference, at the 110 Hz network rate
    ds = slice(None, None, T.DS_FACTOR)
    e_c, e_h = wrap(d['cao'] - d['ref'])[ds], wrap(d['hyb'] - d['ref'])[ds]
    td = t[ds]
    ax_err.axhline(0, color=c['ref'], lw=1.3, ls=(0, (3, 2)), label='FSR reference')
    err_c, = ax_err.plot([], [], color=c['osc'], lw=1.2, label='coupled oscillators')
    err_h, = ax_err.plot([], [], color=c['est'], lw=1.9, label='AO+TCN')
    lim = np.percentile(np.abs(np.concatenate([e_c, e_h])), 99.5) * 1.15
    ax_err.set_ylim(-lim, lim)
    ax_err.set_ylabel('rad', labelpad=4)
    ax_err.legend(**sc.legend_kw)

    cao_hand = sc.hand(c['osc'], 2.4, 6, 4)
    hyb_hand = sc.hand(c['est'], 3.2, 7.5, 5)
    sc.context.set_text(d['context'])
    for f in range(int(np.floor((t[-1] - t0) * fps)) + 1):
        tf = min(t0 + f / fps, t[-1])
        i = int(np.searchsorted(t, tf, side='right'))
        for m, line in in_lines.items():
            line.set_data(t[:i], d['bp'][m][:i] + offsets[m])
        box.set_x(tf - win_s)
        box.set_width(win_s)
        box_label.set_x(tf)
        corr_line.set_data(t[:i], d['corr'][:i])
        k = int(np.searchsorted(td, tf, side='right'))
        err_c.set_data(td[:k], e_c[:k])
        err_h.set_data(td[:k], e_h[:k])
        j = max(i - 1, 0)
        cao_hand(d['cao'][j])
        hyb_hand(d['hyb'][j])
        sc.set_ref(d['ref'][j])
        sc.info.set_text(f'mean error over the last second\n'
                         f'AO+TCN {d["mae_h"][j]:.2f} rad\n'
                         f'coupled oscillators {d["mae_c"][j]:.2f} rad')
        sc.grab(tf)
    return sc.save(path, fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='extracted TUdatalib dataset')
    ap.add_argument('--which', choices=('cold-start', 'hybrid', 'both'), default='both')
    ap.add_argument('--recording', default='Ramp_30s', help='name in params/recordings.json')
    ap.add_argument('--seconds', type=float, default=12.0, help='length of each animation')
    ap.add_argument('--hybrid-start', type=float, default=114.0,
                    help='start of the AO+TCN window, in seconds from the start of the recording')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--out', default=str(MEDIA))
    args = ap.parse_args()

    params = gp.load_params()
    root = gp.find_dataset_root(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.which in ('cold-start', 'both'):
        jobs.append(('ferro_to_phase', render, prepare(root, args.recording, args.seconds, params)))
    if args.which in ('hybrid', 'both'):
        jobs.append(('hybrid', render_hybrid,
                     prepare_hybrid(root, args.recording, args.hybrid_start, args.seconds, params)))
    for stem, fn, d in jobs:
        for theme in THEMES:
            p = fn(d, theme, out / f'{stem}_{theme}.gif', fps=args.fps)
            print(f'wrote {p} ({p.stat().st_size / 1e6:.1f} MB)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
