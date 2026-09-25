/* ESP32 portability / real-time benchmark of the single-sensor gait phase estimator.
 *
 * On boot: runs the estimator over the embedded recording excerpt, reports per-sample
 * execution time (CPU cycle counter) and the deviation from the Python reference.
 * Then serves a binary serial protocol so tools/hil_validate.py can stream whole
 * recordings through the device and compare every output sample with the reference:
 *
 *   'I'                      -> one info line, '\n' terminated
 *   'Z'                      -> reset estimator state, replies "OK\n"
 *   'B'                      -> re-run the on-device self-test, text report, ends "END\n"
 *   'S' u16 n, n x float32   -> process n raw samples (volts); replies
 *                               'R' u16 n, n x { f32 phase, f32 omega, f32 filtered,
 *                                                u32 cycles, u32 event }
 *   'A'                      -> resend the last 'R' reply (host-side byte loss recovery)
 *   'T'                      -> TCN self-test (agreement with PyTorch + inference time), ends "END\n"
 *   'D'                      -> dual-core benchmark: AO on core 1 while TCN loops on core 0
 *   'H' u16 hold             -> TCN update every `hold` 110 Hz samples (1 = every), "OK\n"
 *   'M' u16 n, n x 3 float32 -> hybrid estimator (C-AO + TCN) on n samples of [G, VM, RF];
 *                               replies 'R' u16 n, n x { f32 cao_phase, f32 hybrid_phase,
 *                               f32 bp_G, f32 omega_G, u32 cycles_cao, u32 cycles_tcn,
 *                               u32 flags (bit0 event, bit1 TCN ran on this sample) }
 * A partial 'S' frame times out after 1 s and is discarded; the host waits longer than
 * that before resynchronising with 'Z'.
 * All multi-byte fields little-endian.
 */
#include <Arduino.h>
#include <math.h>
#include <string.h>
#include "board.h"

extern "C" {
#include "gp_estimator.h"
#include "gp_cao.h"
#include "gp_tcn.h"
}
#include "params_gen.h"
#include "excerpt_gen.h"
#include "tcn_weights_gen.h"
#include "tcn_test_gen.h"

#ifndef GP_SERIAL_BAUD
#define GP_SERIAL_BAUD 921600
#endif

static gp_estimator_t g_est;
static gp_estimator_params_t g_params;

/* Per-sample cycle histogram for the boot self-test (64-cycle bins up to 131k cycles):
 * exact min/max/mean, median and p99 to 0.27 us. Keeps RAM free for the TCN weights. */
#define HIST_BINS 2048
#define HIST_SHIFT 6
static uint16_t g_hist[HIST_BINS];

static uint32_t hist_quantile(uint32_t n, double q)
{
    uint32_t target = (uint32_t)(q * n), acc = 0;
    for (int b = 0; b < HIST_BINS; ++b) {
        acc += g_hist[b];
        if (acc > target) return ((uint32_t)b << HIST_SHIFT) + (1u << (HIST_SHIFT - 1));
    }
    return (HIST_BINS - 1) << HIST_SHIFT;
}

/* Buffers (heap; static DRAM is limited to ~160 kB on the ESP32) */
#define STREAM_MAX 512
static float *g_in;            /* [STREAM_MAX]      single-sensor stream input */
static uint8_t *g_out;         /* [STREAM_MAX * 20] single-sensor stream reply */
static float *g_in_m;          /* [STREAM_MAX * 3]  hybrid stream input */
static uint8_t *g_out_m;       /* [STREAM_MAX * 28] hybrid stream reply */

static void setup_params(void)
{
    g_params.fs = GPG_FS;
    g_params.sos_hp = GPG_SOS_HP; g_params.n_hp = GPG_N_HP;
    g_params.sos_lp = GPG_SOS_LP; g_params.n_lp = GPG_N_LP;
    g_params.ao.N = GPG_AO_N;
    g_params.ao.nu_phi = GPG_AO_NU_PHI;
    g_params.ao.nu_omega = GPG_AO_NU_OMEGA;
    g_params.ao.eta = GPG_AO_ETA;
    g_params.ao.omega_init = GPG_AO_OMEGA_INIT;
    g_params.ao.alpha_init = GPG_AO_ALPHA_INIT;
    g_params.ao.omega_min = GPG_AO_OMEGA_MIN;
    g_params.ev.env_alpha = GPG_EV_ENV_ALPHA;
    g_params.ev.k_env = GPG_EV_K_ENV;
    g_params.ev.k_std_bp = GPG_EV_K_STD_BP;
    g_params.ev.min_stride_fraction = GPG_EV_MIN_STRIDE_FRACTION;
    g_params.corr.kp = GPG_CORR_KP;
    g_params.corr.phase_offset = GPG_CORR_PHASE_OFFSET;
    g_params.corr.event_ref_phase = GPG_CORR_EVENT_REF_PHASE;
}

static inline uint32_t cycles(void) { return board_cycles(); }

static float wrap_pi_f(float x)
{
    x = fmodf(x + (float)M_PI, 2.0f * (float)M_PI);
    if (x < 0) x += 2.0f * (float)M_PI;
    return x - (float)M_PI;
}

/* Run the excerpt through a fresh estimator; report timing + reference deviation. */
static void self_test(void)
{
    gp_estimator_reset(&g_est);
    gp_output_t o;
    double sum_abs = 0.0, max_abs = 0.0, sum_sq = 0.0;
    uint32_t n_events = 0;
    const float f_cpu_mhz = (float)board_cpu_mhz();
    uint64_t total = 0;
    uint32_t c_min = 0xFFFFFFFFu, c_max = 0;
    memset(g_hist, 0, sizeof(g_hist));

    int64_t t_wall0 = board_micros();
    for (int i = 0; i < GPG_EXCERPT_N; ++i) {
        uint32_t c0 = cycles();
        gp_estimator_step(&g_est, (gp_real)GPG_EXCERPT_RAW[i], &o);
        uint32_t c1 = cycles();
        uint32_t dc = c1 - c0;
        total += dc;
        if (dc < c_min) c_min = dc;
        if (dc > c_max) c_max = dc;
        { uint32_t b = dc >> HIST_SHIFT; if (b >= HIST_BINS) b = HIST_BINS - 1; g_hist[b]++; }
        n_events += (uint32_t)o.event;
        float d = wrap_pi_f((float)o.phase - GPG_EXCERPT_REF_PHASE[i]);
        float ad = fabsf(d);
        sum_abs += ad; sum_sq += (double)d * d;
        if (ad > max_abs) max_abs = ad;
    }
    int64_t t_wall1 = board_micros();

    uint32_t c_med = hist_quantile(GPG_EXCERPT_N, 0.50);
    uint32_t c_p99 = hist_quantile(GPG_EXCERPT_N, 0.99);
    double c_mean = (double)total / GPG_EXCERPT_N;
    const double period_us = 1e6 / (double)GPG_FS;

    Serial.printf("SELFTEST %s  n=%d  real=%s  cpu=%.0f MHz\n",
                  GPG_EXCERPT_NAME, GPG_EXCERPT_N, GP_REAL_NAME, f_cpu_mhz);
    Serial.printf("  cycles/sample  min %u  median %u  mean %.0f  p99 %u  max %u\n",
                  c_min, c_med, c_mean, c_p99, c_max);
    Serial.printf("  us/sample      min %.2f  median %.2f  mean %.2f  p99 %.2f  max %.2f\n",
                  c_min / f_cpu_mhz, c_med / f_cpu_mhz, c_mean / f_cpu_mhz,
                  c_p99 / f_cpu_mhz, c_max / f_cpu_mhz);
    Serial.printf("  load @ %.0f Hz  mean %.2f %%  p99 %.2f %%  max %.2f %%  (period %.0f us)\n",
                  (double)GPG_FS, 100.0 * c_mean / f_cpu_mhz / period_us,
                  100.0 * c_p99 / f_cpu_mhz / period_us, 100.0 * c_max / f_cpu_mhz / period_us,
                  period_us);
    Serial.printf("  wall time      %.1f ms for %.1f s of signal (%.1fx real time)\n",
                  (t_wall1 - t_wall0) / 1000.0, GPG_EXCERPT_N / (double)GPG_FS,
                  (GPG_EXCERPT_N / (double)GPG_FS) * 1e6 / (double)(t_wall1 - t_wall0));
    Serial.printf("  vs python ref  MAE %.3e rad  RMSE %.3e rad  max %.3e rad  events %u (ref %d)\n",
                  sum_abs / GPG_EXCERPT_N, sqrt(sum_sq / GPG_EXCERPT_N), max_abs,
                  n_events, GPG_EXCERPT_N_EVENTS);
    gp_estimator_reset(&g_est);
}

/* ------------------------------------------------------------------ TCN ---- */
static gp_tcn_layer_t g_tcn_layers[TCN_N_LAYERS];
static gp_tcn_weights_t g_tcn_w;
static gp_tcn_t g_tcn;
static gp_tcn_work_t *g_tcn_work;         /* ~66 kB, heap */
static int g_tcn_ram_layers = 0;          /* per-tap weight blocks copied to SRAM */
static uint32_t g_tcn_ram_bytes = 0;

static int tcn_total_tapblocks(void)
{
    int n = 0;
    for (int l = 0; l < TCN_N_LAYERS; ++l) n += TCN_LAYERS[l].n_taps;
    return n;
}

static void tcn_setup(void)
{
    /* required buffers first, then the work planes, then as many weight layers in SRAM
     * as the heap allows (largest first: the memory-mapped flash behind the ESP32's
     * cache is what limits the inference time, not the multiply-accumulates) */
    g_in = (float *)malloc(sizeof(float) * STREAM_MAX);
    g_out = (uint8_t *)malloc((size_t)STREAM_MAX * 20);
    g_in_m = (float *)malloc(sizeof(float) * STREAM_MAX * 3);
    g_out_m = (uint8_t *)malloc((size_t)STREAM_MAX * 28);
    g_tcn_work = (gp_tcn_work_t *)malloc(sizeof(gp_tcn_work_t));
    if (!g_in || !g_out || !g_in_m || !g_out_m || !g_tcn_work) {
        Serial.println("FATAL: out of memory"); while (1) delay(1000);
    }
    for (int l = 0; l < TCN_N_LAYERS; ++l) {
        g_tcn_layers[l].n_in = TCN_LAYERS[l].n_in;
        g_tcn_layers[l].n_out = TCN_LAYERS[l].n_out;
        g_tcn_layers[l].n_taps = TCN_LAYERS[l].n_taps;
        g_tcn_layers[l].off = TCN_LAYERS[l].off;
        g_tcn_layers[l].w = TCN_LAYERS[l].w;
        g_tcn_layers[l].b = TCN_LAYERS[l].b;
    }
    g_tcn_w.layers = g_tcn_layers;
    g_tcn_w.gn_w = TCN_GN_W; g_tcn_w.gn_b = TCN_GN_B; g_tcn_w.gn_eps = TCN_GN_EPS;
    g_tcn_w.ha_w = TCN_HA_W; g_tcn_w.ha_b = TCN_HA_B;
    g_tcn_w.hd_w = TCN_HD_W; g_tcn_w.hd_b = TCN_HD_B;
    g_tcn_w.hc_w = TCN_HC_W; g_tcn_w.hc_b = TCN_HC_B;
    gp_tcn_init(&g_tcn, &g_tcn_w);            /* sets every wt[] into flash */
#ifndef GP_TCN_WEIGHTS_IN_FLASH
    /* stage as many per-tap weight blocks (9 kB for 48 filters) in SRAM as the heap
     * allows; the hidden layers first, they are read most */
    {
        static const int order[TCN_N_LAYERS] = { 1, 2, 3, 4, 5, 6, 0 };
        for (int k = 0; k < TCN_N_LAYERS; ++k) {
            int l = order[k];
            size_t nb = sizeof(float) * (size_t)TCN_LAYERS[l].n_out * TCN_LAYERS[l].n_in;
            for (int j = 0; j < TCN_LAYERS[l].n_taps; ++j) {
                float *ram = (float *)malloc(nb);
                if (!ram) continue;
                memcpy(ram, g_tcn_layers[l].wt[j], nb);
                g_tcn_layers[l].wt[j] = ram;
                g_tcn_ram_layers++;              /* counts tap blocks */
                g_tcn_ram_bytes += (uint32_t)nb;
            }
        }
    }
#endif
}

/* Exact windowed inference on the embedded real windows: agreement with PyTorch and
 * time per inference (the number that sets the achievable TCN update rate). */
static void tcn_self_test(void)
{
    const float f_cpu_mhz = (float)board_cpu_mhz();
    const int reps = 5;
    uint32_t c_min = 0xFFFFFFFFu, c_max = 0; uint64_t c_sum = 0; int n_meas = 0;
    float max_dev = 0.0f;

    Serial.printf("TCNTEST windows=%d  positions computed per layer:", TCN_TEST_N);
    for (int l = 0; l < GP_TCN_NL; ++l) Serial.printf(" %d", g_tcn.n_pos[l]);
    Serial.printf("  -> %u MAC/inference (full window would be %u)\n",
                  (unsigned)g_tcn.macs, (unsigned)(GP_TCN_WL * 58560u));
    for (int k = 0; k < TCN_TEST_N; ++k) {
        const float (*win)[GP_TCN_NCH] = (const float (*)[GP_TCN_NCH])(TCN_TEST_WIN + (size_t)k * GP_TCN_WL * GP_TCN_NCH);
        const float *ex = TCN_TEST_EXPECT + (size_t)k * 5;
        gp_tcn_out_t o;
        for (int r = 0; r < reps; ++r) {
            uint32_t c0 = cycles();
            gp_tcn_infer(&g_tcn, win, g_tcn_work, &o);
            uint32_t c1 = cycles();
            uint32_t d = c1 - c0;
            if (d < c_min) c_min = d;
            if (d > c_max) c_max = d;
            c_sum += d; ++n_meas;
        }
        float dev = 0.0f;
        dev = fmaxf(dev, fabsf(o.pa[0] - ex[0])); dev = fmaxf(dev, fabsf(o.pa[1] - ex[1]));
        dev = fmaxf(dev, fabsf(o.pd[0] - ex[2])); dev = fmaxf(dev, fabsf(o.pd[1] - ex[3]));
        dev = fmaxf(dev, fabsf(o.pc - ex[4]));
        if (dev > max_dev) max_dev = dev;
        Serial.printf("  win %d: pa=[%.4f %.4f] pd=[%.4f %.4f] pc=%.4f | torch [%.4f %.4f %.4f %.4f %.4f] | max dev %.2e\n",
                      k, o.pa[0], o.pa[1], o.pd[0], o.pd[1], o.pc, ex[0], ex[1], ex[2], ex[3], ex[4], dev);
    }
    double us_mean = (double)c_sum / n_meas / f_cpu_mhz;
    Serial.printf("  inference time  min %.2f ms  mean %.2f ms  max %.2f ms  (%d runs)\n",
                  c_min / f_cpu_mhz / 1000.0, us_mean / 1000.0, c_max / f_cpu_mhz / 1000.0, n_meas);
    Serial.printf("  -> max exact-window update rate %.1f Hz on one core; at 110 Hz this is %.0f %% load, at 22 Hz %.0f %%\n",
                  1e6 / us_mean, 100.0 * us_mean / (1e6 / 110.0), 100.0 * us_mean / (1e6 / 22.0));
    Serial.printf("  max |device - torch| over heads: %.2e\n", max_dev);
}

static bool read_exact(uint8_t *buf, size_t n, uint32_t timeout_ms);
static uint16_t g_last_n = 0;      /* size of the last reply, for 'A' (resend); bit 15 = 'M' reply */

/* --------------------------------------------------------- hybrid C-AO + TCN ---- */
static gp_cao_t g_cao;
static float g_win[GP_TCN_WL][GP_TCN_NCH];     /* sliding 110 Hz window, oldest first */
static gp_tcn_out_t g_heads;                   /* last TCN output (held between updates) */
static uint32_t g_ds_count = 0;                /* 110 Hz samples seen */
static uint16_t g_hold = 1;
static float g_hybrid = 0.0f;

static void hybrid_reset(void)
{
    gp_cao_reset(&g_cao);
    memset(g_win, 0, sizeof(g_win));
    memset(&g_heads, 0, sizeof(g_heads));
    g_ds_count = 0;
    g_hybrid = 0.0f;
}

static void hybrid_setup(void)
{
    gp_cao_init(&g_cao, &g_params, (gp_real)0.25, 1);
    hybrid_reset();
}

static void handle_multi(void)
{
    uint8_t hdr[2];
    if (!read_exact(hdr, 2, 1000)) return;
    uint16_t n = (uint16_t)(hdr[0] | (hdr[1] << 8));
    if (n == 0 || n > STREAM_MAX) return;
    if (!read_exact((uint8_t *)g_in_m, (size_t)n * 12, 2000)) return;

    gp_cao_output_t o;
    uint8_t *p = g_out_m;
    for (uint16_t i = 0; i < n; ++i) {
        gp_real raw[3] = { (gp_real)g_in_m[3 * i], (gp_real)g_in_m[3 * i + 1], (gp_real)g_in_m[3 * i + 2] };
        uint32_t c0 = cycles();
        gp_cao_step(&g_cao, raw, &o);
        uint32_t c1 = cycles();
        uint32_t cy_tcn = 0, flags = (uint32_t)o.event;
        if (g_cao.sample % 3 == 1) {            /* sample index 0, 3, 6, ... -> 110 Hz */
            /* shift the window by one and append the three bandpass values */
            memmove(&g_win[0][0], &g_win[1][0], sizeof(float) * (GP_TCN_WL - 1) * GP_TCN_NCH);
            for (int c = 0; c < GP_TCN_NCH; ++c) g_win[GP_TCN_WL - 1][c] = (float)o.filtered[c];
            if (g_ds_count % g_hold == 0) {
                uint32_t t0 = cycles();
                gp_tcn_infer(&g_tcn, g_win, g_tcn_work, &g_heads);
                cy_tcn = cycles() - t0;
                flags |= 2u;
            }
            g_hybrid = gp_tcn_blend(&g_heads, (float)o.phase);
            g_ds_count++;
        }
        float ph = (float)o.phase, bp0 = (float)o.filtered[0], om0 = (float)o.omega[0];
        uint32_t cy = c1 - c0;
        memcpy(p, &ph, 4); memcpy(p + 4, &g_hybrid, 4); memcpy(p + 8, &bp0, 4); memcpy(p + 12, &om0, 4);
        memcpy(p + 16, &cy, 4); memcpy(p + 20, &cy_tcn, 4); memcpy(p + 24, &flags, 4);
        p += 28;
    }
    g_last_n = 0;                              /* 'A' resend below uses the M buffer */
    uint8_t rh[3] = { 'R', hdr[0], hdr[1] };
    Serial.write(rh, 3);
    Serial.write(g_out_m, (size_t)n * 28);
    Serial.flush();
    g_last_n = n | 0x8000u;                    /* high bit: last reply was an 'M' reply */
}

/* ------------------------------------------------ dual-core deployment layout ---- */
/* The TCN runs as its own task pinned to core 0 (the core that otherwise idles when no
 * radio is used) while the sample-rate estimator stays on core 1. The benchmark
 * measures the single-sensor AO per-sample latency on core 1 WHILE core 0 runs
 * back-to-back TCN inferences, and reports the inference rate core 0 sustained. */
#if BOARD_HAS_SECOND_CORE
static volatile bool g_tcn_bg_run = false;
static volatile uint32_t g_tcn_bg_count = 0;
static volatile uint64_t g_tcn_bg_cycles = 0;
static volatile bool g_tcn_bg_done = false;

static void tcn_bg_task(void *arg)
{
    (void)arg;
    const float (*win)[GP_TCN_NCH] = (const float (*)[GP_TCN_NCH])TCN_TEST_WIN;
    gp_tcn_out_t o;
    while (g_tcn_bg_run) {
        uint32_t c0 = cycles();
        gp_tcn_infer(&g_tcn, win, g_tcn_work, &o);
        g_tcn_bg_cycles += (uint32_t)(cycles() - c0);
        g_tcn_bg_count++;
    }
    g_tcn_bg_done = true;
    vTaskDelete(NULL);
}

static void dual_core_test(void)
{
    const float f_cpu_mhz = (float)board_cpu_mhz();
    g_tcn_bg_count = 0; g_tcn_bg_cycles = 0; g_tcn_bg_done = false; g_tcn_bg_run = true;
    xTaskCreatePinnedToCore(tcn_bg_task, "tcn_bg", 6144, NULL, 1, NULL, 0);
    delay(100);                                   /* let the task reach steady state */
    g_tcn_bg_count = 0; g_tcn_bg_cycles = 0;      /* count only inside the timed window */
    int64_t t0 = board_micros();
    Serial.println("DUALCORE: TCN inference loop on core 0, AO self-test on core 1");
    self_test();
    int64_t t1 = board_micros();
    g_tcn_bg_run = false;
    while (!g_tcn_bg_done) delay(10);
    double ms_per = g_tcn_bg_count ? (double)g_tcn_bg_cycles / g_tcn_bg_count / f_cpu_mhz / 1000.0 : 0.0;
    Serial.printf("  core 0 during that window: %u TCN inferences in %.1f ms -> %.1f Hz sustained, %.2f ms each\n",
                  (unsigned)g_tcn_bg_count, (t1 - t0) / 1000.0,
                  g_tcn_bg_count * 1e6 / (double)(t1 - t0), ms_per);
}
#else
static void dual_core_test(void)
{
    Serial.println("DUALCORE: not applicable, single-core target (the TCN runs inline in the sample loop)");
}
#endif

static void print_info(void)
{
    char binfo[160];
    board_info(binfo, sizeof(binfo));
    Serial.printf("INFO %s cpu=%uMHz real=%s sizeof(est)=%u "
                  "sizeof(cao)=%u fs=%.1f ao_N=%d tcn_ram_tapblocks=%d/%d tcn_ram_kb=%u "
                  "heap_free=%u build=%s %s\n",
                  binfo, (unsigned)board_cpu_mhz(), GP_REAL_NAME,
                  (unsigned)sizeof(gp_estimator_t), (unsigned)sizeof(gp_cao_t),
                  (double)GPG_FS, GPG_AO_N, g_tcn_ram_layers, tcn_total_tapblocks(),
                  (unsigned)(g_tcn_ram_bytes / 1024), (unsigned)board_free_heap(),
                  __DATE__, __TIME__);
}

static bool read_exact(uint8_t *buf, size_t n, uint32_t timeout_ms)
{
    size_t got = 0;
    uint32_t t0 = millis();
    while (got < n) {
        int avail = Serial.available();
        if (avail > 0) {
            size_t k = Serial.readBytes((char *)buf + got, min((size_t)avail, n - got));
            got += k;
            t0 = millis();
        } else if (millis() - t0 > timeout_ms) {
            return false;
        }
    }
    return true;
}


static void send_reply(uint16_t n)
{
    uint8_t rh[3] = { 'R', (uint8_t)(n & 0xFF), (uint8_t)(n >> 8) };
    Serial.write(rh, 3);
    Serial.write(g_out, (size_t)n * 20);
    Serial.flush();
}

static void handle_stream(void)
{
    uint8_t hdr[2];
    if (!read_exact(hdr, 2, 1000)) return;
    uint16_t n = (uint16_t)(hdr[0] | (hdr[1] << 8));
    if (n == 0 || n > STREAM_MAX) return;
    if (!read_exact((uint8_t *)g_in, (size_t)n * 4, 1000)) return;

    gp_output_t o;
    uint8_t *p = g_out;
    for (uint16_t i = 0; i < n; ++i) {
        uint32_t c0 = cycles();
        gp_estimator_step(&g_est, (gp_real)g_in[i], &o);
        uint32_t c1 = cycles();
        float ph = (float)o.phase, om = (float)o.omega, fl = (float)o.filtered;
        uint32_t cy = c1 - c0, evf = (uint32_t)o.event;
        memcpy(p, &ph, 4); memcpy(p + 4, &om, 4); memcpy(p + 8, &fl, 4);
        memcpy(p + 12, &cy, 4); memcpy(p + 16, &evf, 4);
        p += 20;
    }
    g_last_n = n;
    send_reply(n);
}

void setup()
{
    board_serial_setup();
    Serial.begin(GP_SERIAL_BAUD);
    delay(300);
    setup_params();
    gp_estimator_init(&g_est, &g_params);
    tcn_setup();                       /* also allocates the self-test cycle buffer */
    hybrid_setup();
    Serial.println();
    Serial.println("=== gait phase estimator: ESP32 port ===");
    print_info();
    self_test();
    tcn_self_test();
    Serial.println("READY");
}

void loop()
{
    if (Serial.available() <= 0) return;
    int c = Serial.read();
    switch (c) {
    case 'I': print_info(); break;
    case 'Z': gp_estimator_reset(&g_est); hybrid_reset(); Serial.println("OK"); break;
    case 'H': {
        uint8_t h[2];
        if (read_exact(h, 2, 1000)) { g_hold = (uint16_t)(h[0] | (h[1] << 8)); if (!g_hold) g_hold = 1; }
        Serial.println("OK");
        break;
    }
    case 'M': handle_multi(); break;
    case 'B': self_test(); Serial.println("END"); break;
    case 'T': tcn_self_test(); Serial.println("END"); break;
    case 'D': dual_core_test(); Serial.println("END"); break;
    case 'S': handle_stream(); break;
    case 'A':                                               /* resend last reply */
        if (g_last_n & 0x8000u) {
            uint16_t n = g_last_n & 0x7FFFu;
            uint8_t rh[3] = { 'R', (uint8_t)(n & 0xFF), (uint8_t)(n >> 8) };
            Serial.write(rh, 3); Serial.write(g_out_m, (size_t)n * 28); Serial.flush();
        } else if (g_last_n) {
            send_reply(g_last_n);
        }
        break;
    default: break;   /* ignore stray bytes (line endings etc.) */
    }
}
