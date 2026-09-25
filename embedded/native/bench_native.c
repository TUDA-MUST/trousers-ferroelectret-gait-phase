/* bench_native.c -- the boot self-tests of the firmware as a plain C program, for a
 * Linux target such as the Raspberry Pi Zero 2 W (or any PC with a C compiler).
 *
 * Runs the single-sensor estimator over the embedded 20 s excerpt and the exact
 * windowed TCN over the embedded real windows, compares with the Python / PyTorch
 * reference values, and reports execution time per sample / per inference measured
 * with CLOCK_MONOTONIC. Build with the Makefile next to this file:
 *
 *     make                      # gcc -O2, native
 *     make CFLAGS="-O2 -mcpu=cortex-a53"           # 64-bit Raspberry Pi OS
 *     ./bench_native
 *
 * A Linux process is not a hard real-time context; the max values include scheduler
 * preemption, so read the median and p99 for the algorithm's cost.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "gp_estimator.h"
#include "gp_cao.h"
#include "gp_tcn.h"
#include "params_gen.h"
#include "excerpt_gen.h"
#include "tcn_weights_gen.h"
#include "tcn_test_gen.h"

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static float wrap_pi_f(float x)
{
    x = fmodf(x + (float)M_PI, 2.0f * (float)M_PI);
    if (x < 0) x += 2.0f * (float)M_PI;
    return x - (float)M_PI;
}

static gp_estimator_params_t g_params;
static gp_tcn_layer_t g_layers[TCN_N_LAYERS];
static gp_tcn_weights_t g_w;
static gp_tcn_t g_tcn;
static gp_tcn_work_t g_work;

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

static void report_timing(const char *what, double *us, int n, double period_us)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < n; ++i) sum += us[i];
    qsort(us, n, sizeof(double), cmp_d);
    printf("  %-14s min %.2f  median %.2f  mean %.2f  p99 %.2f  max %.2f us",
           what, us[0], us[n / 2], sum / n, us[(n * 99) / 100], us[n - 1]);
    if (period_us > 0)
        printf("  -> load mean %.2f %%  p99 %.2f %%", 100.0 * (sum / n) / period_us,
               100.0 * us[(n * 99) / 100] / period_us);
    printf("\n");
}

static void self_test_ao(void)
{
    static double us[GPG_EXCERPT_N];
    gp_estimator_t est;
    gp_output_t o;
    double sum_abs = 0.0, max_abs = 0.0, sum_sq = 0.0;
    unsigned n_events = 0;
    int i;
    gp_estimator_init(&est, &g_params);
    for (i = 0; i < GPG_EXCERPT_N; ++i) {
        double t0 = now_us();
        gp_estimator_step(&est, (gp_real)GPG_EXCERPT_RAW[i], &o);
        us[i] = now_us() - t0;
        n_events += (unsigned)o.event;
        {
            float d = wrap_pi_f((float)o.phase - GPG_EXCERPT_REF_PHASE[i]);
            float ad = fabsf(d);
            sum_abs += ad; sum_sq += (double)d * d;
            if (ad > max_abs) max_abs = ad;
        }
    }
    printf("SELFTEST %s  n=%d  real=%s\n", GPG_EXCERPT_NAME, GPG_EXCERPT_N, GP_REAL_NAME);
    report_timing("AO us/sample", us, GPG_EXCERPT_N, 1e6 / (double)GPG_FS);
    printf("  vs python ref  MAE %.3e rad  RMSE %.3e rad  max %.3e rad  events %u (ref %d)\n",
           sum_abs / GPG_EXCERPT_N, sqrt(sum_sq / GPG_EXCERPT_N), max_abs, n_events,
           GPG_EXCERPT_N_EVENTS);
}

static void self_test_cao(void)
{
    /* the excerpt has one channel; feed it to all three to time the coupled estimator */
    static double us[GPG_EXCERPT_N];
    gp_cao_t cao;
    gp_cao_output_t o;
    int i;
    gp_cao_init(&cao, &g_params, (gp_real)0.25, 1);
    for (i = 0; i < GPG_EXCERPT_N; ++i) {
        gp_real raw[3] = { (gp_real)GPG_EXCERPT_RAW[i], (gp_real)GPG_EXCERPT_RAW[i], (gp_real)GPG_EXCERPT_RAW[i] };
        double t0 = now_us();
        gp_cao_step(&cao, raw, &o);
        us[i] = now_us() - t0;
    }
    printf("C-AO (3 channels, same signal on all three, timing only)\n");
    report_timing("C-AO us/sample", us, GPG_EXCERPT_N, 1e6 / (double)GPG_FS);
}

static void self_test_tcn(void)
{
    const int reps = 20;
    static double us[TCN_TEST_N * 20];
    float max_dev = 0.0f;
    int k, r, l, n = 0;
    for (l = 0; l < TCN_N_LAYERS; ++l) {
        g_layers[l].n_in = TCN_LAYERS[l].n_in;
        g_layers[l].n_out = TCN_LAYERS[l].n_out;
        g_layers[l].n_taps = TCN_LAYERS[l].n_taps;
        g_layers[l].off = TCN_LAYERS[l].off;
        g_layers[l].w = TCN_LAYERS[l].w;
        g_layers[l].b = TCN_LAYERS[l].b;
    }
    g_w.layers = g_layers;
    g_w.gn_w = TCN_GN_W; g_w.gn_b = TCN_GN_B; g_w.gn_eps = TCN_GN_EPS;
    g_w.ha_w = TCN_HA_W; g_w.ha_b = TCN_HA_B;
    g_w.hd_w = TCN_HD_W; g_w.hd_b = TCN_HD_B;
    g_w.hc_w = TCN_HC_W; g_w.hc_b = TCN_HC_B;
    gp_tcn_init(&g_tcn, &g_w);

    printf("TCNTEST windows=%d  positions computed per layer:", TCN_TEST_N);
    for (l = 0; l < GP_TCN_NL; ++l) printf(" %d", g_tcn.n_pos[l]);
    printf("  -> %u MAC/inference\n", (unsigned)g_tcn.macs);
    for (k = 0; k < TCN_TEST_N; ++k) {
        const float (*win)[GP_TCN_NCH] = (const float (*)[GP_TCN_NCH])(TCN_TEST_WIN + (size_t)k * GP_TCN_WL * GP_TCN_NCH);
        const float *ex = TCN_TEST_EXPECT + (size_t)k * 5;
        gp_tcn_out_t o;
        float dev = 0.0f;
        for (r = 0; r < reps; ++r) {
            double t0 = now_us();
            gp_tcn_infer(&g_tcn, win, &g_work, &o);
            us[n++] = now_us() - t0;
        }
        dev = fmaxf(dev, fabsf(o.pa[0] - ex[0])); dev = fmaxf(dev, fabsf(o.pa[1] - ex[1]));
        dev = fmaxf(dev, fabsf(o.pd[0] - ex[2])); dev = fmaxf(dev, fabsf(o.pd[1] - ex[3]));
        dev = fmaxf(dev, fabsf(o.pc - ex[4]));
        if (dev > max_dev) max_dev = dev;
        printf("  win %d: pa=[%.4f %.4f] pd=[%.4f %.4f] pc=%.4f | torch [%.4f %.4f %.4f %.4f %.4f] | max dev %.2e\n",
               k, o.pa[0], o.pa[1], o.pd[0], o.pd[1], o.pc, ex[0], ex[1], ex[2], ex[3], ex[4], dev);
    }
    report_timing("TCN us/infer", us, n, 1e6 / 110.0);
    printf("  max |device - torch| over heads: %.2e\n", max_dev);
}

int main(void)
{
    setup_params();
    printf("=== gait phase estimator: native benchmark (real=%s) ===\n", GP_REAL_NAME);
    self_test_ao();
    self_test_cao();
    self_test_tcn();
    return 0;
}
