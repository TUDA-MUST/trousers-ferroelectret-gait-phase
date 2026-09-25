#include "gp_tcn.h"
#include <math.h>
#include <string.h>

static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

void gp_tcn_init(gp_tcn_t *t, gp_tcn_weights_t *w)
{
    int l, tt, j;
    t->w = w;
    for (l = 0; l < GP_TCN_NL; ++l) {
        gp_tcn_layer_t *L = &w->layers[l];
        for (j = 0; j < GP_TCN_MAX_TAPS; ++j)
            L->wt[j] = (j < L->n_taps) ? L->w + (size_t)j * L->n_out * L->n_in : NULL;
    }
    memset(t->need, 0, sizeof(t->need));
    /* the heads read only the last position of the last layer */
    t->need[GP_TCN_NL - 1][GP_TCN_WL - 1] = 1;
    /* layer l at position tt needs layer l-1 at tt + off[j] for every live tap, and,
     * through the residual connection, at tt itself (offset 0 is always a live tap). */
    for (l = GP_TCN_NL - 1; l >= 1; --l) {
        const gp_tcn_layer_t *L = &w->layers[l];
        for (tt = 0; tt < GP_TCN_WL; ++tt) {
            if (!t->need[l][tt]) continue;
            t->need[l - 1][tt] = 1;
            for (j = 0; j < L->n_taps; ++j) {
                int src = tt + L->off[j];
                if (src >= 0) t->need[l - 1][src] = 1;
            }
        }
    }
    t->macs = 0;
    for (l = 0; l < GP_TCN_NL; ++l) {
        const gp_tcn_layer_t *L = &w->layers[l];
        int n = 0;
        for (tt = 0; tt < GP_TCN_WL; ++tt) n += t->need[l][tt];
        t->n_pos[l] = n;
        t->macs += (uint32_t)n * (uint32_t)L->n_out * (uint32_t)L->n_in * (uint32_t)L->n_taps;
    }
    t->macs += GP_TCN_NF * 5;
}

/* One dilated causal conv layer + ReLU (+ residual for l >= 1), computed only where needed.
 * Output channel is the OUTER loop so that one channel's weights (n_taps rows of n_in
 * floats) stay in cache while all positions are swept: weights that live in
 * memory-mapped flash on the ESP32 are then read once per inference. */
static void conv_layer(const gp_tcn_layer_t *L, const uint8_t *need,
                       const float *in, int in_stride,      /* in[t*in_stride + c] */
                       const float *res,                     /* residual plane or NULL */
                       float *out)                           /* out[t*NF + o] */
{
    const int n_in = L->n_in, n_out = L->n_out, n_taps = L->n_taps;
    int tt, o, j, c;
    const int n4 = n_in & ~3;
    for (o = 0; o < n_out; ++o) {
        const float bo = L->b[o];
        for (tt = 0; tt < GP_TCN_WL; ++tt) {
            /* four independent accumulators: the FPU's multiply-add latency, not its
             * throughput, bounds a single serial sum */
            float a0 = bo, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            if (!need[tt]) continue;
            for (j = 0; j < n_taps; ++j) {
                int src = tt + L->off[j];
                const float *wj, *irow;
                if (src < 0) continue;                               /* zero padding */
                wj = L->wt[j] + (size_t)o * n_in;                    /* tap j, channel o */
                irow = in + src * in_stride;
                for (c = 0; c < n4; c += 4) {
                    a0 += wj[c] * irow[c];
                    a1 += wj[c + 1] * irow[c + 1];
                    a2 += wj[c + 2] * irow[c + 2];
                    a3 += wj[c + 3] * irow[c + 3];
                }
                for (; c < n_in; ++c) a0 += wj[c] * irow[c];
            }
            a0 = relu((a0 + a1) + (a2 + a3));
            if (res) a0 += res[tt * GP_TCN_NF + o];
            out[tt * GP_TCN_NF + o] = a0;
        }
    }
}

void gp_tcn_infer(const gp_tcn_t *t, const float win[GP_TCN_WL][GP_TCN_NCH],
                  gp_tcn_work_t *work, gp_tcn_out_t *out)
{
    const gp_tcn_weights_t *w = t->w;
    int tt, c, l, o;

    /* GroupNorm(1, 3): one mean / biased variance over all channels and positions */
    {
        double s = 0.0, ss = 0.0;
        const int n = GP_TCN_WL * GP_TCN_NCH;
        for (tt = 0; tt < GP_TCN_WL; ++tt)
            for (c = 0; c < GP_TCN_NCH; ++c) { double v = win[tt][c]; s += v; ss += v * v; }
        {
            float mean = (float)(s / n);
            float var = (float)(ss / n - (s / n) * (s / n));
            float inv = 1.0f / sqrtf(var + w->gn_eps);
            for (tt = 0; tt < GP_TCN_WL; ++tt)
                for (c = 0; c < GP_TCN_NCH; ++c)
                    work->xn[tt][c] = (win[tt][c] - mean) * inv * w->gn_w[c] + w->gn_b[c];
        }
    }

    /* layer 0 reads the normalised input, no residual */
    {
        float *cur = &work->h0[0][0], *nxt = &work->h1[0][0];
        conv_layer(&w->layers[0], t->need[0], &work->xn[0][0], GP_TCN_NCH, NULL, cur);
        for (l = 1; l < GP_TCN_NL; ++l) {
            conv_layer(&w->layers[l], t->need[l], cur, GP_TCN_NF, cur, nxt);
            { float *tmp = cur; cur = nxt; nxt = tmp; }
        }
        /* heads on the last position */
        {
            const float *f = cur + (GP_TCN_WL - 1) * GP_TCN_NF;
            float z;
            for (o = 0; o < 2; ++o) {
                float a = w->ha_b[o], d = w->hd_b[o];
                for (c = 0; c < GP_TCN_NF; ++c) {
                    a += w->ha_w[o * GP_TCN_NF + c] * f[c];
                    d += w->hd_w[o * GP_TCN_NF + c] * f[c];
                }
                out->pa[o] = a; out->pd[o] = d;
            }
            z = w->hc_b[0];
            for (c = 0; c < GP_TCN_NF; ++c) z += w->hc_w[c] * f[c];
            out->pc = 1.0f / (1.0f + expf(-z));
        }
    }
}

float gp_tcn_blend(const gp_tcn_out_t *o, float phi_ao)
{
    const float two_pi = 6.28318530717958647692f;
    float ph_abs = atan2f(o->pa[0], o->pa[1]);
    float ph_delta = atan2f(o->pd[0], o->pd[1]);
    float add = phi_ao + ph_delta;
    float sb, cb, r;
    ph_abs = fmodf(ph_abs, two_pi); if (ph_abs < 0) ph_abs += two_pi;
    add = fmodf(add, two_pi);       if (add < 0) add += two_pi;
    sb = o->pc * sinf(add) + (1.0f - o->pc) * sinf(ph_abs);
    cb = o->pc * cosf(add) + (1.0f - o->pc) * cosf(ph_abs);
    r = atan2f(sb, cb);
    r = fmodf(r, two_pi); if (r < 0) r += two_pi;
    return r;
}
