/* gp_tcn.h -- exact windowed inference of the paper's multi-head TCN in plain C.
 *
 * Reproduces MultiHeadTCN.forward() on one 167-sample window of the three bandpass
 * channels at 110 Hz: GroupNorm(1,3) over the whole window, seven causal dilated
 * convolutions (BatchNorm folded into the weights at export), ReLU, residual sum from
 * the second layer on, and the three linear heads on the last position. Taps that
 * would reach before the window start are pruned at export (they read zeros).
 *
 * Only the positions each layer actually needs for the final output are computed:
 * the dependency masks are derived backwards from the last position at init, which
 * halves the work of the top layers without changing a single value.
 *
 * Weights are always float; activations are float regardless of gp_real, because the
 * network was trained and evaluated in float32.
 */
#ifndef GP_TCN_H
#define GP_TCN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GP_TCN_WL   167
#define GP_TCN_NCH  3
#define GP_TCN_NF   48
#define GP_TCN_NL   7

#define GP_TCN_MAX_TAPS 5

typedef struct {
    int n_in, n_out, n_taps;
    const int   *off;                  /* negative sample offsets, length n_taps (0 = current) */
    const float *w;                    /* [n_taps][n_out][n_in], contiguous (export layout) */
    const float *b;                    /* [n_out] */
    const float *wt[GP_TCN_MAX_TAPS];  /* per-tap [n_out][n_in] blocks; gp_tcn_init() points
                                          them into `w`, the caller may repoint individual
                                          taps at RAM copies (9 kB each for 48 filters) */
} gp_tcn_layer_t;

typedef struct {
    gp_tcn_layer_t *layers;         /* GP_TCN_NL entries (wt[] filled by gp_tcn_init) */
    const float *gn_w;              /* GroupNorm affine, [3] */
    const float *gn_b;
    float gn_eps;
    const float *ha_w, *ha_b;       /* absolute-phase head [2][48], [2] */
    const float *hd_w, *hd_b;       /* AO-delta head       [2][48], [2] */
    const float *hc_w, *hc_b;       /* confidence head     [1][48], [1] */
} gp_tcn_weights_t;

typedef struct {
    float pa[2];        /* [sin, cos] of absolute phase */
    float pd[2];        /* [sin, cos] of delta phase */
    float pc;           /* confidence in (0, 1) */
} gp_tcn_out_t;

typedef struct {
    gp_tcn_weights_t *w;                  /* layers' wt[] are filled in by gp_tcn_init */
    uint8_t need[GP_TCN_NL][GP_TCN_WL];   /* need[l][t] = layer l output at t is used */
    int n_pos[GP_TCN_NL];                 /* positions computed per layer (diagnostic) */
    uint32_t macs;                        /* multiply-accumulates per inference */
} gp_tcn_t;

/* Work buffers: two activation planes [WL][NF] plus the normalised input [WL][NCH].
 * Provided by the caller so the library allocates nothing (about 66 kB for 48 filters). */
typedef struct {
    float xn[GP_TCN_WL][GP_TCN_NCH];
    float h0[GP_TCN_WL][GP_TCN_NF];
    float h1[GP_TCN_WL][GP_TCN_NF];
} gp_tcn_work_t;

/* `w` is modified: each layer's wt[] is set to point into its `w` array. */
void gp_tcn_init(gp_tcn_t *t, gp_tcn_weights_t *w);

/* `win` is [WL][NCH], oldest sample first; the caller zero-fills positions before the
 * stream start. Returns the three head outputs for the last position. */
void gp_tcn_infer(const gp_tcn_t *t, const float win[GP_TCN_WL][GP_TCN_NCH],
                  gp_tcn_work_t *work, gp_tcn_out_t *out);

/* Confidence-weighted circular blend with the oscillator phase (Python _predict_blend). */
float gp_tcn_blend(const gp_tcn_out_t *o, float phi_ao);

#ifdef __cplusplus
}
#endif
#endif /* GP_TCN_H */
