/* gp_ao.h -- adaptive oscillator (Righetti 2006 / Yan 2017), RK4 integration.
 *
 * Port of AdaptiveOscillator in adaptive_oscillator_pipeline.py with integrator='rk4'.
 * State: N harmonic phases phi[n], amplitudes alpha[n], offset alpha_0, fundamental
 * frequency omega. The phases are wrapped to [0, 2*pi) after every step; the dynamics
 * only ever read sin/cos of phi, so this is exact in real arithmetic and keeps
 * single-precision resolution constant over long recordings (the Python reference
 * leaves phi unbounded in float64).
 */
#ifndef GP_AO_H
#define GP_AO_H

#include "gp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GP_AO_MAX_N 8

typedef struct {
    int N;              /* number of harmonics */
    gp_real nu_phi;     /* phase learning rate */
    gp_real nu_omega;   /* frequency learning rate */
    gp_real eta;        /* amplitude learning rate */
    gp_real omega_init; /* initial fundamental frequency, rad/s (reference: 2*pi*0.9) */
    gp_real alpha_init; /* initial harmonic amplitude (reference: 0.01) */
    gp_real omega_min;  /* lower clamp on omega (reference: 0.1) */
} gp_ao_params_t;

typedef struct {
    gp_ao_params_t p;
    gp_real phi[GP_AO_MAX_N];
    gp_real alpha[GP_AO_MAX_N];
    gp_real alpha_0;
    gp_real omega;
} gp_ao_t;

void gp_ao_init(gp_ao_t *ao, const gp_ao_params_t *p);
void gp_ao_reset(gp_ao_t *ao);

/* One RK4 step with the input held constant over dt. Returns the reconstruction
 * v_hat evaluated at the state BEFORE the step (as the reference does). */
gp_real gp_ao_step(gp_ao_t *ao, gp_real v_input, gp_real dt);

static inline gp_real gp_ao_phase(const gp_ao_t *ao)         { return gp_mod2pi(ao->phi[0]); }
static inline gp_real gp_ao_stride_period(const gp_ao_t *ao) { return GP_TWO_PI / ao->omega; }

#ifdef __cplusplus
}
#endif
#endif /* GP_AO_H */
