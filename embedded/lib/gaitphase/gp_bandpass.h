/* gp_bandpass.h -- causal IIR filter in second-order sections (scipy.signal.sosfilt form).
 *
 * The estimator's preprocessing is a first-order Butterworth highpass (0.5 Hz) followed
 * by a fourth-order Butterworth lowpass (15 Hz), each run as its own sosfilt with the
 * state initialised to the steady-state response for the first input sample
 * (scipy's sosfilt_zi(sos) * x[0]). This file reproduces exactly that: transposed
 * direct form II per section, coefficients and zi vectors exported from scipy.
 */
#ifndef GP_BANDPASS_H
#define GP_BANDPASS_H

#include "gp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GP_SOS_MAX_SECTIONS 4

typedef struct {
    gp_real b0, b1, b2, a1, a2;   /* a0 normalised to 1 */
    gp_real zi0, zi1;             /* steady-state state for a unit step (sosfilt_zi) */
} gp_sos_section_t;

typedef struct {
    int n_sections;
    gp_sos_section_t s[GP_SOS_MAX_SECTIONS];
    gp_real z0[GP_SOS_MAX_SECTIONS];
    gp_real z1[GP_SOS_MAX_SECTIONS];
    int primed;                   /* 0 until the first sample has initialised the state */
} gp_sos_t;

/* Copy coefficients; state is initialised on the first call to gp_sos_step(). */
void gp_sos_init(gp_sos_t *f, const gp_sos_section_t *sections, int n_sections);
void gp_sos_reset(gp_sos_t *f);

/* One sample in, one sample out. On the first call the state is set to zi * x. */
gp_real gp_sos_step(gp_sos_t *f, gp_real x);

/* The estimator's two-stage bandpass. */
typedef struct {
    gp_sos_t hp;
    gp_sos_t lp;
} gp_bandpass_t;

void gp_bandpass_init(gp_bandpass_t *bp,
                      const gp_sos_section_t *hp, int n_hp,
                      const gp_sos_section_t *lp, int n_lp);
void gp_bandpass_reset(gp_bandpass_t *bp);
gp_real gp_bandpass_step(gp_bandpass_t *bp, gp_real x);

#ifdef __cplusplus
}
#endif
#endif /* GP_BANDPASS_H */
