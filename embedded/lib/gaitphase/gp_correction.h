/* gp_correction.h -- event-anchored phase error correction (Yan et al. 2017, Eqs. 8-11).
 *
 * Port of PhaseErrorLearner with learn_ref_phase=False, adaptive_kp=False,
 * event_ref_phase=0 (the configuration used throughout the paper).
 *   on event k at AO phase phi_n:  P_e = wrap(0 - phi_n), unwrapped against the
 *                                  previous value; eps = kp (P~ - phi_e)
 *   every sample:                   phi_e += eps * w * exp(-w (t - t_k)) * dt, |phi_e| <= 2pi
 *   output:                         phi = (phi_n + phi_e + phase_offset) mod 2pi
 */
#ifndef GP_CORRECTION_H
#define GP_CORRECTION_H

#include "gp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    gp_real kp;
    gp_real phase_offset;
    gp_real event_ref_phase;   /* 0 in the paper configuration */
} gp_correction_params_t;

typedef struct {
    gp_correction_params_t p;
    gp_real phi_e;
    gp_real epsilon_phi;
    gp_real P_tilde_prev;
    int32_t last_event_sample;  /* -1 = no event yet */
    uint32_t n_events;
} gp_correction_t;

void gp_correction_init(gp_correction_t *c, const gp_correction_params_t *p);
void gp_correction_reset(gp_correction_t *c);

void    gp_correction_on_event(gp_correction_t *c, gp_real phi_n, uint32_t sample);
void    gp_correction_step(gp_correction_t *c, gp_real omega, uint32_t sample, gp_real dt);
gp_real gp_correction_output(const gp_correction_t *c, gp_real phi_n);

#ifdef __cplusplus
}
#endif
#endif /* GP_CORRECTION_H */
