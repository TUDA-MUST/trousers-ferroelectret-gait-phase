#include "gp_correction.h"

void gp_correction_init(gp_correction_t *c, const gp_correction_params_t *p)
{
    c->p = *p;
    gp_correction_reset(c);
}

void gp_correction_reset(gp_correction_t *c)
{
    c->phi_e = 0;
    c->epsilon_phi = 0;
    c->P_tilde_prev = 0;
    c->last_event_sample = -1;
    c->n_events = 0;
}

void gp_correction_on_event(gp_correction_t *c, gp_real phi_n, uint32_t sample)
{
    /* Eq. 8: phase error at the event, wrapped to (-pi, pi] */
    gp_real P_e = gp_wrap_pi(c->p.event_ref_phase - phi_n);
    /* Eq. 9: unwrap against the previous event's value */
    gp_real P_tilde = P_e;
    const gp_real half_pi = GP_PI / (gp_real)2;
    if (P_e > half_pi && c->P_tilde_prev < -half_pi)       P_tilde = P_e - GP_TWO_PI;
    else if (P_e < -half_pi && c->P_tilde_prev > half_pi)  P_tilde = P_e + GP_TWO_PI;
    c->P_tilde_prev = P_tilde;
    /* Eq. 10 gain */
    c->epsilon_phi = c->p.kp * (P_tilde - c->phi_e);
    c->last_event_sample = (int32_t)sample;
    c->n_events++;
}

void gp_correction_step(gp_correction_t *c, gp_real omega, uint32_t sample, gp_real dt)
{
    gp_real dt_since, dphi_e;
    if (c->last_event_sample < 0) return;
    dt_since = (gp_real)(sample - (uint32_t)c->last_event_sample) * dt;
    dphi_e = c->epsilon_phi * omega * GP_EXP(-omega * dt_since);
    c->phi_e += dphi_e * dt;
    if (c->phi_e >  GP_TWO_PI) c->phi_e =  GP_TWO_PI;
    if (c->phi_e < -GP_TWO_PI) c->phi_e = -GP_TWO_PI;
}

gp_real gp_correction_output(const gp_correction_t *c, gp_real phi_n)
{
    return gp_mod2pi(phi_n + c->phi_e - c->p.event_ref_phase + c->p.phase_offset);
}
