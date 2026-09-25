#include "gp_estimator.h"

void gp_estimator_init(gp_estimator_t *e, const gp_estimator_params_t *p)
{
    e->dt = (gp_real)1 / p->fs;
    gp_bandpass_init(&e->bp, p->sos_hp, p->n_hp, p->sos_lp, p->n_lp);
    gp_ao_init(&e->ao, &p->ao);
    gp_events_init(&e->ev, &p->ev);
    gp_correction_init(&e->corr, &p->corr);
    e->sample = 0;
}

void gp_estimator_reset(gp_estimator_t *e)
{
    gp_bandpass_reset(&e->bp);
    gp_ao_reset(&e->ao);
    gp_events_reset(&e->ev);
    gp_correction_reset(&e->corr);
    e->sample = 0;
}

void gp_estimator_step(gp_estimator_t *e, gp_real raw_v, gp_output_t *out)
{
    const uint32_t i = e->sample;
    gp_real x, phi_n, omega, T;
    int event;

    /* 0. causal bandpass (the AO and the detector both read the filtered signal) */
    x = gp_bandpass_step(&e->bp, raw_v);

    /* 1. oscillator */
    gp_ao_step(&e->ao, x, e->dt);
    phi_n = gp_ao_phase(&e->ao);
    omega = e->ao.omega;
    T = gp_ao_stride_period(&e->ao);

    /* 2. event detector */
    event = gp_events_step(&e->ev, x, i, e->dt, T);

    /* 3./4. phase error at the event, then the every-sample relaxation */
    if (event) gp_correction_on_event(&e->corr, phi_n, i);
    gp_correction_step(&e->corr, omega, i, e->dt);

    /* 5. corrected output */
    out->phase    = gp_correction_output(&e->corr, phi_n);
    out->phi_ao   = phi_n;
    out->omega    = omega;
    out->filtered = x;
    out->phi_e    = e->corr.phi_e;
    out->event    = event;

    e->sample = i + 1;
}
