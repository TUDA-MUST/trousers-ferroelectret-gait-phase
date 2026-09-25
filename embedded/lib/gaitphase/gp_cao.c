#include "gp_cao.h"

void gp_cao_init(gp_cao_t *c, const gp_estimator_params_t *p, gp_real kappa,
                 int confidence_weighted)
{
    int k;
    c->dt = (gp_real)1 / p->fs;
    c->kappa = kappa;
    c->confidence_weighted = confidence_weighted;
    for (k = 0; k < GP_CAO_NCH; ++k) {
        gp_bandpass_init(&c->bp[k], p->sos_hp, p->n_hp, p->sos_lp, p->n_lp);
        gp_ao_init(&c->ao[k], &p->ao);
    }
    gp_events_init(&c->ev, &p->ev);
    gp_correction_init(&c->corr, &p->corr);
    c->sample = 0;
}

void gp_cao_reset(gp_cao_t *c)
{
    int k;
    for (k = 0; k < GP_CAO_NCH; ++k) {
        gp_bandpass_reset(&c->bp[k]);
        gp_ao_reset(&c->ao[k]);
    }
    gp_events_reset(&c->ev);
    gp_correction_reset(&c->corr);
    c->sample = 0;
}

void gp_cao_step(gp_cao_t *c, const gp_real raw_v[GP_CAO_NCH], gp_cao_output_t *out)
{
    const uint32_t i = c->sample;
    gp_real x[GP_CAO_NCH];
    int k, event;

    /* 1. each oscillator reads its own bandpass channel */
    for (k = 0; k < GP_CAO_NCH; ++k) {
        x[k] = gp_bandpass_step(&c->bp[k], raw_v[k]);
        gp_ao_step(&c->ao[k], x[k], c->dt);
    }

    /* 2. Kuramoto-type frequency consensus, weighted by oscillator amplitude */
    if (c->kappa > (gp_real)0) {
        gp_real omega_mean;
        if (c->confidence_weighted) {
            gp_real q[GP_CAO_NCH], q_total = 0;
            for (k = 0; k < GP_CAO_NCH; ++k) {
                int n;
                q[k] = 0;
                for (n = 0; n < c->ao[k].p.N; ++n) q[k] += GP_FABS(c->ao[k].alpha[n]);
                q_total += q[k];
            }
            if (q_total > (gp_real)1e-10) {
                gp_real s = 0;
                for (k = 0; k < GP_CAO_NCH; ++k) s += q[k] * c->ao[k].omega;
                omega_mean = s / q_total;
            } else {
                gp_real s = 0;
                for (k = 0; k < GP_CAO_NCH; ++k) s += c->ao[k].omega;
                omega_mean = s / (gp_real)GP_CAO_NCH;
            }
        } else {
            gp_real s = 0;
            for (k = 0; k < GP_CAO_NCH; ++k) s += c->ao[k].omega;
            omega_mean = s / (gp_real)GP_CAO_NCH;
        }
        for (k = 0; k < GP_CAO_NCH; ++k) {
            gp_real om = c->ao[k].omega + c->kappa * (omega_mean - c->ao[k].omega) * c->dt;
            c->ao[k].omega = om < c->ao[k].p.omega_min ? c->ao[k].p.omega_min : om;
        }
    }

    /* 3./4. primary channel: event detection and phase correction */
    {
        gp_real phi_n = gp_ao_phase(&c->ao[0]);
        gp_real omega = c->ao[0].omega;
        event = gp_events_step(&c->ev, x[0], i, c->dt, gp_ao_stride_period(&c->ao[0]));
        if (event) gp_correction_on_event(&c->corr, phi_n, i);
        gp_correction_step(&c->corr, omega, i, c->dt);
        out->phase = gp_correction_output(&c->corr, phi_n);
        out->phi_ao = phi_n;
    }
    for (k = 0; k < GP_CAO_NCH; ++k) {
        out->omega[k] = c->ao[k].omega;
        out->filtered[k] = x[k];
    }
    out->event = event;
    c->sample = i + 1;
}
