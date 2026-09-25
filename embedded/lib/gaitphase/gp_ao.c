#include "gp_ao.h"

void gp_ao_init(gp_ao_t *ao, const gp_ao_params_t *p)
{
    ao->p = *p;
    if (ao->p.N > GP_AO_MAX_N) ao->p.N = GP_AO_MAX_N;
    gp_ao_reset(ao);
}

void gp_ao_reset(gp_ao_t *ao)
{
    int n;
    for (n = 0; n < GP_AO_MAX_N; ++n) {
        ao->phi[n] = 0;
        ao->alpha[n] = ao->p.alpha_init;
    }
    ao->alpha_0 = 0;
    ao->omega = ao->p.omega_init;
}

/* Mirrors AdaptiveOscillator._derivatives(). */
static gp_real derivs(const gp_ao_t *ao,
                      const gp_real *phi, gp_real omega, const gp_real *alpha, gp_real alpha_0,
                      gp_real v_input,
                      gp_real *dphi, gp_real *domega, gp_real *dalpha, gp_real *dalpha_0)
{
    const int N = ao->p.N;
    gp_real s[GP_AO_MAX_N], c[GP_AO_MAX_N];
    gp_real v_hat = alpha_0;
    gp_real F, alpha_sum = 0, g;
    int n;

    for (n = 0; n < N; ++n) {
        s[n] = GP_SIN(phi[n]);
        c[n] = GP_COS(phi[n]);
        v_hat += alpha[n] * s[n];
    }
    F = v_input - v_hat;
    for (n = 0; n < N; ++n) alpha_sum += GP_FABS(alpha[n]);
    if (alpha_sum < (gp_real)1e-10) alpha_sum = (gp_real)1e-10;
    g = F / alpha_sum;

    for (n = 0; n < N; ++n) {
        dphi[n]   = omega * (gp_real)(n + 1) + ao->p.nu_phi * g * c[n];
        dalpha[n] = ao->p.eta * F * s[n];
    }
    *domega   = ao->p.nu_omega * g * c[0];
    *dalpha_0 = ao->p.eta * F;
    return v_hat;
}

static inline gp_real clamp_omega(gp_real om, gp_real om_min)
{
    return om < om_min ? om_min : om;
}

gp_real gp_ao_step(gp_ao_t *ao, gp_real v, gp_real dt)
{
    const int N = ao->p.N;
    const gp_real om_min = ao->p.omega_min;
    gp_real k1p[GP_AO_MAX_N], k2p[GP_AO_MAX_N], k3p[GP_AO_MAX_N], k4p[GP_AO_MAX_N];
    gp_real k1a[GP_AO_MAX_N], k2a[GP_AO_MAX_N], k3a[GP_AO_MAX_N], k4a[GP_AO_MAX_N];
    gp_real k1o, k2o, k3o, k4o, k1a0, k2a0, k3a0, k4a0;
    gp_real phi_t[GP_AO_MAX_N], al_t[GP_AO_MAX_N], om_t, al0_t;
    gp_real v_hat;
    const gp_real hdt = (gp_real)0.5 * dt;
    int n;

    v_hat = derivs(ao, ao->phi, ao->omega, ao->alpha, ao->alpha_0, v, k1p, &k1o, k1a, &k1a0);

    for (n = 0; n < N; ++n) { phi_t[n] = ao->phi[n] + hdt * k1p[n]; al_t[n] = ao->alpha[n] + hdt * k1a[n]; }
    om_t  = clamp_omega(ao->omega + hdt * k1o, om_min);
    al0_t = ao->alpha_0 + hdt * k1a0;
    derivs(ao, phi_t, om_t, al_t, al0_t, v, k2p, &k2o, k2a, &k2a0);

    for (n = 0; n < N; ++n) { phi_t[n] = ao->phi[n] + hdt * k2p[n]; al_t[n] = ao->alpha[n] + hdt * k2a[n]; }
    om_t  = clamp_omega(ao->omega + hdt * k2o, om_min);
    al0_t = ao->alpha_0 + hdt * k2a0;
    derivs(ao, phi_t, om_t, al_t, al0_t, v, k3p, &k3o, k3a, &k3a0);

    for (n = 0; n < N; ++n) { phi_t[n] = ao->phi[n] + dt * k3p[n]; al_t[n] = ao->alpha[n] + dt * k3a[n]; }
    om_t  = clamp_omega(ao->omega + dt * k3o, om_min);
    al0_t = ao->alpha_0 + dt * k3a0;
    derivs(ao, phi_t, om_t, al_t, al0_t, v, k4p, &k4o, k4a, &k4a0);

    {
        const gp_real w = dt / (gp_real)6;
        for (n = 0; n < N; ++n) {
            ao->phi[n]   = gp_mod2pi(ao->phi[n] + w * (k1p[n] + 2 * k2p[n] + 2 * k3p[n] + k4p[n]));
            ao->alpha[n] = ao->alpha[n] + w * (k1a[n] + 2 * k2a[n] + 2 * k3a[n] + k4a[n]);
        }
        ao->omega   = clamp_omega(ao->omega + w * (k1o + 2 * k2o + 2 * k3o + k4o), om_min);
        ao->alpha_0 = ao->alpha_0 + w * (k1a0 + 2 * k2a0 + 2 * k3a0 + k4a0);
    }
    return v_hat;
}
