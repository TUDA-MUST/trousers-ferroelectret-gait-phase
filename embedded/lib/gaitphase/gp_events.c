#include "gp_events.h"

void gp_events_init(gp_event_detector_t *d, const gp_event_params_t *p)
{
    d->p = *p;
    gp_events_reset(d);
}

void gp_events_reset(gp_event_detector_t *d)
{
    d->envelope = 0;
    d->bp_mean = 0;
    d->bp_var = 0;
    d->threshold = 0;
    d->primed = 0;
    d->prev_crossed = 0;
    d->last_event_sample = -1;
    d->n_events = 0;
}

int gp_events_step(gp_event_detector_t *d, gp_real x, uint32_t sample, gp_real dt,
                   gp_real stride_period)
{
    const gp_real a = d->p.env_alpha;
    const gp_real one_minus_a = (gp_real)1 - a;
    gp_real abs_x = GP_FABS(x), diff, std, theta;
    int crossed, fired = 0;

    if (!d->primed) {
        d->envelope = abs_x;
        d->bp_mean = x;
        d->primed = 1;
    } else {
        d->envelope = one_minus_a * d->envelope + a * abs_x;
    }
    /* reference updates mean/var on the first sample too */
    d->bp_mean = one_minus_a * d->bp_mean + a * x;
    diff = x - d->bp_mean;
    d->bp_var = one_minus_a * d->bp_var + a * (diff * diff);
    std = GP_SQRT(d->bp_var);

    theta = -d->p.k_env * d->envelope - d->p.k_std_bp * std;
    d->threshold = theta;
    crossed = (x <= theta);

    if (crossed && !d->prev_crossed) {
        int ok;
        if (d->last_event_sample < 0) {
            ok = 1;                                   /* time - (-inf) > anything */
        } else {
            gp_real elapsed = (gp_real)(sample - (uint32_t)d->last_event_sample) * dt;
            ok = elapsed > d->p.min_stride_fraction * stride_period;
        }
        if (ok) {
            fired = 1;
            d->last_event_sample = (int32_t)sample;
            d->n_events++;
        }
    }
    d->prev_crossed = crossed;
    return fired;
}
