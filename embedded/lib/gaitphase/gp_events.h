/* gp_events.h -- adaptive envelope-threshold gait event detector on the bandpass signal.
 *
 * Port of GaitEventDetector mode 'bp_threshold' (direction 'neg'):
 *   E[n]   = (1-a) E[n-1] + a |x[n]|          running envelope
 *   mu[n]  = (1-a) mu[n-1] + a x[n]            running mean
 *   var[n] = (1-a) var[n-1] + a (x[n]-mu[n])^2
 *   theta  = -k_env E - k_std sqrt(var)
 * An event fires on the first sample with x <= theta after a sample above it, provided
 * at least min_stride_fraction * T(t) has elapsed since the previous event, where T(t)
 * is the oscillator's current stride period. Time is kept as a sample index.
 */
#ifndef GP_EVENTS_H
#define GP_EVENTS_H

#include "gp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    gp_real env_alpha;
    gp_real k_env;
    gp_real k_std_bp;
    gp_real min_stride_fraction;
} gp_event_params_t;

typedef struct {
    gp_event_params_t p;
    gp_real envelope;
    gp_real bp_mean;
    gp_real bp_var;
    gp_real threshold;         /* last computed threshold (for logging) */
    int     primed;
    int     prev_crossed;
    int32_t last_event_sample; /* -1 = none yet (reference: -inf) */
    uint32_t n_events;
} gp_event_detector_t;

void gp_events_init(gp_event_detector_t *d, const gp_event_params_t *p);
void gp_events_reset(gp_event_detector_t *d);

/* Returns 1 if a gait event fires on this sample. `stride_period` in seconds, `dt` the
 * sample period, `sample` the running sample index (time = sample * dt). */
int gp_events_step(gp_event_detector_t *d, gp_real x, uint32_t sample, gp_real dt,
                   gp_real stride_period);

#ifdef __cplusplus
}
#endif
#endif /* GP_EVENTS_H */
