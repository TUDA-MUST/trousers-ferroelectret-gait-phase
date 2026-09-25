/* gp_estimator.h -- single-sensor gait phase estimator: bandpass -> AO -> events ->
 * correction, one call per ADC sample. Mirrors the per-sample loop of run_pipeline().
 */
#ifndef GP_ESTIMATOR_H
#define GP_ESTIMATOR_H

#include "gp_types.h"
#include "gp_bandpass.h"
#include "gp_ao.h"
#include "gp_events.h"
#include "gp_correction.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    gp_real fs;                       /* sample rate, Hz */
    const gp_sos_section_t *sos_hp; int n_hp;
    const gp_sos_section_t *sos_lp; int n_lp;
    gp_ao_params_t ao;
    gp_event_params_t ev;
    gp_correction_params_t corr;
} gp_estimator_params_t;

typedef struct {
    gp_real dt;
    gp_bandpass_t bp;
    gp_ao_t ao;
    gp_event_detector_t ev;
    gp_correction_t corr;
    uint32_t sample;                  /* index of the next sample */
} gp_estimator_t;

typedef struct {
    gp_real phase;      /* corrected gait phase, [0, 2pi) */
    gp_real phi_ao;     /* raw oscillator phase, [0, 2pi) */
    gp_real omega;      /* rad/s */
    gp_real filtered;   /* bandpass output */
    gp_real phi_e;      /* correction state */
    int     event;      /* 1 on the sample an event fired */
} gp_output_t;

void gp_estimator_init(gp_estimator_t *e, const gp_estimator_params_t *p);
void gp_estimator_reset(gp_estimator_t *e);

/* Feed one raw sensor sample (volts). */
void gp_estimator_step(gp_estimator_t *e, gp_real raw_v, gp_output_t *out);

#ifdef __cplusplus
}
#endif
#endif /* GP_ESTIMATOR_H */
