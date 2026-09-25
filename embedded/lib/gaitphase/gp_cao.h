/* gp_cao.h -- coupled adaptive oscillators (C-AO): three channels, Kuramoto-type
 * frequency consensus, event correction on the primary channel.
 *
 * Port of multi_pipeline._run_coupled_ao() in the paper configuration (kappa = 0.25,
 * confidence-weighted consensus, no event voting, no cross-channel events). Each
 * channel has its own bandpass and oscillator; after every step the three stride
 * frequencies relax towards their amplitude-weighted mean. Only the primary channel
 * (index 0, M. Gastrocnemius) drives an event detector and the phase correction, and
 * its corrected phase is the output. The secondary channels' detectors and learners of
 * the reference implementation do not influence the output and are not ported.
 *
 * Channel order is [G, VM, RF]. The bandpass outputs are also exposed because the TCN
 * reads them.
 */
#ifndef GP_CAO_H
#define GP_CAO_H

#include "gp_estimator.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GP_CAO_NCH 3

typedef struct {
    gp_real dt;
    gp_real kappa;
    int confidence_weighted;
    gp_bandpass_t bp[GP_CAO_NCH];
    gp_ao_t ao[GP_CAO_NCH];
    gp_event_detector_t ev;     /* primary channel */
    gp_correction_t corr;       /* primary channel */
    uint32_t sample;
} gp_cao_t;

typedef struct {
    gp_real phase;                    /* corrected phase of the primary channel */
    gp_real phi_ao;                   /* primary oscillator phase */
    gp_real omega[GP_CAO_NCH];        /* after coupling */
    gp_real filtered[GP_CAO_NCH];     /* bandpass outputs, TCN input */
    int event;
} gp_cao_output_t;

void gp_cao_init(gp_cao_t *c, const gp_estimator_params_t *p, gp_real kappa,
                 int confidence_weighted);
void gp_cao_reset(gp_cao_t *c);
void gp_cao_step(gp_cao_t *c, const gp_real raw_v[GP_CAO_NCH], gp_cao_output_t *out);

#ifdef __cplusplus
}
#endif
#endif /* GP_CAO_H */
