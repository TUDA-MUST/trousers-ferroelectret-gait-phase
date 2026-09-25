#include "gp_bandpass.h"

void gp_sos_init(gp_sos_t *f, const gp_sos_section_t *sections, int n_sections)
{
    int i;
    if (n_sections > GP_SOS_MAX_SECTIONS) n_sections = GP_SOS_MAX_SECTIONS;
    f->n_sections = n_sections;
    for (i = 0; i < n_sections; ++i) f->s[i] = sections[i];
    gp_sos_reset(f);
}

void gp_sos_reset(gp_sos_t *f)
{
    int i;
    for (i = 0; i < GP_SOS_MAX_SECTIONS; ++i) { f->z0[i] = 0; f->z1[i] = 0; }
    f->primed = 0;
}

gp_real gp_sos_step(gp_sos_t *f, gp_real x)
{
    int i;
    if (!f->primed) {
        /* scipy: zi = sosfilt_zi(sos) * x[0] */
        for (i = 0; i < f->n_sections; ++i) {
            f->z0[i] = f->s[i].zi0 * x;
            f->z1[i] = f->s[i].zi1 * x;
        }
        f->primed = 1;
    }
    /* scipy _sosfilt: transposed direct form II, sections in cascade */
    for (i = 0; i < f->n_sections; ++i) {
        const gp_sos_section_t *s = &f->s[i];
        gp_real xc = x;
        x = s->b0 * xc + f->z0[i];
        f->z0[i] = s->b1 * xc - s->a1 * x + f->z1[i];
        f->z1[i] = s->b2 * xc - s->a2 * x;
    }
    return x;
}

void gp_bandpass_init(gp_bandpass_t *bp,
                      const gp_sos_section_t *hp, int n_hp,
                      const gp_sos_section_t *lp, int n_lp)
{
    gp_sos_init(&bp->hp, hp, n_hp);
    gp_sos_init(&bp->lp, lp, n_lp);
}

void gp_bandpass_reset(gp_bandpass_t *bp)
{
    gp_sos_reset(&bp->hp);
    gp_sos_reset(&bp->lp);
}

gp_real gp_bandpass_step(gp_bandpass_t *bp, gp_real x)
{
    /* Two separate sosfilt calls in the reference: the lowpass state is primed with
     * the highpass OUTPUT of the first sample, which gp_sos_step does implicitly. */
    gp_real y_hp = gp_sos_step(&bp->hp, x);
    return gp_sos_step(&bp->lp, y_hp);
}
