/* gp_types.h -- scalar type and math shims for the portable gait phase estimator.
 *
 * The estimator is written in plain C99 against a single scalar type `gp_real`.
 * Build with -DGP_USE_DOUBLE to get a double-precision variant that reproduces the
 * Python float64 reference to rounding; the default single-precision variant is the
 * deployment configuration for MCUs with a single-precision FPU (ESP32, Cortex-M4F).
 */
#ifndef GP_TYPES_H
#define GP_TYPES_H

#include <math.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GP_USE_DOUBLE
typedef double gp_real;
#define GP_SIN(x)   sin(x)
#define GP_COS(x)   cos(x)
#define GP_EXP(x)   exp(x)
#define GP_SQRT(x)  sqrt(x)
#define GP_FABS(x)  fabs(x)
#define GP_FMOD(x, y) fmod((x), (y))
#define GP_ATAN2(y, x) atan2((y), (x))
#define GP_REAL_NAME "double"
#else
typedef float gp_real;
#define GP_SIN(x)   sinf(x)
#define GP_COS(x)   cosf(x)
#define GP_EXP(x)   expf(x)
#define GP_SQRT(x)  sqrtf(x)
#define GP_FABS(x)  fabsf(x)
#define GP_FMOD(x, y) fmodf((x), (y))
#define GP_ATAN2(y, x) atan2f((y), (x))
#define GP_REAL_NAME "float"
#endif

#define GP_PI     ((gp_real)3.14159265358979323846)
#define GP_TWO_PI ((gp_real)6.28318530717958647692)

/* Python-style modulo: result in [0, 2*pi) for any finite x. */
static inline gp_real gp_mod2pi(gp_real x)
{
    x = GP_FMOD(x, GP_TWO_PI);
    if (x < (gp_real)0) x += GP_TWO_PI;
    return x;
}

/* Wrap to (-pi, pi]: (x + pi) mod 2pi - pi, Python semantics. */
static inline gp_real gp_wrap_pi(gp_real x)
{
    return gp_mod2pi(x + GP_PI) - GP_PI;
}

#ifdef __cplusplus
}
#endif
#endif /* GP_TYPES_H */
