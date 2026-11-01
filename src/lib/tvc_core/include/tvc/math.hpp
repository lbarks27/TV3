#pragma once

// Freestanding-friendly math for NuttX (-nostdinc++) and host.

#include <math.h>
#include <string.h>
#include <stdint.h>

namespace tvc {

inline float fminf2(float a, float b) { return a < b ? a : b; }
inline float fmaxf2(float a, float b) { return a > b ? a : b; }
inline float fabsf2(float a) { return a < 0.f ? -a : a; }
inline float copysignf2(float mag, float sgn)
{
	return (sgn < 0.f) ? -fabsf2(mag) : fabsf2(mag);
}
inline bool isfinitef2(float a)
{
#if defined(__PX4_NUTTX)
	// Avoid -Wfloat-equal under -Werror; use bit patterns.
	union { float f; uint32_t u; } v;
	v.f = a;
	return ((v.u & 0x7f800000u) != 0x7f800000u);
#else
	return isfinite(a) != 0;
#endif
}

} // namespace tvc
