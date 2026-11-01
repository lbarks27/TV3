#pragma once

#include "tvc/types.hpp"
#include "tvc/quat.hpp"

namespace tvc {

/**
 * Frame contract (Prototype firmware design):
 *   World ENU-sim: +X east, +Y north, +Z up
 *   Body  FLU-sim: +X forward (M0 azimuth), +Y left, +Z up (thrust)
 *   PX4   NED/FRD: standard PX4, after SENS_BOARD_ROT
 *
 * Cube Orange+ is top-side up (level with the pad). The Cube arrow is 90°
 * CCW from M0 looking down (+Y_FLU); the USB port faces opposite M0 (−X).
 * Verified pad bench: SENS_BOARD_ROT=2 (YAW_90) so PX4 FRD +X is M0.
 *
 * Attitude conversion changes the world and body bases in rotation-matrix form.
 */

// ENU <-> NED (involutory)
inline Vec3 enu_to_ned(const Vec3 &enu)
{
	return Vec3{enu.y, enu.x, -enu.z};
}

inline Vec3 ned_to_enu(const Vec3 &ned)
{
	return Vec3{ned.y, ned.x, -ned.z};
}

// FLU <-> FRD (involutory)
inline Vec3 flu_to_frd(const Vec3 &flu)
{
	return Vec3{flu.x, -flu.y, -flu.z};
}

inline Vec3 frd_to_flu(const Vec3 &frd)
{
	return Vec3{frd.x, -frd.y, -frd.z};
}

/** q_sim (ENU←FLU) → q_px4 (NED←FRD). */
Quat q_sim_to_px4(const Quat &q_sim);

/** q_px4 (NED←FRD) → q_sim (ENU←FLU). */
Quat q_px4_to_sim(const Quat &q_px4);

} // namespace tvc
