#pragma once

#include "tvc/math.hpp"

namespace tvc {

static constexpr int kMaxMounts = 3;
static constexpr int kAnglesPerMount = 2;
static constexpr int kNumAngles = kMaxMounts * kAnglesPerMount; // 6

struct Vec3 {
	float x{0.f};
	float y{0.f};
	float z{0.f};

	float &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
	float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

struct Quat {
	float w{1.f};
	float x{0.f};
	float y{0.f};
	float z{0.f};
};

struct Mount {
	Vec3 position{};
	Vec3 base_dir{};
	Vec3 axis1{};
	Vec3 axis2{};
	float thrust{0.f};           // nominal / schedule fallback [N]
	float min_angles[2]{};       // [a2, a1] rad
	float max_angles[2]{};
	float rest_angles[2]{};      // [a2, a1] rad
};

struct SolverConfig {
	float damping{0.15f};          // ridge = damping^2
	float finite_diff_eps{1e-4f};
	float tol{1e-5f};
	int force_search_iters{16};
	int a2_polish_iters{2};
	// Scale τx/τz in a2 LS (<1 deprioritizes that axis vs pitch).
	float roll_torque_weight{1.f};
	float yaw_torque_weight{1.f};
	// Cube: flip M1/M2 Jx. Default off so SIH stays on JSON geometry.
	bool invert_side_roll{false};
	// Cube: flip Jz on every mount. TvcControl also remaps M0 yaw a2.
	bool invert_yaw{false};
};

struct AllocResult {
	float angles[kNumAngles]{};    // M0a2,M0a1,... rad
	float torque[3]{};
	float force[3]{};
	bool saturated{false};
	float residual{0.f};
	uint32_t solve_us{0};
	bool timed_out{false};
};

inline float clampf(float v, float lo, float hi)
{
	return v < lo ? lo : (v > hi ? hi : v);
}

inline float deg2rad(float d) { return d * 0.017453292519943295f; }
inline float rad2deg(float r) { return r * 57.29577951308232f; }

inline float vnorm(const Vec3 &v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 vunit(const Vec3 &v)
{
	const float n = vnorm(v);
	if (n < 1e-12f) {
		return Vec3{0.f, 0.f, 0.f};
	}
	return Vec3{v.x / n, v.y / n, v.z / n};
}

inline float vdot(const Vec3 &a, const Vec3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 vcross(const Vec3 &a, const Vec3 &b)
{
	return Vec3{
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

inline Vec3 vadd(const Vec3 &a, const Vec3 &b)
{
	return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 vsub(const Vec3 &a, const Vec3 &b)
{
	return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 vscale(const Vec3 &a, float s)
{
	return Vec3{a.x * s, a.y * s, a.z * s};
}

/** Default ring-3 geometry from Prototype config/vehicles/tvc_ring3.json (FLU-sim). */
inline void default_ring3_mounts(Mount out[kMaxMounts])
{
	// M0
	out[0].position = {0.098f, 0.f, 0.f};
	out[0].base_dir = {-1.f, 0.f, 0.f};
	out[0].axis1 = {0.f, 1.f, 0.f};
	out[0].axis2 = {-1.f, 0.f, 0.f};
	out[0].thrust = 12.f;
	out[0].min_angles[0] = deg2rad(-45.f);
	out[0].max_angles[0] = deg2rad(45.f);
	out[0].min_angles[1] = deg2rad(0.f);
	out[0].max_angles[1] = deg2rad(90.f);
	out[0].rest_angles[0] = deg2rad(0.f);
	out[0].rest_angles[1] = deg2rad(0.f);

	// M1
	out[1].position = {-0.049f, 0.08487049f, 0.f};
	out[1].base_dir = {0.5f, -0.8660254f, 0.f};
	out[1].axis1 = {-0.8660254f, -0.5f, 0.f};
	out[1].axis2 = {0.5f, -0.8660254f, 0.f};
	out[1].thrust = 12.f;
	out[1].min_angles[0] = deg2rad(-45.f);
	out[1].max_angles[0] = deg2rad(45.f);
	out[1].min_angles[1] = deg2rad(0.f);
	out[1].max_angles[1] = deg2rad(90.f);
	out[1].rest_angles[0] = deg2rad(0.f);
	out[1].rest_angles[1] = deg2rad(0.f);

	// M2
	out[2].position = {-0.049f, -0.08487049f, 0.f};
	out[2].base_dir = {0.5f, 0.8660254f, 0.f};
	out[2].axis1 = {0.8660254f, -0.5f, 0.f};
	out[2].axis2 = {0.5f, 0.8660254f, 0.f};
	out[2].thrust = 12.f;
	out[2].min_angles[0] = deg2rad(-45.f);
	out[2].max_angles[0] = deg2rad(45.f);
	out[2].min_angles[1] = deg2rad(0.f);
	out[2].max_angles[1] = deg2rad(90.f);
	out[2].rest_angles[0] = deg2rad(0.f);
	out[2].rest_angles[1] = deg2rad(0.f);
}

inline Vec3 default_com()
{
	// Unloaded CoM in tvc_ring3_g12.json (ignored when SD JSON loads).
	return Vec3{0.f, 0.f, 0.4699f};
}

inline Vec3 default_force_axis()
{
	return Vec3{0.f, 0.f, 1.f};
}

} // namespace tvc
