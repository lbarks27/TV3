#include "tvc/vehicle_config.hpp"

#include "tvc/math.hpp"

#include <stdio.h>
#include <stdlib.h>

namespace tvc {

namespace {

VehicleConfig g_loaded{};

bool is_space(char c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

const char *skip_ws(const char *p, const char *e)
{
	while (p < e && is_space(*p)) {
		p++;
	}
	return p;
}

const char *skip_string(const char *p, const char *e)
{
	if (p >= e || *p != '"') {
		return nullptr;
	}
	p++;
	while (p < e) {
		if (*p == '\\') {
			p += 2;
			continue;
		}
		if (*p == '"') {
			return p + 1;
		}
		p++;
	}
	return nullptr;
}

const char *skip_value(const char *p, const char *e);

const char *skip_object_or_array(const char *p, const char *e, char open_c, char close_c)
{
	if (p >= e || *p != open_c) {
		return nullptr;
	}
	int depth = 1;
	p++;
	while (p < e && depth > 0) {
		if (*p == '"') {
			p = skip_string(p, e);
			if (!p) {
				return nullptr;
			}
			continue;
		}
		if (*p == open_c) {
			depth++;
		} else if (*p == close_c) {
			depth--;
		}
		p++;
	}
	return depth == 0 ? p : nullptr;
}

const char *skip_value(const char *p, const char *e)
{
	p = skip_ws(p, e);
	if (p >= e) {
		return nullptr;
	}
	if (*p == '"') {
		return skip_string(p, e);
	}
	if (*p == '{') {
		return skip_object_or_array(p, e, '{', '}');
	}
	if (*p == '[') {
		return skip_object_or_array(p, e, '[', ']');
	}
	if (*p == 't' || *p == 'f' || *p == 'n' || *p == '-' || (*p >= '0' && *p <= '9')) {
		while (p < e && !is_space(*p) && *p != ',' && *p != '}' && *p != ']') {
			p++;
		}
		return p;
	}
	return nullptr;
}

bool key_equals(const char *key_start, const char *key_end, const char *want)
{
	const unsigned n = (unsigned)(key_end - key_start);
	unsigned i = 0;
	while (want[i] != '\0') {
		if (i >= n || key_start[i] != want[i]) {
			return false;
		}
		i++;
	}
	return i == n;
}

/** Find ``"key":`` in an object; returns pointer to the value (after colon). */
const char *find_key(const char *obj, const char *e, const char *key, const char **obj_end)
{
	obj = skip_ws(obj, e);
	if (obj >= e || *obj != '{') {
		return nullptr;
	}
	const char *end = skip_object_or_array(obj, e, '{', '}');
	if (!end) {
		return nullptr;
	}
	if (obj_end) {
		*obj_end = end;
	}
	const char *p = obj + 1;
	while (p < end - 1) {
		p = skip_ws(p, end);
		if (p >= end - 1 || *p == '}') {
			break;
		}
		if (*p != '"') {
			return nullptr;
		}
		const char *ks = p + 1;
		const char *after = skip_string(p, end);
		if (!after) {
			return nullptr;
		}
		const char *ke = after - 1;
		p = skip_ws(after, end);
		if (p >= end || *p != ':') {
			return nullptr;
		}
		p = skip_ws(p + 1, end);
		if (key_equals(ks, ke, key)) {
			return p;
		}
		p = skip_value(p, end);
		if (!p) {
			return nullptr;
		}
		p = skip_ws(p, end);
		if (p < end && *p == ',') {
			p++;
		}
	}
	return nullptr;
}

bool parse_number(const char *p, const char *e, float &out, const char **end_out)
{
	p = skip_ws(p, e);
	if (p >= e) {
		return false;
	}
	char *endp = nullptr;
	// strtof needs a C string; copy a short token.
	char buf[64];
	unsigned n = 0;
	const char *q = p;
	if (*q == '-') {
		buf[n++] = *q++;
	}
	while (q < e && n + 1 < sizeof(buf) &&
	       ((*q >= '0' && *q <= '9') || *q == '.' || *q == 'e' || *q == 'E' || *q == '+' || *q == '-')) {
		if ((*q == '+' || *q == '-') && n > 0 && buf[n - 1] != 'e' && buf[n - 1] != 'E') {
			break;
		}
		buf[n++] = *q++;
	}
	buf[n] = '\0';
	if (n == 0) {
		return false;
	}
	out = strtof(buf, &endp);
	if (endp == buf) {
		return false;
	}
	if (end_out) {
		*end_out = p + (unsigned)(endp - buf);
	}
	return isfinitef2(out);
}

bool parse_vec3(const char *p, const char *e, Vec3 &out)
{
	if (!p) {
		return false;
	}
	p = skip_ws(p, e);
	if (p >= e || *p != '[') {
		return false;
	}
	p++;
	const char *end = nullptr;
	if (!parse_number(p, e, out.x, &end)) {
		return false;
	}
	p = skip_ws(end, e);
	if (p >= e || *p != ',') {
		return false;
	}
	if (!parse_number(p + 1, e, out.y, &end)) {
		return false;
	}
	p = skip_ws(end, e);
	if (p >= e || *p != ',') {
		return false;
	}
	if (!parse_number(p + 1, e, out.z, &end)) {
		return false;
	}
	p = skip_ws(end, e);
	return p < e && *p == ']';
}

bool parse_pair_deg(const char *obj, const char *e, const char *key, float &lo, float &hi)
{
	const char *val = find_key(obj, e, key, nullptr);
	if (!val) {
		return false;
	}
	val = skip_ws(val, e);
	if (val >= e || *val != '[') {
		return false;
	}
	const char *end = nullptr;
	if (!parse_number(val + 1, e, lo, &end)) {
		return false;
	}
	end = skip_ws(end, e);
	if (end >= e || *end != ',') {
		return false;
	}
	if (!parse_number(end + 1, e, hi, &end)) {
		return false;
	}
	return lo <= hi && isfinitef2(lo) && isfinitef2(hi);
}

bool parse_angle_limits(const char *obj, const char *e, float &a2_min, float &a2_max,
			float &a1_min, float &a1_max)
{
	const char *lim = find_key(obj, e, "angle_limits_deg", nullptr);
	if (!lim) {
		return false;
	}
	const char *lim_end = skip_object_or_array(skip_ws(lim, e), e, '{', '}');
	if (!lim_end) {
		return false;
	}
	return parse_pair_deg(lim, e, "a2", a2_min, a2_max) &&
	       parse_pair_deg(lim, e, "a1", a1_min, a1_max);
}

bool parse_rest_angles(const char *obj, const char *e, float &a2_rest, float &a1_rest)
{
	const char *rest = find_key(obj, e, "rest_angles_deg", nullptr);
	if (!rest) {
		return false;
	}
	const char *val = nullptr;
	val = find_key(rest, e, "a2", nullptr);
	if (!val || !parse_number(val, e, a2_rest, nullptr)) {
		return false;
	}
	val = find_key(rest, e, "a1", nullptr);
	if (!val || !parse_number(val, e, a1_rest, nullptr)) {
		return false;
	}
	return isfinitef2(a2_rest) && isfinitef2(a1_rest);
}

bool parse_float_key(const char *obj, const char *e, const char *key, float &out)
{
	const char *val = find_key(obj, e, key, nullptr);
	if (!val) {
		return false;
	}
	return parse_number(val, e, out, nullptr);
}

void apply_angles(Mount &m, float a2_min, float a2_max, float a1_min, float a1_max,
		  float a2_rest, float a1_rest)
{
	m.min_angles[0] = deg2rad(a2_min);
	m.max_angles[0] = deg2rad(a2_max);
	m.min_angles[1] = deg2rad(a1_min);
	m.max_angles[1] = deg2rad(a1_max);
	m.rest_angles[0] = deg2rad(a2_rest);
	m.rest_angles[1] = deg2rad(a1_rest);
}

bool parse_mount_object(const char *obj, const char *e, Mount &m, float &mass_kg,
			float shared_a2_min, float shared_a2_max, float shared_a1_min,
			float shared_a1_max, float shared_a2_rest, float shared_a1_rest,
			bool have_shared_limits, bool have_shared_rest)
{
	if (!parse_vec3(find_key(obj, e, "position", nullptr), e, m.position)) {
		return false;
	}
	if (!parse_vec3(find_key(obj, e, "base_dir", nullptr), e, m.base_dir)) {
		return false;
	}
	if (!parse_vec3(find_key(obj, e, "axis1", nullptr), e, m.axis1)) {
		return false;
	}
	if (!parse_vec3(find_key(obj, e, "axis2", nullptr), e, m.axis2)) {
		return false;
	}
	m.base_dir = vunit(m.base_dir);
	m.axis1 = vunit(m.axis1);
	m.axis2 = vunit(m.axis2);

	float a2_min = shared_a2_min, a2_max = shared_a2_max;
	float a1_min = shared_a1_min, a1_max = shared_a1_max;
	float a2_rest = shared_a2_rest, a1_rest = shared_a1_rest;
	const bool have_lim = parse_angle_limits(obj, e, a2_min, a2_max, a1_min, a1_max) || have_shared_limits;
	const bool have_rest = parse_rest_angles(obj, e, a2_rest, a1_rest) || have_shared_rest;
	if (!have_lim || !have_rest) {
		return false;
	}
	if (a2_rest < a2_min || a2_rest > a2_max || a1_rest < a1_min || a1_rest > a1_max) {
		return false;
	}
	apply_angles(m, a2_min, a2_max, a1_min, a1_max, a2_rest, a1_rest);

	float thrust = 0.f;
	if (parse_float_key(obj, e, "thrust", thrust)) {
		m.thrust = thrust;
	}

	mass_kg = 0.f;
	float mass_g = 0.f;
	if (parse_float_key(obj, e, "mass_g", mass_g) && mass_g > 0.f) {
		mass_kg = mass_g / 1000.f;
	} else if (parse_float_key(obj, e, "mass_kg", mass_kg) && mass_kg > 0.f) {
		// already kg
	} else {
		const char *eng = find_key(obj, e, "engine", nullptr);
		if (eng && parse_float_key(eng, e, "mass_g", mass_g) && mass_g > 0.f) {
			mass_kg = mass_g / 1000.f;
		}
	}
	return true;
}

} // namespace

uint32_t crc32_ieee(const void *data, unsigned len)
{
	const uint8_t *p = static_cast<const uint8_t *>(data);
	uint32_t crc = 0xffffffffu;
	for (unsigned i = 0; i < len; i++) {
		crc ^= p[i];
		for (int b = 0; b < 8; b++) {
			if (crc & 1u) {
				crc = (crc >> 1) ^ 0xedb88320u;
			} else {
				crc >>= 1;
			}
		}
	}
	return ~crc;
}

// Allocator CoM. Motor mass sits at the gimbal pivots, which is the CoM the
// SIH allocator used on the caught G12 hop. The plant's nozzle-standoff CG
// is a different body and is not what that controller steered with.
Vec3 compose_com_at_pivots(const Vec3 &unloaded_com, float unloaded_mass_kg,
			   const Mount mounts[kMaxMounts], const float motor_mass_kg[kMaxMounts],
			   int n_mounts, float *loaded_mass_kg)
{
	float mtot = unloaded_mass_kg;
	Vec3 acc = vscale(unloaded_com, unloaded_mass_kg);
	for (int i = 0; i < n_mounts; i++) {
		const float mi = motor_mass_kg[i];
		if (mi <= 0.f) {
			continue;
		}
		mtot += mi;
		acc = vadd(acc, vscale(mounts[i].position, mi));
	}
	if (loaded_mass_kg) {
		*loaded_mass_kg = mtot;
	}
	if (mtot <= 1e-9f) {
		return unloaded_com;
	}
	return vscale(acc, 1.f / mtot);
}

bool parse_vehicle_json(const char *json, unsigned len, VehicleConfig &out)
{
	out = VehicleConfig{};
	out.status = VehicleLoadStatus::ParseError;
	if (!json || len == 0) {
		return false;
	}
	const char *e = json + len;
	const char *root = skip_ws(json, e);
	if (root >= e || *root != '{') {
		return false;
	}

	if (!parse_vec3(find_key(root, e, "center_of_mass", nullptr), e, out.unloaded_com)) {
		out.status = VehicleLoadStatus::MissingField;
		return false;
	}

	const char *veh = find_key(root, e, "vehicle", nullptr);
	float mass_g = 0.f;
	if (!veh || !parse_float_key(veh, e, "mass_g", mass_g) || mass_g <= 0.f) {
		out.status = VehicleLoadStatus::MissingField;
		return false;
	}
	out.unloaded_mass_kg = mass_g / 1000.f;

	const char *iner = find_key(veh, e, "inertia_diagonal_g_m2", nullptr);
	Vec3 iner_g{};
	if (!iner || !parse_vec3(iner, e, iner_g)) {
		out.status = VehicleLoadStatus::MissingField;
		return false;
	}
	out.inertia_diag_kgm2[0] = iner_g.x / 1000.f;
	out.inertia_diag_kgm2[1] = iner_g.y / 1000.f;
	out.inertia_diag_kgm2[2] = iner_g.z / 1000.f;
	if (out.inertia_diag_kgm2[0] <= 0.f || out.inertia_diag_kgm2[1] <= 0.f ||
	    out.inertia_diag_kgm2[2] <= 0.f) {
		out.status = VehicleLoadStatus::InvalidValue;
		return false;
	}

	float a2_min = 0.f, a2_max = 0.f, a1_min = 0.f, a1_max = 0.f;
	float a2_rest = 0.f, a1_rest = 0.f;
	const bool have_shared_limits = parse_angle_limits(root, e, a2_min, a2_max, a1_min, a1_max);
	const bool have_shared_rest = parse_rest_angles(root, e, a2_rest, a1_rest);
	if (have_shared_limits) {
		out.a2_min_deg = a2_min;
		out.a2_max_deg = a2_max;
		out.a1_min_deg = a1_min;
		out.a1_max_deg = a1_max;
	}
	if (have_shared_rest) {
		out.a2_rest_deg = a2_rest;
		out.a1_rest_deg = a1_rest;
	}

	float engine_mass_kg = 0.f;
	const char *engine = find_key(root, e, "engine", nullptr);
	if (engine) {
		float emg = 0.f;
		if (parse_float_key(engine, e, "mass_g", emg) && emg > 0.f) {
			engine_mass_kg = emg / 1000.f;
		}
	}

	const char *mounts_val = find_key(root, e, "mounts", nullptr);
	if (!mounts_val) {
		out.status = VehicleLoadStatus::MissingField;
		return false;
	}
	mounts_val = skip_ws(mounts_val, e);
	if (mounts_val >= e || *mounts_val != '[') {
		out.status = VehicleLoadStatus::ParseError;
		return false;
	}
	const char *arr_end = skip_object_or_array(mounts_val, e, '[', ']');
	if (!arr_end) {
		return false;
	}
	const char *p = skip_ws(mounts_val + 1, arr_end);
	int n = 0;
	while (p < arr_end - 1 && *p != ']' && n < kMaxMounts) {
		p = skip_ws(p, arr_end);
		if (p >= arr_end || *p != '{') {
			out.status = VehicleLoadStatus::ParseError;
			return false;
		}
		float mass_kg = 0.f;
		if (!parse_mount_object(p, e, out.mounts[n], mass_kg, a2_min, a2_max, a1_min, a1_max,
					a2_rest, a1_rest, have_shared_limits, have_shared_rest)) {
			out.status = VehicleLoadStatus::MissingField;
			return false;
		}
		if (mass_kg <= 0.f) {
			mass_kg = engine_mass_kg;
		}
		if (mass_kg <= 0.f) {
			out.status = VehicleLoadStatus::MissingField;
			return false;
		}
		out.motor_mass_kg[n] = mass_kg;
		if (out.mounts[n].thrust <= 0.f) {
			out.mounts[n].thrust = 12.f;
		}
		p = skip_object_or_array(p, e, '{', '}');
		if (!p) {
			return false;
		}
		p = skip_ws(p, arr_end);
		if (p < arr_end && *p == ',') {
			p++;
		}
		n++;
	}
	if (n != kMaxMounts) {
		out.status = VehicleLoadStatus::InvalidValue;
		return false;
	}
	out.n_mounts = n;

	if (!have_shared_limits) {
		out.a2_min_deg = rad2deg(out.mounts[0].min_angles[0]);
		out.a2_max_deg = rad2deg(out.mounts[0].max_angles[0]);
		out.a1_min_deg = rad2deg(out.mounts[0].min_angles[1]);
		out.a1_max_deg = rad2deg(out.mounts[0].max_angles[1]);
	}
	if (!have_shared_rest) {
		out.a2_rest_deg = rad2deg(out.mounts[0].rest_angles[0]);
		out.a1_rest_deg = rad2deg(out.mounts[0].rest_angles[1]);
	}

	out.loaded_com = compose_com_at_pivots(out.unloaded_com, out.unloaded_mass_kg,
					       out.mounts, out.motor_mass_kg, out.n_mounts,
					       &out.loaded_mass_kg);
	if (out.loaded_mass_kg <= 0.f || !isfinitef2(out.loaded_com.x) ||
	    !isfinitef2(out.loaded_com.y) || !isfinitef2(out.loaded_com.z)) {
		out.status = VehicleLoadStatus::InvalidValue;
		return false;
	}

	out.valid = true;
	out.status = VehicleLoadStatus::Ok;
	return true;
}

bool load_vehicle_config_file(const char *json_path, const char *crc_path, VehicleConfig &out)
{
	out = VehicleConfig{};
	out.status = VehicleLoadStatus::FileMissing;
	if (!json_path) {
		return false;
	}

	FILE *jf = fopen(json_path, "rb");
	if (!jf) {
		return false;
	}
	// Not on the stack: PX4 hp_default WQ is 2800 B and this runs from module init
	// before USB/MAVLink start. A 24 kB frame hardfaults and the Cube never enumerates.
	static char buf[kVehicleJsonMaxBytes + 1];
	const unsigned nread = (unsigned)fread(buf, 1, kVehicleJsonMaxBytes, jf);
	const bool too_big = (fgetc(jf) != EOF);
	fclose(jf);
	if (nread == 0 || too_big) {
		out.status = VehicleLoadStatus::ParseError;
		return false;
	}
	buf[nread] = '\0';

	if (!crc_path) {
		out.status = VehicleLoadStatus::CrcMissing;
		return false;
	}
	FILE *cf = fopen(crc_path, "r");
	if (!cf) {
		out.status = VehicleLoadStatus::CrcMissing;
		return false;
	}
	char crc_line[64]{};
	if (!fgets(crc_line, sizeof(crc_line), cf)) {
		fclose(cf);
		out.status = VehicleLoadStatus::CrcMissing;
		return false;
	}
	fclose(cf);

	unsigned long expected = 0;
	if (sscanf(crc_line, "crc32=%lx", &expected) != 1 &&
	    sscanf(crc_line, "%lx", &expected) != 1) {
		out.status = VehicleLoadStatus::CrcMissing;
		return false;
	}

	const uint32_t got = crc32_ieee(buf, nread);
	if (got != (uint32_t)expected) {
		out.status = VehicleLoadStatus::CrcMismatch;
		out.crc = got;
		return false;
	}

	if (!parse_vehicle_json(buf, nread, out)) {
		return false;
	}
	out.crc = got;
	g_loaded = out;
	return true;
}

const VehicleConfig &loaded_vehicle_config()
{
	return g_loaded;
}

} // namespace tvc
