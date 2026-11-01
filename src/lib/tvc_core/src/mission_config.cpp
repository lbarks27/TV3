#include "tvc/mission_config.hpp"

#include "tvc/math.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace tvc {

namespace {

MissionConfig g_loaded{};

const char *skip_ws(const char *p, const char *e)
{
	while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
		p++;
	}
	return p;
}

/** First `"key":` value pointer (flattened pack JSON has unique keys). */
const char *find_top_value(const char *json, const char *e, const char *key)
{
	const unsigned klen = (unsigned)strlen(key);
	const char *p = json;

	while (p < e) {
		const char *q = (const char *)memchr(p, '"', (size_t)(e - p));

		if (!q || q + klen + 2 > e) {
			return nullptr;
		}

		if (q[klen + 1] == '"' && memcmp(q + 1, key, klen) == 0) {
			const char *c = skip_ws(q + klen + 2, e);

			if (c < e && *c == ':') {
				return skip_ws(c + 1, e);
			}
		}

		p = q + 1;
	}

	return nullptr;
}

bool parse_number_val(const char *p, const char *e, float &out)
{
	if (!p) {
		return false;
	}

	p = skip_ws(p, e);
	char buf[64];
	unsigned n = 0;
	const char *q = p;

	if (q < e && *q == '-') {
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

	char *endp = nullptr;
	out = strtof(buf, &endp);
	return endp != buf && isfinitef2(out);
}

bool parse_float_field(const char *json, const char *e, const char *key, float &out)
{
	return parse_number_val(find_top_value(json, e, key), e, out);
}

bool parse_int_field(const char *json, const char *e, const char *key, int &out)
{
	float v = 0.f;

	if (!parse_number_val(find_top_value(json, e, key), e, v)) {
		return false;
	}

	out = (int)v;
	return true;
}

bool parse_string_field(const char *json, const char *e, const char *key, char *out, unsigned out_len)
{
	const char *p = find_top_value(json, e, key);

	if (!p || out_len == 0) {
		return false;
	}

	p = skip_ws(p, e);

	if (p >= e || *p != '"') {
		return false;
	}

	p++;
	unsigned n = 0;

	while (p < e && *p != '"' && n + 1 < out_len) {
		out[n++] = *p++;
	}

	out[n] = '\0';
	return n > 0;
}

} // namespace

bool parse_mission_json(const char *json, unsigned len, MissionConfig &out)
{
	out = MissionConfig{};
	out.status = MissionLoadStatus::ParseError;

	if (!json || len == 0) {
		return false;
	}

	const char *e = json + len;
	int mode = 0;

	if (!parse_int_field(json, e, "mode", mode)) {
		out.status = MissionLoadStatus::MissingField;
		return false;
	}

	if (mode < 0 || mode > 2) {
		out.status = MissionLoadStatus::InvalidValue;
		return false;
	}

	out.mode = mode;
	(void)parse_string_field(json, e, "name", out.name, kMissionNameMax);
	(void)parse_float_field(json, e, "hover_z", out.hover_z);
	(void)parse_float_field(json, e, "wp_e", out.wp_e);
	(void)parse_float_field(json, e, "wp_n", out.wp_n);
	(void)parse_float_field(json, e, "wp_u", out.wp_u);
	(void)parse_float_field(json, e, "capture_radius_m", out.capture_radius_m);
	(void)parse_float_field(json, e, "capture_max_speed_m_s", out.capture_max_speed_m_s);
	(void)parse_float_field(json, e, "capture_settle_s", out.capture_settle_s);
	(void)parse_float_field(json, e, "stage_radius_m", out.stage_radius_m);
	(void)parse_float_field(json, e, "stage_speed_m_s", out.stage_speed_m_s);
	(void)parse_float_field(json, e, "vertical_hold_s", out.vertical_hold_s);
	int has_loft = 0;

	if (parse_int_field(json, e, "has_loft", has_loft) && has_loft != 0) {
		out.has_loft = true;
		(void)parse_float_field(json, e, "loft_e", out.loft_e);
		(void)parse_float_field(json, e, "loft_n", out.loft_n);
		(void)parse_float_field(json, e, "loft_u", out.loft_u);
	}

	if (parse_float_field(json, e, "burn_end_s", out.burn_end_s) &&
	    parse_float_field(json, e, "pad_remain_s", out.pad_remain_s) &&
	    out.burn_end_s > 0.f) {
		out.has_burn_gate = true;
	}
	(void)parse_float_field(json, e, "descent_duration_s", out.descent_duration_s);
	(void)parse_float_field(json, e, "min_approach_time_s", out.min_approach_time_s);
	(void)parse_float_field(json, e, "safe_z", out.safe_z);
	(void)parse_float_field(json, e, "land_z", out.land_z);
	(void)parse_float_field(json, e, "safe_half_m", out.safe_half_m);
	(void)parse_float_field(json, e, "safe_vmax", out.safe_vmax);
	(void)parse_float_field(json, e, "force_weight", out.force_weight);
	float heading_deg = 0.f;
	out.heading_rad = NAN;

	if (parse_float_field(json, e, "heading_deg", heading_deg)) {
		out.heading_rad = heading_deg * 0.01745329252f;
	}

	float pos_kp = 0.f, pos_kd = 0.f, pos_tilt = 0.f;
	float alt_kp = 0.f, alt_kd = 0.f, alt_ki = 0.f;
	const bool have_pos = parse_float_field(json, e, "pos_kp", pos_kp) &&
			      parse_float_field(json, e, "pos_kd", pos_kd) &&
			      parse_float_field(json, e, "pos_tilt_deg", pos_tilt);
	const bool have_alt = parse_float_field(json, e, "alt_kp", alt_kp) &&
			      parse_float_field(json, e, "alt_kd", alt_kd) &&
			      parse_float_field(json, e, "alt_ki", alt_ki);

	if (have_pos) {
		out.pos_kp = pos_kp;
		out.pos_kd = pos_kd;
		out.pos_tilt_deg = pos_tilt;
		out.pos_kp_y = pos_kp;
		out.pos_kd_y = pos_kd;
		out.pos_tilt_y_deg = pos_tilt;
		(void)parse_float_field(json, e, "pos_kp_y", out.pos_kp_y);
		(void)parse_float_field(json, e, "pos_kd_y", out.pos_kd_y);
		(void)parse_float_field(json, e, "pos_tilt_y_deg", out.pos_tilt_y_deg);
		if (parse_float_field(json, e, "pos_ki", out.pos_ki)) {
			out.has_pos_ki = true;
			out.pos_ki_y = out.pos_ki;
			(void)parse_float_field(json, e, "pos_ki_y", out.pos_ki_y);
		}
		if (parse_float_field(json, e, "pos_i_limit", out.pos_i_limit)) {
			out.has_pos_i_limit = true;
		}
		if (parse_float_field(json, e, "pos_i_start", out.pos_i_start)) {
			out.has_pos_i_start = true;
		}
		if (parse_float_field(json, e, "lat_priority", out.lat_priority)) {
			out.has_lat_priority = true;
		}
		if (parse_float_field(json, e, "lat_lpf_hz", out.lat_lpf_hz)) {
			out.has_lat_lpf = true;
		}
	}
	(void)parse_float_field(json, e, "max_speed_m_s", out.max_speed_m_s);
	(void)parse_float_field(json, e, "max_speed_y_m_s", out.max_speed_y_m_s);
	(void)parse_float_field(json, e, "max_position_error_m", out.max_position_error_m);
	(void)parse_float_field(json, e, "max_position_error_y_m", out.max_position_error_y_m);
	int prefer_y = 0;
	if (parse_int_field(json, e, "prefer_lateral_y", prefer_y)) {
		out.prefer_lateral_y = prefer_y != 0;
	}

	if (have_alt) {
		out.alt_kp = alt_kp;
		out.alt_kd = alt_kd;
		out.alt_ki = alt_ki;
	}

	float alt_i_limit = 0.f;

	if (parse_float_field(json, e, "alt_i_limit", alt_i_limit) && alt_i_limit >= 0.f) {
		out.alt_i_limit = alt_i_limit;
		out.has_alt_i_limit = true;
	}

	float throttle_min = 0.f;

	if (parse_float_field(json, e, "throttle_min", throttle_min) &&
	    throttle_min > 0.f && throttle_min <= 1.f) {
		out.throttle_min = throttle_min;
	}

	float land_tilt = -1.f;

	if (parse_float_field(json, e, "land_tilt_deg", land_tilt) && land_tilt >= 0.f) {
		out.land_tilt_deg = land_tilt;
	}

	out.has_gains = have_pos && have_alt;
	out.valid = true;
	out.status = MissionLoadStatus::Ok;
	return true;
}

bool load_mission_config_file(const char *json_path, MissionConfig &out)
{
	out = MissionConfig{};
	out.status = MissionLoadStatus::FileMissing;

	if (!json_path) {
		return false;
	}

	FILE *f = fopen(json_path, "rb");

	if (!f) {
		return false;
	}

	// Same as vehicle_config: do not put this on the PX4 work-queue stack.
	static char buf[kMissionJsonMaxBytes + 1];
	const unsigned nread = (unsigned)fread(buf, 1, kMissionJsonMaxBytes, f);
	const bool too_big = (fgetc(f) != EOF);
	fclose(f);

	if (nread == 0 || too_big) {
		out.status = MissionLoadStatus::ParseError;
		return false;
	}

	buf[nread] = '\0';

	if (!parse_mission_json(buf, nread, out)) {
		return false;
	}

	g_loaded = out;
	return true;
}

const MissionConfig &loaded_mission_config()
{
	return g_loaded;
}

} // namespace tvc
