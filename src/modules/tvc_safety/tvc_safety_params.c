/**
 * Allow mission start (0 = inhibited — default for all non-pad builds)
 *
 * Must be 1 to enter LAUNCH_READY. First motion (or HIL `tvc_safety fire`)
 * then latches BURN. Cube does not drive an igniter.
 *
 * @group TVC Safety
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_ALLOW_FIRE, 0);

/**
 * Automatic aborts during BURN / COAST (0 = off)
 *
 * 0: tilt, rate, geofence, allocator timeout, hard-land, and hang-fire
 * do not leave the flight. An explicit abort command still does.
 * 1: those checks request ABORT.
 *
 * @group TVC Safety
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_AUTO_ABORT, 0);

/**
 * Require RTK Fixed for arm
 *
 * @group TVC Safety
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_REQ_RTK, 1);

/**
 * Require GCS/telemetry heartbeat for arm / pre-burn
 *
 * @group TVC Safety
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_REQ_LINK, 1);

/**
 * Max tilt abort [deg] (body tilt from vertical)
 *
 * Checked in BURN / COAST. Also used for arm if gimbals not at rest.
 *
 * @group TVC Safety
 * @min 5
 * @max 90
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_MAX_TILT, 40.f);

/**
 * Max rate abort [deg/s] (body rate magnitude)
 *
 * @group TVC Safety
 * @min 10
 * @max 1000
 * @unit deg/s
 */
PARAM_DEFINE_FLOAT(TVC_MAX_RATE, 250.f);

/**
 * Geofence cylinder radius [m] (0 = disabled)
 *
 * Horizontal radius about local origin. In BURN with RTK lost, effective radius is 0.7x.
 *
 * @group TVC Safety
 * @min 0
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_GEOFENCE_R, 50.f);

/**
 * Hang-fire timeout after per-mount ignition delay [ms]
 *
 * If mount not lit by (delay + hangfire), treat as hang-fire.
 *
 * @group TVC Safety
 * @min 100
 * @max 5000
 * @unit ms
 */
PARAM_DEFINE_INT32(TVC_HANGFIRE_MS, 800);

/**
 * Abort on partial ignition (1 = ABORT if any hang-fire while another lit)
 *
 * @group TVC Safety
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_PART_IGN, 1);

/**
 * Hard-landing height AGL [m] (NED: dist_bottom or -z)
 *
 * BURN only while any mount lit / not yet burned out.
 *
 * @group TVC Safety
 * @min 0.02
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_HL_Z, 0.08f);

/**
 * Hard-landing downward speed threshold [m/s] (NED +down)
 *
 * Abort if height <= TVC_HL_Z and vz_ned >= this (fast descent into pad).
 *
 * @group TVC Safety
 * @min 0.5
 * @unit m/s
 */
PARAM_DEFINE_FLOAT(TVC_HL_VZ, 1.5f);

/**
 * Burnout debounce [ms] all mounts below burnout before COAST
 *
 * @group TVC Safety
 * @min 20
 * @max 1000
 * @unit ms
 */
PARAM_DEFINE_INT32(TVC_BO_DEB_MS, 80);

/**
 * COAST → LAND height AGL [m]
 *
 * @group TVC Safety
 * @min 0.05
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_LAND_Z, 0.15f);

/**
 * COAST → LAND speed limit [m/s]
 *
 * @group TVC Safety
 * @min 0.1
 * @unit m/s
 */
PARAM_DEFINE_FLOAT(TVC_LAND_V, 0.5f);

/**
 * Pre-burn link-loss cancel timeout [ms]
 *
 * ARMED: disarm if link lost this long. LAUNCH_READY is not cancelled
 * (ground igniter is independent of GCS).
 *
 * @group TVC Safety
 * @min 200
 * @max 5000
 * @unit ms
 */
PARAM_DEFINE_INT32(TVC_LINK_PRE, 1000);

/**
 * Minimum 5V rail voltage for rail_v_ok [V] (0 = skip check)
 *
 * Uses system_power.voltage5v_v when available.
 *
 * @group TVC Safety
 * @min 0
 * @unit V
 */
PARAM_DEFINE_FLOAT(TVC_RAIL_VMIN, 4.6f);

/**
 * Gimbal rest tolerance for gimbal_rest_ok [deg]
 *
 * Max |a2 - rest| and |a1 - rest| across mounts for arm gate (rest from vehicle.json; a1 rest is 0°).
 *
 * @group TVC Safety
 * @min 1
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_GIM_REST, 8.f);

/**
 * Consecutive allocator timeout samples before ABORT in BURN
 *
 * 0 = disable alloc abort (still reports alloc_ok for arm).
 *
 * @group TVC Safety
 * @min 0
 * @max 50
 */
PARAM_DEFINE_INT32(TVC_ALLOC_N, 5);

/**
 * Force commander arm (skip stock preflight) via magic param2
 *
 * Pattern A: tvc_safety owns TVC gates; stock MC pre-arm often blocks servo-only.
 * 1 = send force-arm (21196). 0 = normal arm request.
 *
 * @group TVC Safety
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_FORCE_ARM, 1);

/**
 * Mount 0 ignition delay [s] (hang-fire clock) — mirrors TVC_THR_DLY0 when set
 *
 * Prefer TVC_THR_DLY* if present; these exist so safety can compile independently.
 *
 * @group TVC Safety
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_IGN_DLY0, 0.f);

/**
 * Mount 1 ignition delay [s]
 *
 * @group TVC Safety
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_IGN_DLY1, 0.f);

/**
 * Mount 2 ignition delay [s]
 *
 * @group TVC Safety
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_IGN_DLY2, 0.f);

/**
 * Optional RC abort channel (1-based). 0 = disabled.
 *
 * If a receiver is present, PWM > 1800 on this channel requests TVC abort.
 * COM_RC_IN_MODE=1 still allows reading input_rc; this is not stick flying.
 *
 * @group TVC Safety
 * @min 0
 * @max 18
 */
PARAM_DEFINE_INT32(TVC_ABORT_CH, 0);

/**
 * First-motion upward velocity [m/s] (NED-up)
 *
 * LAUNCH_READY → BURN when this speed is held for TVC_MOT_N samples,
 * or height rise / upward accel trips first.
 *
 * @group TVC Safety
 * @min 0.05
 * @max 5
 * @unit m/s
 */
PARAM_DEFINE_FLOAT(TVC_MOT_VZ, 0.4f);

/**
 * First-motion height rise from pad capture [m]
 *
 * @group TVC Safety
 * @min 0.01
 * @max 1
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_MOT_DZ, 0.05f);

/**
 * First-motion upward acceleration [m/s^2] (NED-up, EKF az)
 *
 * G12 average T/W ~1.14 → ~1.4 m/s^2 net at 3.000 kg. Default 2.0 is above
 * that average; the ignition spike still clears it. Velocity and height gates remain.
 *
 * @group TVC Safety
 * @min 0.5
 * @max 20
 * @unit m/s^2
 */
PARAM_DEFINE_FLOAT(TVC_MOT_AZ, 2.0f);

/**
 * First-motion consecutive samples at 50 Hz
 *
 * 3 samples ≈ 60 ms. Rejects a single EKF / bump spike.
 *
 * @group TVC Safety
 * @min 1
 * @max 25
 */
PARAM_DEFINE_INT32(TVC_MOT_N, 3);
