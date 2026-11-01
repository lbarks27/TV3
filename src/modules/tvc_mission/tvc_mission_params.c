/**
 * Mission mode: 0=hold current, 1=hover, 2=real_landing
 *
 * Packed /tvc/mission.json overrides this when valid.
 *
 * @group TVC Mission
 * @value 0 Hold current
 * @value 1 Hover
 * @value 2 Real landing
 * @min 0
 * @max 2
 */
PARAM_DEFINE_INT32(TVC_MIS_MODE, 0);

/**
 * Hover altitude [m] ENU-sim
 *
 * @group TVC Mission
 * @min 0
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_HOVER_Z, 10.f);

/**
 * Approach waypoint east [m]
 *
 * @group TVC Mission
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_WP_E, 5.f);

/**
 * Approach waypoint up [m]
 *
 * @group TVC Mission
 * @min 0
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_WP_U, 5.f);

/**
 * Capture radius [m]
 *
 * @group TVC Mission
 * @min 0.1
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_CAP_R, 1.5f);

/**
 * Landing track duration [s]
 *
 * @group TVC Mission
 * @min 0.5
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_LAND_T, 7.f);

/**
 * Min approach time [s]
 *
 * @group TVC Mission
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_MIN_APPR, 2.f);
