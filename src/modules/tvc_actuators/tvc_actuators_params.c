/**
 * Servo a2 half-range for normalize [deg]
 *
 * @group TVC Actuators
 * @min 5
 * @max 45
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_A2_RANGE, 45.f);

/**
 * Servo a1 range geometric 0-to-fold [deg] (0=vertical, fold=90 → 90)
 *
 * PWM scale only. Mechanical travel is 0..90 (rest = 0°, vertical).
 *
 * @group TVC Actuators
 * @min 10
 * @max 120
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_A1_RANGE, 90.f);

/**
 * Mechanical a1 rest [deg]
 *
 * Disarmed / TVC_ZERO hold this angle (plus trim). Vertical, no hard stop.
 * Must match vehicle.json rest_angles_deg.a1.
 *
 * @group TVC Actuators
 * @min 0
 * @max 45
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_A1_REST, 0.f);

/**
 * Reverse channel bitmask (bit0=M0a2 ... bit5=M2a1)
 *
 * This airframe: 47 = 0b101111 (verified 2026-08-24). Invert both a1s
 * and M0/M1 a2. M2 a2 matches the default PWM sense. Pair with
 * tvc_control invert_side_roll + invert_yaw + M0 yaw a2 remap.
 *
 * @group TVC Actuators
 * @min 0
 * @max 63
 */
PARAM_DEFINE_INT32(TVC_SERVO_REV, 47);

/**
 * Pad zero-gimbal mode (hold rest + trim, PWM live disarmed)
 *
 * Holds all six gimbals at trimmed rest. Sets COM_PREARM_MODE=2 so MAIN
 * moves while disarmed. Denies arm / launch_ready / fire. Leave with
 * tvc_actuators nozero (or QGC TVC Zero done) and param save.
 *
 * @group TVC Actuators
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_ZERO, 0);

/**
 * M0 a2 rest trim [deg] (logical 0, MAIN 1)
 *
 * Added in geometric angle space before TVC_SERVO_REV. +trim moves rest
 * in the +a2 command direction.
 *
 * @group TVC Actuators
 * @min -15
 * @max 15
 * @decimal 1
 * @increment 0.5
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_TRIM_0, 0.f);

/**
 * M0 a1 rest trim [deg] (logical 1, MAIN 5)
 *
 * @group TVC Actuators
 * @min -15
 * @max 15
 * @decimal 1
 * @increment 0.5
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_TRIM_1, 0.f);

/**
 * M1 a2 rest trim [deg] (logical 2, MAIN 2)
 *
 * @group TVC Actuators
 * @min -15
 * @max 15
 * @decimal 1
 * @increment 0.5
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_TRIM_2, 0.f);

/**
 * M1 a1 rest trim [deg] (logical 3, MAIN 6)
 *
 * @group TVC Actuators
 * @min -15
 * @max 15
 * @decimal 1
 * @increment 0.5
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_TRIM_3, 0.f);

/**
 * M2 a2 rest trim [deg] (logical 4, MAIN 3)
 *
 * @group TVC Actuators
 * @min -15
 * @max 15
 * @decimal 1
 * @increment 0.5
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_TRIM_4, 0.f);

/**
 * M2 a1 rest trim [deg] (logical 5, MAIN 7)
 *
 * @group TVC Actuators
 * @min -15
 * @max 15
 * @decimal 1
 * @increment 0.5
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_TRIM_5, 0.f);
