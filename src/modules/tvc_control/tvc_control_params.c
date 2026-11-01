/**
 * @file tvc_control_params.c
 * Attitude + mixer parameters for TVC vehicle.
 */

/**
 * Attitude roll P
 * @group TVC Control
 * @min 0
 * @max 100
 */
PARAM_DEFINE_FLOAT(TVC_ATT_RP, 12.f);

/**
 * Attitude pitch P
 * @group TVC Control
 * @min 0
 * @max 100
 */
PARAM_DEFINE_FLOAT(TVC_ATT_PP, 12.f);

/**
 * Attitude yaw P
 * @group TVC Control
 * @min 0
 * @max 100
 */
PARAM_DEFINE_FLOAT(TVC_ATT_YP, 8.f);

/**
 * Attitude roll D
 * @group TVC Control
 * @min 0
 * @max 50
 */
PARAM_DEFINE_FLOAT(TVC_ATT_RD, 6.f);

/**
 * Attitude pitch D
 * @group TVC Control
 * @min 0
 * @max 50
 */
PARAM_DEFINE_FLOAT(TVC_ATT_PD, 6.f);

/**
 * Attitude yaw D
 * @group TVC Control
 * @min 0
 * @max 50
 */
PARAM_DEFINE_FLOAT(TVC_ATT_YD, 2.f);

/**
 * Attitude roll I
 *
 * Matches the aligned G12 flight profile. 0 disables the axis.
 * @group TVC Control
 * @min 0
 * @max 5
 */
PARAM_DEFINE_FLOAT(TVC_ATT_RI, 0.5f);

/**
 * Attitude pitch I
 * @group TVC Control
 * @min 0
 * @max 5
 */
PARAM_DEFINE_FLOAT(TVC_ATT_PI, 0.5f);

/**
 * Attitude yaw I
 * @group TVC Control
 * @min 0
 * @max 5
 */
PARAM_DEFINE_FLOAT(TVC_ATT_YI, 0.1f);

/**
 * Torque limit [N·m]
 * @group TVC Control
 * @min 0
 * @max 100
 */
PARAM_DEFINE_FLOAT(TVC_TAU_LIM, 15.f);

/**
 * Mixer force weight (0 = torque-only)
 * @group TVC Control
 * @min 0
 * @max 10
 */
PARAM_DEFINE_FLOAT(TVC_FORCE_W, 0.65f);

/**
 * Actuator slew rate [deg/s]
 * @group TVC Control
 * @min 1
 * @max 2000
 */
PARAM_DEFINE_FLOAT(TVC_SLEW_DPS, 300.f);

/**
 * Loaded vehicle mass [kg]. Sentinel -1 blocks arming until set.
 *
 * Leftover identity param — arm is gated on SD vehicle.json + CRC.
 * G12 flight article: unloaded 2.6056 kg, loaded 3.0556 kg.
 * Ignored when vehicle.json loads.
 *
 * @group TVC Control
 * @min -1
 * @max 100
 * @unit kg
 */
PARAM_DEFINE_FLOAT(TVC_MASS, -1.f);

/**
 * CoM Z in body FLU-sim [m]
 *
 * Ignored when /fs/microsd/tvc/vehicle.json loads.
 * Fallback is the 2026-10-24 scale measurement, 18.5 in above the origin.
 * @group TVC Control
 * @min -2
 * @max 2
 * @unit m
 */
PARAM_DEFINE_FLOAT(TVC_COM_Z, 0.4699f);

/**
 * Control loop rate [Hz]
 * @group TVC Control
 * @min 50
 * @max 200
 * @unit Hz
 */
PARAM_DEFINE_INT32(TVC_CTRL_HZ, 100);

/**
 * Onboard bench attitude hold (no fire)
 *
 * 1: run tvc_control closed-loop in PREFLIGHT, fake hover thrust,
 *    hold level and freeze heading at enable, mix a1 from throttle
 *    (stick up = a1 rest 0°, stick down = fold 90°), clamp a2. Pair with
 *    COM_PREARM_MODE=2 so MAIN servos are live while disarmed.
 *    `tvc_control bench` sets both and recaptures heading.
 * 0: flight gates (default)
 *
 * @group TVC Control
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_BENCH, 0);

/**
 * Bench axial throttle (a1 fold via mixer)
 *
 * -1: use joystick/RC throttle if valid, else a1 rest
 *  0: full fold (min axial, a1 = 90°)
 *  1: rest (a1 = 0°, vertical, max axial)
 *
 * @group TVC Control
 * @min -1
 * @max 1
 */
PARAM_DEFINE_FLOAT(TVC_BENCH_THR, -1.f);

/**
 * Max |a2| from roll and pitch [deg]
 *
 * Non-heading share of differential a2. Heading is added separately
 * and is limited by TVC_A2_YAW. Horn scale stays TVC_A2_RANGE.
 *
 * @group TVC Control
 * @min 5
 * @max 45
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_A2_MAX, 30.f);

/**
 * Max |a2| from heading [deg]
 *
 * Yaw's share of the final a2 command, on top of the roll/pitch share.
 *
 * @group TVC Control
 * @min 0
 * @max 15
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_A2_YAW, 5.f);
