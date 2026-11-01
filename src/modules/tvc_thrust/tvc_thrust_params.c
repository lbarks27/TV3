/**
 * Nominal constant thrust per mount when no curve loaded [N]
 *
 * @group TVC Thrust
 * @min 0
 * @unit N
 */
PARAM_DEFINE_FLOAT(TVC_THR_NOM, 12.f);

/**
 * Lit threshold [N]
 *
 * @group TVC Thrust
 * @min 0
 * @unit N
 */
PARAM_DEFINE_FLOAT(TVC_THR_LIT, 0.5f);

/**
 * Burnout threshold [N]
 *
 * @group TVC Thrust
 * @min 0
 * @unit N
 */
PARAM_DEFINE_FLOAT(TVC_THR_BO, 0.25f);

/**
 * Curve ignition delay mount 0 [s]
 *
 * @group TVC Thrust
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_THR_DLY0, 0.f);

/**
 * Curve ignition delay mount 1 [s]
 *
 * @group TVC Thrust
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_THR_DLY1, 0.f);

/**
 * Curve ignition delay mount 2 [s]
 *
 * @group TVC Thrust
 * @min 0
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_THR_DLY2, 0.f);

/**
 * Synthetic trapezoid peak thrust [N] when no CSV curve
 *
 * 0 = use TVC_THR_NOM constant during burn instead of trapezoid.
 *
 * @group TVC Thrust
 * @min 0
 * @unit N
 */
PARAM_DEFINE_FLOAT(TVC_THR_PEAK, 0.f);

/**
 * Synthetic trapezoid burn duration [s]
 *
 * @group TVC Thrust
 * @min 0.1
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_THR_BURN, 4.f);

/**
 * Synthetic trapezoid rise time [s]
 *
 * @group TVC Thrust
 * @min 0.01
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_THR_RISE, 0.3f);

/**
 * Synthetic trapezoid fall time [s]
 *
 * @group TVC Thrust
 * @min 0.01
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_THR_FALL, 0.4f);

/**
 * Load motor CSV curves from SD on start (1=yes)
 *
 * Paths: /fs/microsd/tvc/motor0.csv ... motor2.csv (time_s, thrust_n)
 *
 * @group TVC Thrust
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_THR_SDCURV, 1);
