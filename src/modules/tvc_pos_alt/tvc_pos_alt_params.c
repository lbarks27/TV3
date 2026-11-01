/**
 * Position P XY
 *
 * @group TVC Position
 * @min 0
 */
PARAM_DEFINE_FLOAT(TVC_POS_P, 0.4f);

/**
 * Position D XY
 *
 * @group TVC Position
 * @min 0
 */
PARAM_DEFINE_FLOAT(TVC_POS_D, 8.5f);

/**
 * Max tilt [deg]
 *
 * @group TVC Position
 * @min 1
 * @max 45
 * @unit deg
 */
PARAM_DEFINE_FLOAT(TVC_POS_TILT, 10.f);

/**
 * Altitude force P
 *
 * @group TVC Position
 * @min 0
 */
PARAM_DEFINE_FLOAT(TVC_ALT_P, 14.f);

/**
 * Altitude force D
 *
 * @group TVC Position
 * @min 0
 */
PARAM_DEFINE_FLOAT(TVC_ALT_D, 18.f);

/**
 * Altitude force I
 *
 * @group TVC Position
 * @min 0
 */
PARAM_DEFINE_FLOAT(TVC_ALT_I, 0.2f);

/**
 * Axial mix arm time after fire [s] (-1 = thrust-margin gate)
 *
 * -1 arms with the thrust-margin gate (the G12 hop). 0 arms immediately.
 * A positive value waits that many seconds after first motion.
 *
 * @group TVC Position
 * @min -1
 * @unit s
 */
PARAM_DEFINE_FLOAT(TVC_AXMIX_ARM, -1.f);
