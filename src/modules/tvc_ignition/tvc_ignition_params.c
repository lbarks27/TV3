/**
 * Unused (onboard fire GPIO removed)
 *
 * Kept so stored params from older images still parse.
 *
 * @group TVC Ignition
 * @min 10
 * @max 2000
 * @unit ms
 */
PARAM_DEFINE_INT32(TVC_IGN_PULSE, 200);

/**
 * Unused (onboard fire GPIO removed)
 *
 * Cube never drives an igniter. Leave 0.
 *
 * @group TVC Ignition
 * @boolean
 */
PARAM_DEFINE_INT32(TVC_IGN_GPIO_EN, 0);

/**
 * Unused (onboard fire GPIO removed)
 *
 * @group TVC Ignition
 * @min 1
 * @max 6
 */
PARAM_DEFINE_INT32(TVC_IGN_FIRE_CH, 1);

/**
 * Unused (onboard fire GPIO removed)
 *
 * @group TVC Ignition
 * @value 0 Active-low fire
 * @value 1 Active-high fire
 */
PARAM_DEFINE_INT32(TVC_IGN_FIRE_POL, 1);
