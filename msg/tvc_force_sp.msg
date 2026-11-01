# Axial force target for gimbal mixer (body +Z_sim)

uint64 timestamp                        # us

float32 f_axial_n
float32 az_cmd                          # vertical accel demand [m/s^2], ENU +up (throttle-like)
float32 force_weight
bool mix_armed
bool safe_land                          # no longer latched; loops stay closed through the pad
