# Gimbal angles for 3 dual-axis mounts (radians)

uint64 timestamp                        # us

float32[6] angles_rad                   # M0a2,M0a1,M1a2,M1a1,M2a2,M2a1
bool saturated
bool timed_out
uint32 solve_us
float32 residual
