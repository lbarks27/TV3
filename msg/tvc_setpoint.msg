# Mission / outer-loop setpoint in ENU-sim frame

uint64 timestamp                        # us

float32[3] pos                          # m, ENU-sim
float32[3] vel                          # m/s, ENU-sim
float32 heading_rad                     # yaw about +Z_up; NaN if free
uint8 phase                             # 0=none, 1=approach, 2=landing, ...
float32 path_speed_ff                   # optional path speed feedforward
