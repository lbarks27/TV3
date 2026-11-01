# Aggregated onboard flight log (flight_log_v1 schema, ENU-sim / FLU-sim)
# Published at 100 Hz by tvc_log for ULog / pad review.

uint64 timestamp                        # us

float32 time_s                          # seconds from tvc_log start
float32[4] q                            # wxyz, ENU-sim <- FLU-sim
float32[4] q_ref
float32[3] omega                        # rad/s, FLU-sim
float32[3] tau_cmd                      # N.m, FLU-sim
float32[3] tau_act
float32[6] gimbal_angles                # M0a2,M0a1,M1a2,M1a1,M2a2,M2a1 rad
float32[3] position                     # m, ENU-sim
float32[3] velocity                     # m/s, ENU-sim
float32[3] position_ref
float32[3] velocity_ref
float32[3] force_body                   # N, FLU-sim
float32[3] mount_thrusts                # N
float32 force_target_axial              # N
float32 force_weight
float32[3] att_integral
float32 alt_integral
float32 heading_ref_rad                 # NaN if free
uint8 mission_phase                     # 0 none, 1 approach, 2 landing
uint8 vehicle_state                     # tvc_vehicle_state.state
uint8 abort_reason
uint8 saturated
uint8 mix_armed
uint8 fire_latched
