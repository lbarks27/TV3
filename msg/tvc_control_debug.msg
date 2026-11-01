# High-rate control debug (log rate; may downsample telem)

uint64 timestamp                        # us

float32[3] tau_cmd
float32[3] tau_act
float32[3] force_body
float32[4] q
float32[4] q_ref
float32[3] omega
float32[3] att_integral
float32 alt_integral
uint8 mission_phase
