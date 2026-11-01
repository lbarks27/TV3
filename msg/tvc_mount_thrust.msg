# Per-mount thrust estimates (N)

uint64 timestamp                        # us

float32[3] thrust                        # N
bool[3] valid
uint8[3] source                         # 1=curve 3=zero
float32 burn_time_s
bool[3] lit
bool[3] burned_out
