# TVC vehicle high-level state (sim-frame mission FSM)

uint64 timestamp                        # us

uint8 STATE_DISARMED = 0
uint8 STATE_PREFLIGHT = 1
uint8 STATE_ARMED = 2
uint8 STATE_LAUNCH_READY = 3
uint8 STATE_BURN = 4
uint8 STATE_COAST = 5
uint8 STATE_LAND = 6
uint8 STATE_ABORT = 7
uint8 STATE_SAFE = 8

uint8 state
uint8 abort_reason
bool arm_ok
bool ignite_ok                          # LAUNCH_READY and first-motion will start the mission
bool fire_latched                       # mission / burn clock started (first motion or HIL fire)
bool origin_set                         # pad captured in the local NED frame
float32 pad_x                           # NED north of the landing pad [m]
float32 pad_y                           # NED east of the landing pad [m]
float32 pad_z                           # NED down of the landing pad [m]
