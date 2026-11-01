# Onboard ignition is disabled. Topic stays for logs / listeners.
# gpio_enabled / fire_asserted / pulse_active are always false.

uint64 timestamp                        # us

bool gpio_enabled                       # always 0 (no Cube fire rail)
bool fire_asserted                      # always 0
bool pulse_active                       # always 0
bool setup_ok                           # 1 = module alive, GPIO not used
uint8 last_error                        # 0=ok
