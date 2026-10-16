// The log over Wi-Fi, as `esphome logs` gives it: `nc <speaker> 23` (make fw-logs) shows the
// ESP_LOG output live, whatever the zakhar server does. The serial output stays as it is.
#pragma once

// Call once the network is up. A failure is logged; the serial log goes on regardless.
void net_log_start(void);
