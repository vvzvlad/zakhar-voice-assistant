// The log over Wi-Fi, as `esphome logs` gives it: `nc <speaker> 23` (make fw-logs) shows the
// ESP_LOG output live, whatever the zakhar server does. The serial output stays as it is.
// Separately, the zakhar server can ask for the log (`log_stream`): the lines then queue for
// zakhar_link, which sends them to the server.
#pragma once

#include <stdbool.h>
#include <stddef.h>

// Call once the network is up. A failure is logged; the serial log goes on regardless.
void net_log_start(void);

// Lines queue for the server while on; off empties the queue. Off from boot. Safe from any task.
void net_log_forward(bool on);

// Copies whole queued lines into buf without blocking, up to max bytes (at least 512, the longest
// line), and returns the bytes copied; no NUL is added. For one consumer task.
size_t net_log_take(char *buf, size_t max);
