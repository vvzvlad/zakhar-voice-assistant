// The dial (volume, steps of 5 %) and the mute switch, plus the server's set_volume, all applied on
// one task.
#pragma once

#include "esp_err.h"

// Starts the controls task. Call after zl_start() went through.
esp_err_t controls_start(void);

// zakhar_link's on_set_volume: queued for the controls task (the callback must not block).
void controls_request_volume(int percent);
