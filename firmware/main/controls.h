// The dial (volume, steps of 5 %) and the mute switch, plus the server's set_volume and
// volume_boost, all applied on one task; the center button on another, started early.
#pragma once

#include "esp_err.h"

// Starts the controls task. Call after zl_start() went through.
esp_err_t controls_start(void);

// zakhar_link's on_set_volume: queued for the controls task (the callback must not block).
void controls_request_volume(int percent);

// zakhar_link's on_volume_boost: queued for the controls task (the callback must not block).
void controls_request_boost(int points);

// Watches the center button on a task of its own: a hold of FORGET_HOLD_MS calls on_hold, a press
// released before it calls on_short. Call it right after the Wi-Fi start: it must work with no
// network and no audio, so on_short must be harmless before zakhar_link starts.
esp_err_t controls_button_start(void (*on_hold)(void), void (*on_short)(void));
