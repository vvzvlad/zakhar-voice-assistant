// The LED ring: three RGB zones in sysfs showing the Voice PE's indication (firmware/main/ui_core.h
// decides which). The setters are safe from any thread.
#pragma once

#include <stdbool.h>

#include "ui_core.h"

// Opens the LEDs and starts the ring's thread; after zl_start() (it polls zl_playing()).
void leds_start(void);
void leds_set_link(bool up);
void leds_set_phase(led_phase_t phase);
void leds_set_capture(bool on);
void leds_set_volume(int percent);
