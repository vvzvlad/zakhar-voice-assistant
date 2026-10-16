// The LED ring: the device's indication (ui_core's ind_choose), drawn from the Wi-Fi provisioning,
// the network, the zakhar_link connection, the voice session, a capture, the playback and the mute
// state. Every setter is safe from any task and does not block.
#pragma once

#include <stdbool.h>

#include "ui_core.h"

void leds_start(void);
void leds_set_provisioning(bool on);  // waiting for a Wi-Fi network over BLE provisioning
void leds_set_network(bool up);  // Wi-Fi has an IP
void leds_set_link(bool up);     // zakhar_link is connected
void leds_set_phase(led_phase_t phase);
void leds_set_capture(bool on);
void leds_set_playing(bool on);  // playback audio is going out
void leds_set_muted(bool on);    // the mute switch
void leds_set_volume(int percent);
