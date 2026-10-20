// The speaker volume, percent: kept in NVS, applied through the DAC, reported to the server. The
// DAC plays it plus the server's boost, within 1..100 %; a volume of 0 stays muted.
#pragma once

#include <stdint.h>

// Loads the stored volume (50, the stock initial volume, when none is stored). Call after
// nvs_flash_init(), before the DAC setup reads volume_get().
void volume_init(void);
int volume_get(void);
// 0..100: applies, stores and reports it (zl_report_volume). After zl_start() went through, from
// one task.
void volume_set(int percent);
// The server's volume_boost, percent points -100..100, added on the DAC only: not stored, not reported,
// not shown on the LEDs. From the task that calls volume_set().
void volume_set_boost(int points);
// The music's share of the volume, Q16: what keeps the music at the dial under the boost
// (audio_core volume_music_scale). From any task.
uint32_t volume_music_gain(void);
