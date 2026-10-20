// The speaker volume, percent: kept in NVS, applied through the DAC, reported to the server. The
// music plays at it, zakhar's replies at it plus the server's boost, within 1..100 %; a volume of
// 0 stays muted.
#pragma once

#include <stdint.h>

// Loads the stored volume (50, the stock initial volume, when none is stored). Call after
// nvs_flash_init(), before the DAC setup reads volume_get().
void volume_init(void);
int volume_get(void);
// 0..100: applies, stores and reports it (zl_report_volume). After zl_start() went through, from
// one task.
void volume_set(int percent);
// The server's volume_boost, percent points -100..100, for the replies only: not stored, not
// reported, not shown on the LEDs. From the task that calls volume_set().
void volume_set_boost(int points);
// What brings each stream from the DAC down to its own level, Q16 (audio_core volume_levels):
// the music to the volume, the replies to the volume plus the boost. From any task.
uint32_t volume_music_gain(void);
uint32_t volume_voice_gain(void);
