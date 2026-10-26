// The board's sound card through tinyalsa: the microphones and the speaker's own signal in, the
// speaker out.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Opens the capture side and starts its thread: from its first frame it feeds the microphone tap
// and the wake-word engine (wake_word_init() first). False when it did not start (logged).
bool audio_start_capture(void);
// Starts the speaker thread, which plays zl_play_read() (zl_start() first). False when it did not
// start (logged).
bool audio_start_playback(void);
// The playback's Q16 gain, from any thread; unity until set.
void audio_set_gain(uint32_t gain);
// zakhar_link's microphone tap (zl_config_t mic_position / mic_read).
uint32_t audio_mic_position(void);
size_t audio_mic_read(uint32_t *pos, int16_t *out, size_t max_frames);
