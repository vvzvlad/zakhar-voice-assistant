// The Voice PE's two I2S buses, both clocked by the XMOS with the ESP32-S3 as the slave.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// After the XMOS answers (its clocks run then): enables the speaker bus, which sends silence until
// audio_start_playback(), and the microphone bus with its task, which fills the tap and runs the
// wake-word engine on channel 1.
esp_err_t audio_start(void);

// Starts the speaker task: zakhar_link's playback (zl_play_read) on both slots, silence when it
// runs dry. Call after zl_start() went through.
esp_err_t audio_start_playback(void);

// zakhar_link's microphone tap: 16 kHz, 2 channels interleaved (ch0 the XMOS AGC output, ch1 its
// noise-suppressed output), positions in frames.
uint32_t audio_mic_position(void);
size_t audio_mic_read(uint32_t *pos, int16_t *out, size_t max_frames);
