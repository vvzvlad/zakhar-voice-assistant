// The Voice PE's two I2S buses, both clocked by the XMOS with the ESP32-S3 as the slave.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

// After the XMOS answers (its clocks run then): enables the speaker bus, which sends silence until
// audio_start_playback(), and the microphone bus with its task, which fills the tap and runs the
// wake-word engine on channel 1 (and the stop word on channel 0 while it is raw).
esp_err_t audio_start(void);

// Starts the speaker task: zakhar_link's playback (zl_play_read) on both slots, silence when it
// runs dry. Call after zl_start() went through.
esp_err_t audio_start_playback(void);

// zakhar_link's microphone tap: 16 kHz, 2 channels interleaved (ch0 the XMOS AGC output, ch1 its
// noise-suppressed output at the wake-word gain, exactly the wake-word engine's input), positions
// in frames. While the speaker plays and echo_cancel is on, ch0 is the raw microphone 0 with the
// speaker's own playback cancelled, at gain 1; during an auto volume measurement it is the raw
// microphone 0 as read, at gain 1.
uint32_t audio_mic_position(void);
size_t audio_mic_read(uint32_t *pos, int16_t *out, size_t max_frames);

// The server's echo cancellation settings for the stop word, each -1 to keep the current one:
// level (echo_cancel) 0 off, 1..3 on, with the AEC's residual echo suppression from mild to
// strongest; mode (echo_cancel_mode) 0..3 SR low cost, SR high performance, FD low cost, FD high
// performance; filter (echo_cancel_filter) 1..8, its filter length. Taken from the next microphone
// chunk on, a new mode or filter length re-creating the AEC; 1, 2 and 4 until the first call.
void audio_set_echo_cancel(int level, int mode, int filter);

// The server's auto_volume: while on, every 15 s of quiet (nothing playing, no session open, no
// capture held) the mic task measures the room's noise on the raw channel 0 for about a second
// and reports it with zl_report_ambient(). Off until the first call; taken from the next
// microphone chunk on.
void audio_set_auto_volume(bool on);
