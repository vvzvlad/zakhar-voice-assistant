// The music player: a Sendspin player (sendspin-cpp) that Music Assistant finds by mDNS
// (`_sendspin._tcp`) and connects to. Its stream (48 kHz stereo 16-bit) goes through a ring the
// speaker task mixes under zakhar's playback (audio.c).
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// The network is up (an IP) or down: kept by the Wi-Fi/IP event handler, from before music_start().
void music_set_network(bool up);

// Allocates the ring and starts the player's task, which starts the client, advertises it and runs
// its loop. After zl_start() (mDNS is up) and with the network up.
esp_err_t music_start(void);

// A stream is on: from its start until its end.
bool music_active(void);

// Up to max_frames stereo frames of the stream for the speaker task, never waiting; the count
// copied, 0 while no stream is on.
size_t music_read(int16_t *out, size_t max_frames);

// The speaker task wrote `frames` frames music_read() gave it into the I2S DMA queue.
void music_played(size_t frames);

// The music's gain, Q16 (65536 = 1.0): Music Assistant's volume, 0 while it mutes.
uint32_t music_gain(void);

#ifdef __cplusplus
}
#endif
