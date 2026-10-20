// The music ring's pure part: no atomics, no ESP headers. Implemented in audio_core.c and
// host-tested with it (make fw-voicepe-host-test); a header of its own because its user,
// music.cpp, is C++ and cannot include audio_core.h's C11 atomics.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Interleaved stereo int16 frames from one writer to one reader. head is the writer's next frame
// and tail the reader's, both 0..cap-1; one slot is kept free, so head == tail is empty and a full
// ring holds cap - 1 frames. Nothing here is shared between threads: the caller keeps head and
// tail where both sides see them, fills a copy, and stores back the index its side moved (the
// writer head, the reader tail).
typedef struct {
    int16_t *buf;  // cap * 2 samples
    uint32_t cap;
    uint32_t head;
    uint32_t tail;
} music_ring_t;

// Copies up to n frames from `in` (any alignment) at head, as many as fit, and moves head past
// them; returns the count.
size_t music_ring_write(music_ring_t *r, const void *in, size_t n);
// Copies up to max frames at tail to `out` and moves tail past them; returns the count.
size_t music_ring_read(music_ring_t *r, int16_t *out, size_t max);
// The flush a stream's end leaves for the reader: `to` is the head the ring had when the stream
// ended, its writer idle. When `to` lies within the frames held (tail..head), tail moves to it: the
// ended stream's frames are dropped, the frames a next stream wrote after it are kept.
void music_ring_flush(music_ring_t *r, uint32_t to);

#ifdef __cplusplus
}
#endif
