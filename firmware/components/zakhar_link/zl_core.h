// The pure core of zakhar_link: no ESP headers, host-tested (make fw-host-test). Message
// encode/decode over cJSON, the session filter and the playback queue accounting. Nothing here
// locks; the transport (zakhar_link.c) serialises every call.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "zakhar_link.h"

#define ZL_MIC_RATE 16000                       // the microphone tap
#define ZL_FRAME_FRAMES (ZL_MIC_RATE * 32 / 1000)  // 32 ms of microphone audio per binary frame
#define ZL_KIND_MIC 0x01      // binary frame kinds: the first byte
#define ZL_KIND_PLAY 0x02

// ---------------------------------------------------------------- messages, server -> device

typedef enum {
    ZL_MSG_CONFIG,
    ZL_MSG_STATE,
    ZL_MSG_TRANSCRIPT,
    ZL_MSG_REPLY,
    ZL_MSG_PLAY_START,
    ZL_MSG_PLAY_END,
    ZL_MSG_PLAY_STOP,
    ZL_MSG_SET_VOLUME,
    ZL_MSG_WAKE_PEAK_STREAM,
    ZL_MSG_CAPTURE,
    ZL_MSG_DETECT,
} zl_msg_type_t;

typedef struct {
    zl_msg_type_t type;
    bool has_session;
    uint32_t session;
    zl_state_t state;     // STATE
    const char *text;     // STATE (NULL when absent), TRANSCRIPT and REPLY ("" when absent)
    uint32_t id;          // PLAY_START, PLAY_END
    uint8_t wake, vad;    // CONFIG, as uint8 cutoffs
    int value;            // SET_VOLUME, percent
    bool on;              // WAKE_PEAK_STREAM, DETECT
    uint32_t capture_ms;  // CAPTURE
    void *json;           // the parsed object that `text` points into
} zl_msg_t;

// One text frame. False for anything the device ignores: not a JSON object, an unknown type, a
// known field of the wrong type or out of its range. On true, release with zl_msg_free().
bool zl_decode(const char *data, size_t len, zl_msg_t *m);
void zl_msg_free(zl_msg_t *m);

// ---------------------------------------------------------------- messages, device -> server
// Each returns a NUL-terminated JSON text (NULL when out of memory); release with zl_free().

char *zl_encode_hello(const zl_config_t *cfg, int volume);
char *zl_encode_wake(uint32_t session, const char *word);
char *zl_encode_played(uint32_t id, bool dropped);
char *zl_encode_volume(int percent);
char *zl_encode_wake_peak(int percent);
char *zl_encode_audio_end(const char *reason);
void zl_free(char *json);

// ---------------------------------------------------------------- numbers

const char *zl_state_name(zl_state_t state);  // the wire value: "listening" ...
uint8_t zl_cutoff_u8(double percent);  // trunc(percent / 100 * 255)
int zl_peak_percent(uint8_t peak);     // 0..255 -> 0..100, rounded
uint32_t zl_backoff_next(uint32_t s);  // 1, 2, 4 ... 30 seconds

// ---------------------------------------------------------------- the voice session

typedef struct {
    uint32_t session;  // the last wake's number, 0 before the first
    bool open;         // from the wake to its idle or error
    bool mic;          // the session's audio streams: from the wake to its first state other than listening
    bool capture;      // a capture streams
    bool detect_off;   // a `detect` turned wake words off; zero, so on, with every connection
} zl_sess_t;

// What a wake word does: starts a session, or is refused and why.
typedef enum {
    ZL_WAKE_START,
    ZL_WAKE_DETECT_OFF,  // a `detect` turned detection off
    ZL_WAKE_CAPTURE,     // a capture runs
} zl_wake_t;

// What a wake word would do now; changes nothing.
zl_wake_t zl_sess_wake_check(const zl_sess_t *s);

// A wake word: refused while detection is off or a capture runs (as zl_sess_wake_check says);
// otherwise the next session starts (new number, open, the mic on) and replaces the current one.
zl_wake_t zl_sess_wake(zl_sess_t *s);

// A `detect`: wake words start sessions (true) or are refused (false). A capture does not care.
void zl_sess_detect(zl_sess_t *s, bool on);

// The session filter: a message without `session` passes, one with it only for the current one.
bool zl_sess_accepts(const zl_sess_t *s, const zl_msg_t *m);

// A `state` that passed the filter: anything but listening stops the mic, idle and error close.
void zl_sess_state(zl_sess_t *s, zl_state_t state);

// A `capture`: false (busy) while a session is open; otherwise the capture starts.
bool zl_sess_capture(zl_sess_t *s);

// The connection is gone: everything stops, and detection is on again for the next connection.
// Reports whether a session was open and a capture ran.
void zl_sess_lost(zl_sess_t *s, bool *was_open, bool *was_capture);

// ---------------------------------------------------------------- the playback queue

#define ZL_PLAYQ_STREAMS 8  // streams queued at once (the chime, a filler and the answer fit)

typedef struct {
    uint32_t id;
    size_t left;  // its samples still in the buffer
    bool ended;   // play_end arrived
} zl_stream_t;

// A ring of mono samples holding streams back to back. Only the newest stream can be receiving
// (between play_start and play_end); 0x02 payload bytes go to it, anything else is discarded.
typedef struct {
    int16_t *buf;
    size_t cap, head, count;
    zl_stream_t streams[ZL_PLAYQ_STREAMS];  // in buffer order, the oldest first
    size_t nstreams;
    bool receiving;
    bool has_odd;  // a frame ended inside a sample: its low byte waits for the next one
    uint8_t odd;
    uint32_t done[ZL_PLAYQ_STREAMS];  // streams whose last sample is out, to report `played`
    size_t ndone;
    size_t overflow;  // samples discarded since the buffer was full
} zl_playq_t;

void zl_playq_init(zl_playq_t *q, int16_t *buf, size_t cap);
// play_start: false when no slot is free, queued and done streams share ZL_PLAYQ_STREAMS (the
// stream is then dropped). A stream still receiving is ended first.
bool zl_playq_start(zl_playq_t *q, uint32_t id);
// 0x02 payload bytes, little-endian int16, split anywhere.
void zl_playq_write(zl_playq_t *q, const uint8_t *data, size_t len);
// play_end of the receiving stream; any other id is ignored.
void zl_playq_end(zl_playq_t *q, uint32_t id);
// Copies up to max samples out in stream order; a stream whose last sample went out after its
// play_end moves to `done`.
size_t zl_playq_read(zl_playq_t *q, int16_t *out, size_t max);
// play_stop or a wake: empties the buffer; writes the ids of the dropped streams to `dropped`
// (ZL_PLAYQ_STREAMS slots) and returns their count. `done` is kept.
size_t zl_playq_flush(zl_playq_t *q, uint32_t *dropped);
// Moves the `done` ids into ids (ZL_PLAYQ_STREAMS slots); returns their count.
size_t zl_playq_take_done(zl_playq_t *q, uint32_t *ids);
