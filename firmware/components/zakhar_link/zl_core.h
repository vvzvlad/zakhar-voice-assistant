// The pure core of zakhar_link: no ESP headers, host-tested (make fw-host-test). Message
// encode/decode over cJSON, the session filter, the playback queue accounting and the model
// receiver's. Nothing here locks; the transport (zakhar_link.c) serialises every call.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "zakhar_link.h"

#define ZL_MIC_RATE 16000                       // the microphone tap
#define ZL_FRAME_FRAMES (ZL_MIC_RATE * 32 / 1000)  // 32 ms of microphone audio per binary frame
#define ZL_KIND_MIC 0x01      // binary frame kinds: the first byte
#define ZL_KIND_PLAY 0x02
#define ZL_KIND_MODEL 0x03
#define ZL_KIND_CAPTURE 0x04  // then the u32 LE offset of its first frame, then the samples
// A held capture that cannot be sent is dropped this long after its last frame was recorded.
#define ZL_CAPTURE_KEEP_FRAMES (ZL_MIC_RATE * 2)

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
    ZL_MSG_CAPTURE_RESUME,
    ZL_MSG_CAPTURE_STOP,
    ZL_MSG_DETECT,
    ZL_MSG_MODEL,
    ZL_MSG_LOG_STREAM,
    ZL_MSG_VOLUME_BOOST,
} zl_msg_type_t;

typedef struct {
    zl_msg_type_t type;
    bool has_session;
    uint32_t session;
    zl_state_t state;     // STATE
    const char *text;     // STATE (NULL when absent), TRANSCRIPT and REPLY ("" when absent)
    bool duck;            // STATE: false only when it carries `duck: false`
    uint32_t id;          // PLAY_START, PLAY_END, CAPTURE (> 0), CAPTURE_RESUME, CAPTURE_STOP
    uint8_t wake, vad;    // CONFIG, as uint8 cutoffs
    int play_prime_ms;    // CONFIG: 0..2000, the playback reserve
    bool auto_volume;     // CONFIG
    int value;            // SET_VOLUME, percent; VOLUME_BOOST, percent points -100..100
    bool on;              // WAKE_PEAK_STREAM, DETECT, LOG_STREAM
    uint32_t capture_ms;  // CAPTURE
    uint32_t from;        // CAPTURE_RESUME: the offset of the first frame the server lacks
    zl_model_t model;     // MODEL: its kind
    const char *name;     // MODEL
    uint32_t size;        // MODEL: the bytes its 0x03 frames carry, > 0
    void *json;           // the parsed object that `text` and `name` point into
} zl_msg_t;

// One text frame. False for anything the device ignores: not a JSON object, an unknown type, a
// known field of the wrong type or out of its range, a `config` that lacks any of its fields. On
// true, release with zl_msg_free().
bool zl_decode(const char *data, size_t len, zl_msg_t *m);
void zl_msg_free(zl_msg_t *m);

// ---------------------------------------------------------------- messages, device -> server
// Each returns a NUL-terminated JSON text (NULL when out of memory); release with zl_free().

// capture: the held capture's id, 0 when none is held. uptime_s: seconds since the device booted.
// reset: why it booted, the wire value ("poweron" ...).
char *zl_encode_hello(const zl_config_t *cfg, int volume, uint32_t capture, uint32_t uptime_s, const char *reset);
char *zl_encode_wake(uint32_t session, const char *word, bool music);
// A stream that played out carries `dry` and `gap_ms`; a dropped one only `dropped`.
char *zl_encode_played(uint32_t id, bool dropped, uint32_t dry, uint32_t gap_ms);
char *zl_encode_volume(int percent);
char *zl_encode_wake_peak(int percent);
char *zl_encode_audio_end(const char *reason, uint32_t id);
char *zl_encode_model_loaded(zl_model_t kind, const char *name, bool ok);
// len bytes of log lines, not NUL-terminated.
char *zl_encode_log(const char *text, size_t len);
// The room's noise level, dBFS, as the caller rounded it.
char *zl_encode_ambient(double dbfs);
char *zl_encode_button(void);

// The device's health for `stats`. A field whose has_ flag is false is left out.
typedef struct {
    bool has_rssi;
    int rssi;      // dBm of the access point
    int ncpu;      // cpu0 .. of the first ncpu cores go out (ESP chips have one or two)
    int cpu[2];    // each core's load, percent
    bool has_temp;
    double temp;   // the chip's temperature, °C; goes out rounded to one decimal
    uint32_t heap, heap_min, heap_block;  // internal RAM: free, lowest free, largest free block
    bool has_psram;
    uint32_t psram, psram_min;
    uint32_t wifi_drops;  // Wi-Fi disconnect events since zl_start() at boot, failed reconnects included
    int wifi_reason;      // the last one's reason code: goes out only after one (wifi_drops > 0)
} zl_stats_t;
char *zl_encode_stats(const zl_stats_t *st);
void zl_free(char *json);

// ---------------------------------------------------------------- numbers

const char *zl_state_name(zl_state_t state);  // the wire value: "listening" ...
const char *zl_model_name(zl_model_t kind);   // the wire value: "wake", "vad"
uint8_t zl_cutoff_u8(double percent);  // trunc(percent / 100 * 255)
int zl_peak_percent(uint8_t peak);     // 0..255 -> 0..100, rounded
uint32_t zl_backoff_next(uint32_t s);  // 1, 2, 4 ... 30 seconds
// Whether `limit` ticks or more passed from `heard` to `now`. The difference is signed: `heard`
// can be stamped by the client's task after `now` was read, and that is not silence.
bool zl_silent(uint32_t now, uint32_t heard, uint32_t limit);
// Whether a capture that started at microphone position `start` has its `frames` frames by `pos`
// (positions count frames written and wrap).
bool zl_capture_done(uint32_t pos, uint32_t start, uint32_t frames);
// Whether such a capture, held but not being sent, is to be dropped by `pos`: ZL_CAPTURE_KEEP_FRAMES
// after its last frame was recorded, before that frame leaves the tap.
bool zl_capture_expired(uint32_t pos, uint32_t start, uint32_t frames);
// Writes v at p as 4 little-endian bytes (a capture frame's offset).
void zl_put_u32le(uint8_t *p, uint32_t v);

// ---------------------------------------------------------------- the voice session

typedef struct {
    uint32_t session;   // the last session the server took (listening), 0 before the first
    uint32_t proposed;  // a wake the server has not ruled on yet; 0 = none. Its audio streams meanwhile
    uint32_t last;      // the last proposed number: numbers are never reused
    bool open;          // the current session: from its listening to its idle or error
    bool quiet;         // its last shown listening carried `duck: false`: the music does not duck
    bool mic;           // the current session's audio streams: from its listening, or a follow-up one,
                        // to its next state other than listening
    bool capture;       // a capture is held: from its `capture` until it ends; a lost link keeps it
    bool detect_off;    // a `detect` turned wake words off; zero, so on, with every connection
} zl_sess_t;

// What a wake word does: proposes a session, or is refused and why.
typedef enum {
    ZL_WAKE_START,
    ZL_WAKE_DETECT_OFF,  // a `detect` turned detection off
    ZL_WAKE_CAPTURE,     // a capture runs
    ZL_WAKE_MIC,         // the audio of the session or of a proposed one still streams: its phrase is
                         // still being said
} zl_wake_t;

// What a wake word would do now; changes nothing.
zl_wake_t zl_sess_wake_check(const zl_sess_t *s);

// A wake word: refused while detection is off, a capture runs, the session's mic still streams or a
// wake is proposed already (as zl_sess_wake_check says) — a detection inside the phrase must not cut
// it off; otherwise it proposes the next number (`proposed`, which `wake` carries) and its audio
// streams. The current session stays current until the server takes the proposal.
zl_wake_t zl_sess_wake(zl_sess_t *s);

// A `detect`: wake words propose sessions (true) or are refused (false). A capture does not care.
void zl_sess_detect(zl_sess_t *s, bool on);

// The session filter: a message without `session` passes, one with it only for the current or the
// proposed one. play_start and play_end pass whatever their `session`: only the server's play_stop
// drops the playback, which it sends once it takes a wake: at once when its wake-word check is off
// or nothing is going, otherwise once that check confirms it.
bool zl_sess_accepts(const zl_sess_t *s, const zl_msg_t *m);

// A `state` that passed the filter; returns whether it is shown (on_state). Of the proposed session:
// listening takes the proposal (it becomes the current session, open, its mic on), anything else
// drops it unshown. Of the current one: anything but listening stops its mic, idle and error close
// it; a listening of the open session whose mic is off (a follow-up) turns the mic on again, and
// *rewind is true only when that starts the stream (no proposal streams), so the caller sets where
// its audio starts.
bool zl_sess_state(zl_sess_t *s, const zl_msg_t *m, bool *rewind);

// A `capture`: false (busy) while a session is open or a wake is proposed; otherwise the capture is
// held, in place of one held already.
bool zl_sess_capture(zl_sess_t *s);

// Whether the session's microphone audio streams: the current session's mic is on or a wake is
// proposed (a held capture streams on its own terms).
bool zl_sess_streams(const zl_sess_t *s);

// Whether the microphone is in use for the server: a session is open, a wake is proposed or a
// capture is held.
bool zl_sess_busy(const zl_sess_t *s);

// Whether the device's own music ducks: a session is open whose last listening did not carry
// `duck: false`, or a capture is held; not while a wake is only proposed.
bool zl_sess_ducks(const zl_sess_t *s);

// The connection is gone: the session ends, a proposal is dropped, a held capture stays held, and
// detection is on again for the next connection. Reports whether a session was open.
void zl_sess_lost(zl_sess_t *s, bool *was_open);

// ---------------------------------------------------------------- the playback queue

#define ZL_PLAYQ_STREAMS 8  // streams queued at once (the chime, a filler and the answer fit)

typedef struct {
    uint32_t id;
    size_t left;   // its samples still in the buffer
    bool ended;    // play_end arrived
    bool playing;  // primed: read out from now until it runs dry before its play_end
    bool started;  // at least one of its samples went out
    uint32_t dry;  // times it ran dry after it started, before its play_end
    size_t gap;    // silence samples played mid-stream: reads that came short while it led and had started
} zl_stream_t;

// A stream whose last sample is out, with its `dry` and `gap` for `played`.
typedef struct {
    uint32_t id;
    uint32_t dry;
    size_t gap;
} zl_played_t;

// A ring of mono samples holding streams back to back. Only the newest stream can be receiving
// (between play_start and play_end); 0x02 payload bytes go to it, anything else is discarded.
// A stream starts, and resumes after running dry, only once it holds `prime` samples or has
// ended: a stream that arrives late plays with a pause instead of a run of clicks. The pause
// mid-stream is counted on the stream (`dry`, `gap`); the wait before its first sample is not.
typedef struct {
    int16_t *buf;
    size_t cap, head, count, prime;
    zl_stream_t streams[ZL_PLAYQ_STREAMS];  // in buffer order, the oldest first
    size_t nstreams;
    bool receiving;
    bool has_odd;  // a frame ended inside a sample: its low byte waits for the next one
    uint8_t odd;
    zl_played_t done[ZL_PLAYQ_STREAMS];  // streams whose last sample is out, to report `played`
    size_t ndone;
    size_t overflow;  // samples discarded since the buffer was full
} zl_playq_t;

void zl_playq_init(zl_playq_t *q, int16_t *buf, size_t cap, size_t prime);
// play_start: false when no slot is free, queued and done streams share ZL_PLAYQ_STREAMS (the
// stream is then dropped). A stream still receiving is ended first.
bool zl_playq_start(zl_playq_t *q, uint32_t id);
// 0x02 payload bytes, little-endian int16, split anywhere.
void zl_playq_write(zl_playq_t *q, const uint8_t *data, size_t len);
// play_end of the receiving stream; any other id is ignored.
void zl_playq_end(zl_playq_t *q, uint32_t id);
// Copies up to max samples out in stream order, stopping at a stream not primed; a stream whose
// last sample went out after its play_end moves to `done`. The caller plays silence for the
// samples it did not get: when the head stream has started, they count as its `gap`.
size_t zl_playq_read(zl_playq_t *q, int16_t *out, size_t max);
// play_stop: empties the buffer; writes the ids of the dropped streams to `dropped`
// (ZL_PLAYQ_STREAMS slots) and returns their count. `done` is kept.
size_t zl_playq_flush(zl_playq_t *q, uint32_t *dropped);
// Moves the `done` streams into out (ZL_PLAYQ_STREAMS slots); returns their count.
size_t zl_playq_take_done(zl_playq_t *q, zl_played_t *out);

// ---------------------------------------------------------------- the model receiver

// Counts the bytes of one model: `model` announced `size`, the 0x03 payloads bring them in order.
// The bytes themselves stay with the caller.
typedef struct {
    bool active;  // from the start until the last byte is in
    size_t size, got;
} zl_modelrx_t;

// A `model` of `size` bytes (> 0); one still receiving is dropped.
void zl_modelrx_start(zl_modelrx_t *r, size_t size);
// n payload bytes in: returns how many of them belong to the model, to be written at offset *at
// (the rest lies past `size` and is dropped); *done turns true with the last byte, and the
// receiver is inactive from then on. 0 while inactive.
size_t zl_modelrx_write(zl_modelrx_t *r, size_t n, size_t *at, bool *done);
