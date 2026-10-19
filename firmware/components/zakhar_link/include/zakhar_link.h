// zakhar_link -- the device side of the zakhar WebSocket device protocol
// (zakhar-voice-assistant docs/device-protocol.md): one connection to the server, the voice
// session and the captures over a microphone tap, the settings and diagnostics messages and, on a
// device with a speaker, the playback buffer. Shared by the espanel wall panel and the Voice PE
// firmware.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The `state` values of a voice session.
typedef enum {
    ZL_STATE_LISTENING,
    ZL_STATE_TRANSCRIBING,
    ZL_STATE_THINKING,
    ZL_STATE_SPEAKING,
    ZL_STATE_IDLE,
    ZL_STATE_ERROR,
} zl_state_t;

// The `kind` of a model the server sends.
typedef enum {
    ZL_MODEL_WAKE,
    ZL_MODEL_VAD,
    ZL_MODEL_STOP,
} zl_model_t;

typedef struct {
    const char *id;     // hello.id: the device name in the server config
    const char *fw;     // hello.fw
    bool screen;        // hello.caps.screen
    int mic_channels;   // 1 or 2, interleaved in the tap
    int preroll_ms;     // audio before the detection sent after a wake
    // The microphone tap: 16 kHz int16, mic_channels interleaved, positions counted in frames
    // (one sample per channel). mic_position() is the count of frames written so far, called from
    // the detection, the component and the client's task; mic_read() copies up to max_frames
    // frames from *pos and advances *pos past them.
    uint32_t (*mic_position)(void);
    size_t (*mic_read)(uint32_t *pos, int16_t *out, size_t max_frames);
    // The highest wake-word probability (0..255) since the previous call; resets it.
    uint8_t (*wake_peak_take)(void);
    int play_rate;       // playback sample rate, Hz; 0: no speaker
    int play_buffer_ms;  // the playback buffer (PSRAM), hello.play.buffer_ms
    int volume;          // speaker volume at start, percent
} zl_config_t;

// Every callback may be NULL. They run on the component task or on the WebSocket client's task,
// which answers the server's pings (the pong is due within 5 s), so they must not block; `text`
// is valid only during the call.
typedef struct {
    void (*on_connection)(bool connected);
    // A `state` of the current session. A session still open when the connection is lost ends
    // with ZL_STATE_ERROR and NULL text.
    void (*on_state)(zl_state_t state, const char *text);
    void (*on_transcript)(const char *text);
    // announcement: the reply came without a session (a reminder, an agent's speak).
    void (*on_reply)(const char *text, bool announcement);
    // The `config` cutoffs as uint8 = trunc(percent / 100 * 255), for the wake-word model and
    // the VAD gate.
    void (*on_config)(uint8_t wake, uint8_t vad);
    // Apply and store the volume, then report it with zl_report_volume().
    void (*on_set_volume)(int percent);
    // A `capture` starts (true) or ends (false): show a recording indication; wake words are
    // ignored meanwhile. A lost link does not end it: it ends once all its audio is sent, or 2 s
    // after its last frame was recorded when it cannot be sent.
    void (*on_capture)(bool active);
    // Asked on a wake word that would start a session (online, detection on, no capture, no
    // session streaming its mic); false ignores it.
    bool (*can_start)(void);
    // A model the server sent (`model` and its 0x03 frames), all `size` bytes in. The callee owns
    // `data` (16-byte aligned, in PSRAM, released with heap_caps_free) and returns whether the
    // model loaded; the answer goes to the server as `model_loaded`. Runs on the client's task
    // and, unlike the others, may take a few milliseconds there.
    bool (*on_model)(zl_model_t kind, uint8_t *data, size_t size);
} zl_callbacks_t;

// Copies cfg (its strings too) and cb, allocates the playback buffer, starts mDNS and the component
// task. Before every connect the task finds the server by mDNS (`_zakhar._tcp`, the first answer
// with an IPv4 address) and connects to ws://<address>:<port>/device; while none answers, and
// after a lost link, it keeps trying with the backoff. Call once, with the network up.
void zl_start(const zl_config_t *cfg, const zl_callbacks_t *cb);

// A wake word detection. Only stores `word` (a string that outlives the call) and the microphone
// position (mic_position(), so it must be callable from the detection task; the pre-roll counts
// back from it) and notifies the component task, so it is safe in the detection callback with
// little stack.
void zl_wake(const char *word);

// The volume changed on the device (the dial, or after on_set_volume): sent as `volume` and
// carried by the next hello. Safe from any task after zl_start().
void zl_report_volume(int percent);

// Speakers: the next playback samples (mono, play_rate) for the I2S writer task, after
// zl_start(). Returns the count copied, fewer than max (0 included) when the buffer runs dry.
size_t zl_play_read(int16_t *out, size_t max);

// Whether a stream is in the playback buffer (from its play_start until its last sample is out);
// false on a device without a speaker and before zl_start(). Takes the component's lock briefly.
bool zl_playing(void);

// The stop word: only notifies the component task, which flushes the playback and sends `stop`
// (with the open session's number), so it is safe in the detection callback. After zl_start().
void zl_stop(void);

#ifdef __cplusplus
}
#endif
