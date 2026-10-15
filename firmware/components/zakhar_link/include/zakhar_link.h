// zakhar_link -- the device side of the zakhar WebSocket device protocol
// (zakhar-voice-assistant docs/device-protocol.md): one connection to the server, the voice
// session over a microphone tap, the settings and diagnostics messages and, on a device with a
// speaker, the playback buffer. Shared by the espanel wall panel and the Voice PE firmware.
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

typedef struct {
    const char *url;    // ws://<server>:8200/device
    const char *id;     // hello.id: the device name in the server config
    const char *fw;     // hello.fw
    const char *model;  // hello.model
    bool screen;        // hello.caps.screen
    int mic_channels;   // 1 or 2, interleaved in the tap
    int preroll_ms;     // audio before the detection sent after a wake
    // The microphone tap: 16 kHz int16, mic_channels interleaved, positions counted in frames
    // (one sample per channel). mic_position() is the count of frames written so far;
    // mic_read() copies up to max_frames frames from *pos and advances *pos past them.
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
    // ignored meanwhile.
    void (*on_capture)(bool active);
    // Asked on every wake word; false ignores it.
    bool (*can_start)(void);
} zl_callbacks_t;

// Copies cfg (its strings too) and cb, allocates the playback buffer and starts the component
// task, which connects to cfg->url and reconnects with a backoff. Call once, with the network up.
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

#ifdef __cplusplus
}
#endif
