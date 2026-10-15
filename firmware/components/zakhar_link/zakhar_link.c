// The ESP side of zakhar_link (zakhar_link.h) over esp_websocket_client; the protocol logic is in
// the pure core (zl_core.h).
//
// Two tasks. The component task owns the client: it creates it, and when the link is lost
// destroys it and connects a new one after the backoff. It sends everything the device starts:
// the wake and the microphone frames, the capture and its audio_end, wake_peak once a second,
// `played` of finished streams and `volume`. The client's own task runs the event handler: the
// hello on connect and every received frame in arrival order, so a play_start, its 0x02 frames
// and its play_end reach the playback queue in the order sent. One mutex guards the shared state.
// Nothing sends or calls back while holding it: the client's task holds the client lock around
// the handler, and a send from the component task takes that lock.
#include "zakhar_link.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "zl_core.h"

static const char *TAG = "zakhar_link";

#define PASS_MS 20             // the component task's loop: microphone frames and timers
#define MAX_FRAMES_PER_PASS 8  // at most 8 microphone frames (256 ms) per pass
#define SEND_TIMEOUT_MS 2000   // a send that cannot go out in this long fails and the link drops
#define SILENCE_MS 30000       // nothing heard for this long drops the link
#define RX_CAP 4096            // a text frame larger than this is skipped; fits a long reply
#define PEAK_MS 1000           // wake_peak period
// Core 0, priority 3, for both tasks: next to the wake-word task, away from the UI on core 1.
#define TASK_STACK (6 * 1024)
#define TASK_PRIO 3
#define TASK_CORE 0

// Component task notification bits.
#define EV_LINK_DOWN (1u << 0)  // the client's task ended
#define EV_WAKE (1u << 1)
#define EV_CAPTURE (1u << 2)
#define EV_PLAYED (1u << 3)
#define EV_VOLUME (1u << 4)

static zl_config_t s_cfg;
static zl_callbacks_t s_cb;
static TaskHandle_t s_task;
static SemaphoreHandle_t s_lock;
static _Atomic(const char *) s_wake_word;
static _Atomic uint32_t s_wake_pos;  // the microphone position at the last detection

// Under s_lock. s_ws is written by the component task only, which reads it without the lock.
static esp_websocket_client_handle_t s_ws;
static bool s_online;          // the hello is out and the link not lost yet
static TickType_t s_heard;     // the last frame received
static uint32_t s_backoff = 1; // seconds before the next connect; 1 again after a `config`
static zl_sess_t s_sess;
static uint32_t s_capture_ms;  // the last `capture`
static TickType_t s_capture_since, s_capture_ticks;
static bool s_peak_on;
static int s_volume;
static zl_playq_t s_playq;

// The client's task only.
static char s_rx[RX_CAP];
static bool s_rx_skip;   // the text frame being received does not fit s_rx
static uint8_t s_kind;   // the kind byte of the binary frame being received

// The component task only: the next microphone frame to send (set by wake() and capture_start(),
// advanced by mic_poll(), all three on that task), and a microphone frame, the kind byte at byte 1
// so the samples from byte 2 are aligned.
static uint32_t s_mic_pos;
static int16_t s_tx[1 + ZL_FRAME_FRAMES * 2];

static void lock(void)
{
    xSemaphoreTake(s_lock, portMAX_DELAY);
}

static void unlock(void)
{
    xSemaphoreGive(s_lock);
}

static void notify(uint32_t bits)
{
    xTaskNotify(s_task, bits, eSetBits);
}

// Sends one JSON text and releases it.
static void send_json(esp_websocket_client_handle_t ws, char *json)
{
    if (json == NULL) {
        ESP_LOGE(TAG, "message not built: out of memory");
        return;
    }
    if (esp_websocket_client_send_text(ws, json, (int)strlen(json), pdMS_TO_TICKS(SEND_TIMEOUT_MS)) < 0) {
        ESP_LOGW(TAG, "not sent: %s", json);
    }
    zl_free(json);
}

static void send_dropped(esp_websocket_client_handle_t ws, const uint32_t *ids, size_t n)
{
    for (size_t i = 0; i < n; i++) send_json(ws, zl_encode_played(ids[i], true));
}

// ---------------------------------------------------------------- the client's task

static void on_connected(esp_websocket_client_handle_t ws)
{
    lock();
    const int volume = s_volume;
    unlock();
    ESP_LOGI(TAG, "connected to %s", s_cfg.url);
    send_json(ws, zl_encode_hello(&s_cfg, volume));
    lock();
    s_online = true;
    unlock();
    if (s_cb.on_connection != NULL) s_cb.on_connection(true);
}

static void on_text(esp_websocket_client_handle_t ws, const char *json, size_t len)
{
    zl_msg_t m;
    if (!zl_decode(json, len, &m)) {
        ESP_LOGD(TAG, "ignored: %.*s", (int)len, json);
        return;
    }
    const bool speaker = s_cfg.play_rate > 0;
    uint32_t dropped[ZL_PLAYQ_STREAMS];
    size_t ndropped = 0, overflow = 0;
    bool done = false;
    lock();
    const bool current = zl_sess_accepts(&s_sess, &m);
    if (current) {
        switch (m.type) {
        case ZL_MSG_CONFIG:
            s_backoff = 1;
            break;
        case ZL_MSG_STATE:
            zl_sess_state(&s_sess, m.state);
            break;
        case ZL_MSG_PLAY_START:
            if (speaker && !zl_playq_start(&s_playq, m.id)) dropped[ndropped++] = m.id;
            break;
        case ZL_MSG_PLAY_END:
            if (!speaker) break;
            zl_playq_end(&s_playq, m.id);
            done = s_playq.ndone > 0;
            overflow = s_playq.overflow;
            s_playq.overflow = 0;
            break;
        case ZL_MSG_PLAY_STOP:
            if (speaker) ndropped = zl_playq_flush(&s_playq, dropped);
            break;
        case ZL_MSG_WAKE_PEAK_STREAM:
            s_peak_on = m.on;
            break;
        case ZL_MSG_CAPTURE:
            s_capture_ms = m.capture_ms;
            break;
        default:
            break;
        }
    }
    unlock();
    if (!current) {
        ESP_LOGD(TAG, "dropped, session %u is not current: %.*s", (unsigned)m.session, (int)len, json);
        zl_msg_free(&m);
        return;
    }
    switch (m.type) {
    case ZL_MSG_CONFIG:
        ESP_LOGI(TAG, "config: wake %u, vad %u", m.wake, m.vad);
        if (s_cb.on_config != NULL) s_cb.on_config(m.wake, m.vad);
        break;
    case ZL_MSG_STATE:
        ESP_LOGI(TAG, "session %u: %s%s%s", (unsigned)m.session, zl_state_name(m.state), m.text != NULL ? ": " : "",
                 m.text != NULL ? m.text : "");
        if (s_cb.on_state != NULL) s_cb.on_state(m.state, m.text);
        break;
    case ZL_MSG_TRANSCRIPT:
        ESP_LOGI(TAG, "transcript: %s", m.text);
        if (s_cb.on_transcript != NULL) s_cb.on_transcript(m.text);
        break;
    case ZL_MSG_REPLY:
        ESP_LOGI(TAG, "%s: %s", m.has_session ? "reply" : "announcement", m.text);
        if (s_cb.on_reply != NULL) s_cb.on_reply(m.text, !m.has_session);
        break;
    case ZL_MSG_PLAY_END:
        if (overflow > 0) ESP_LOGW(TAG, "stream %u: %u samples did not fit the buffer", (unsigned)m.id, (unsigned)overflow);
        if (done) notify(EV_PLAYED);
        break;
    case ZL_MSG_SET_VOLUME:
        if (s_cb.on_set_volume != NULL) s_cb.on_set_volume(m.value);
        break;
    case ZL_MSG_WAKE_PEAK_STREAM:
        ESP_LOGI(TAG, "wake_peak stream %s", m.on ? "on" : "off");
        if (m.on) (void)s_cfg.wake_peak_take();  // the peak counts from now
        break;
    case ZL_MSG_CAPTURE:
        notify(EV_CAPTURE);
        break;
    default:
        break;
    }
    send_dropped(ws, dropped, ndropped);
    zl_msg_free(&m);
}

static void on_data(const esp_websocket_event_data_t *d)
{
    if (d->op_code == WS_TRANSPORT_OPCODES_TEXT) {
        if (d->payload_offset == 0) s_rx_skip = d->payload_len > RX_CAP;
        if (!s_rx_skip) memcpy(s_rx + d->payload_offset, d->data_ptr, d->data_len);
        if (d->payload_offset + d->data_len < d->payload_len) return;  // more of the frame follows
        if (s_rx_skip) {
            ESP_LOGW(TAG, "text frame of %d B skipped", d->payload_len);
            return;
        }
        on_text(d->client, s_rx, d->payload_len);
    } else if (d->op_code == WS_TRANSPORT_OPCODES_BINARY) {
        const uint8_t *p = (const uint8_t *)d->data_ptr;
        size_t n = d->data_len;
        if (d->payload_offset == 0 && n > 0) {
            s_kind = p[0];
            p++;
            n--;
        }
        if (s_kind != ZL_KIND_PLAY || s_cfg.play_rate <= 0) return;
        lock();
        zl_playq_write(&s_playq, p, n);
        unlock();
    }
}

static void on_ws_event(void *arg, esp_event_base_t base, int32_t id, void *event_data)
{
    (void)arg;
    (void)base;
    const esp_websocket_event_data_t *d = event_data;
    lock();
    const bool current = d->client == s_ws;  // events of a client being destroyed are not
    if (current) s_heard = xTaskGetTickCount();
    unlock();
    if (!current) return;
    switch (id) {
    case WEBSOCKET_EVENT_CONNECTED:
        on_connected(d->client);
        break;
    case WEBSOCKET_EVENT_DATA:
        on_data(d);
        break;
    case WEBSOCKET_EVENT_CLOSED:
        ESP_LOGW(TAG, "closed by the server, code %d", d->close_status_code);
        break;
    case WEBSOCKET_EVENT_FINISH:  // every way the client stops ends here
        notify(EV_LINK_DOWN);
        break;
    default:
        break;
    }
}

// ---------------------------------------------------------------- the component task

// Seconds to the next connect, and the backoff after it.
static uint32_t backoff_take(void)
{
    lock();
    const uint32_t s = s_backoff;
    s_backoff = zl_backoff_next(s);
    unlock();
    return s;
}

// The link is gone (or silent): the client goes, the session, the capture and the playback end.
// Returns the delay to the next connect.
static uint32_t link_lost(void)
{
    esp_websocket_client_handle_t ws = s_ws;
    lock();
    s_ws = NULL;
    const bool was_online = s_online;
    s_online = false;
    unlock();
    esp_websocket_client_destroy(ws);
    bool was_open, was_capture;
    uint32_t ids[ZL_PLAYQ_STREAMS];
    lock();
    zl_sess_lost(&s_sess, &was_open, &was_capture);
    s_peak_on = false;
    (void)zl_playq_flush(&s_playq, ids);
    (void)zl_playq_take_done(&s_playq, ids);
    unlock();
    const uint32_t delay = backoff_take();
    ESP_LOGW(TAG, "link lost, reconnecting in %u s", (unsigned)delay);
    if (was_open && s_cb.on_state != NULL) s_cb.on_state(ZL_STATE_ERROR, NULL);
    if (was_capture && s_cb.on_capture != NULL) s_cb.on_capture(false);
    if (was_online && s_cb.on_connection != NULL) s_cb.on_connection(false);
    return delay;
}

// Starts a client; false when it could not even be created (the caller waits the backoff). A
// client that fails later ends in WEBSOCKET_EVENT_FINISH.
static bool link_connect(void)
{
    const esp_websocket_client_config_t wc = {
        .uri = s_cfg.url,
        .disable_auto_reconnect = true,   // the backoff is ours
        .disable_pingpong_discon = true,  // the silence rule is ours
        .buffer_size = 1 + ZL_FRAME_FRAMES * s_cfg.mic_channels * (int)sizeof(int16_t),  // a microphone frame in one piece
        .task_stack = TASK_STACK,
        .task_prio = TASK_PRIO,
        .task_core_id_set = true,
        .task_core_id = TASK_CORE,
    };
    esp_websocket_client_handle_t ws = esp_websocket_client_init(&wc);
    if (ws == NULL) {
        ESP_LOGE(TAG, "client init failed");
        return false;
    }
    if (esp_websocket_register_events(ws, WEBSOCKET_EVENT_ANY, on_ws_event, NULL) != ESP_OK) {
        ESP_LOGE(TAG, "event register failed");
        esp_websocket_client_destroy(ws);
        return false;
    }
    lock();
    s_ws = ws;
    unlock();
    if (esp_websocket_client_start(ws) != ESP_OK) notify(EV_LINK_DOWN);
    return true;
}

static void wake(void)
{
    const char *word = atomic_load(&s_wake_word);
    lock();
    const bool online = s_online;
    unlock();
    if (!online) {
        ESP_LOGI(TAG, "wake word ignored: no server");
        return;
    }
    if (s_cb.can_start != NULL && !s_cb.can_start()) {
        ESP_LOGI(TAG, "wake word ignored: the start gate is closed");
        return;
    }
    // Counted back from the detection, not from now: this task may reach the event late.
    const uint32_t pos = atomic_load(&s_wake_pos);
    const uint32_t preroll = (uint32_t)s_cfg.preroll_ms * (ZL_MIC_RATE / 1000);
    uint32_t dropped[ZL_PLAYQ_STREAMS];
    size_t ndropped = 0;
    lock();
    const bool started = zl_sess_wake(&s_sess);
    const uint32_t session = s_sess.session;
    if (started) {
        s_mic_pos = pos - (pos < preroll ? pos : preroll);
        ndropped = zl_playq_flush(&s_playq, dropped);
    }
    unlock();
    if (!started) {
        ESP_LOGI(TAG, "wake word ignored: a capture is running");
        return;
    }
    ESP_LOGI(TAG, "wake word \"%s\": session %u", word, (unsigned)session);
    send_json(s_ws, zl_encode_wake(session, word));
    send_dropped(s_ws, dropped, ndropped);
}

// A `capture` from the server: from now, no pre-roll; refused while a session is open.
static void capture_start(TickType_t now)
{
    const uint32_t pos = s_cfg.mic_position();
    lock();
    const bool online = s_online;
    const bool started = online && zl_sess_capture(&s_sess);
    const uint32_t ms = s_capture_ms;
    if (started) {
        s_mic_pos = pos;
        s_capture_since = now;
        s_capture_ticks = pdMS_TO_TICKS(ms);
    }
    unlock();
    if (!online) return;
    if (!started) {
        ESP_LOGI(TAG, "capture refused: a session is running");
        send_json(s_ws, zl_encode_audio_end("busy"));
        return;
    }
    ESP_LOGI(TAG, "capture: %u ms", (unsigned)ms);
    if (s_cb.on_capture != NULL) s_cb.on_capture(true);
}

static void capture_poll(TickType_t now)
{
    lock();
    const bool over = s_sess.capture && now - s_capture_since >= s_capture_ticks;
    if (over) s_sess.capture = false;
    unlock();
    if (!over) return;
    ESP_LOGI(TAG, "capture complete");
    send_json(s_ws, zl_encode_audio_end("capture"));
    if (s_cb.on_capture != NULL) s_cb.on_capture(false);
}

// The session's or the capture's microphone audio: whole frames only, at most
// MAX_FRAMES_PER_PASS per pass, so the pre-roll goes out at that pace and then the live audio.
static void mic_poll(void)
{
    lock();
    const bool on = s_online && (s_sess.mic || s_sess.capture);
    unlock();
    if (!on) return;
    const size_t samples = ZL_FRAME_FRAMES * (size_t)s_cfg.mic_channels;
    for (int i = 0; i < MAX_FRAMES_PER_PASS; i++) {
        if (s_cfg.mic_position() - s_mic_pos < ZL_FRAME_FRAMES) break;
        if (s_cfg.mic_read(&s_mic_pos, s_tx + 1, ZL_FRAME_FRAMES) != ZL_FRAME_FRAMES) break;
        const int len = 1 + (int)(samples * sizeof(int16_t));
        if (esp_websocket_client_send_bin(s_ws, (const char *)s_tx + 1, len, pdMS_TO_TICKS(SEND_TIMEOUT_MS)) < 0) break;
    }
}

static void peak_poll(void)
{
    lock();
    const bool on = s_online && s_peak_on;
    unlock();
    if (on) send_json(s_ws, zl_encode_wake_peak(zl_peak_percent(s_cfg.wake_peak_take())));
}

static void send_played(void)
{
    uint32_t ids[ZL_PLAYQ_STREAMS];
    lock();
    const size_t n = zl_playq_take_done(&s_playq, ids);
    const bool online = s_online;
    unlock();
    for (size_t i = 0; online && i < n; i++) send_json(s_ws, zl_encode_played(ids[i], false));
}

static void send_volume(void)
{
    lock();
    const bool online = s_online;
    const int volume = s_volume;
    unlock();
    if (online) send_json(s_ws, zl_encode_volume(volume));
}

static void link_task(void *arg)
{
    (void)arg;
    TickType_t connect_at = xTaskGetTickCount(), peak_at = connect_at;
    for (;;) {
        uint32_t ev = 0;
        xTaskNotifyWait(0, UINT32_MAX, &ev, pdMS_TO_TICKS(PASS_MS));
        const TickType_t now = xTaskGetTickCount();
        lock();
        const bool silent = s_online && now - s_heard >= pdMS_TO_TICKS(SILENCE_MS);
        unlock();
        if (silent) ESP_LOGW(TAG, "nothing heard for %d s", SILENCE_MS / 1000);
        if (s_ws != NULL && ((ev & EV_LINK_DOWN) || silent)) connect_at = now + pdMS_TO_TICKS(link_lost() * 1000);
        if (s_ws == NULL && (int32_t)(now - connect_at) >= 0 && !link_connect()) {
            connect_at = now + pdMS_TO_TICKS(backoff_take() * 1000);
        }
        if (ev & EV_CAPTURE) capture_start(now);
        if (ev & EV_WAKE) wake();
        if (ev & EV_PLAYED) send_played();
        if (ev & EV_VOLUME) send_volume();
        mic_poll();
        capture_poll(now);
        if (now - peak_at >= pdMS_TO_TICKS(PEAK_MS)) {
            peak_at = now;
            peak_poll();
        }
    }
}

// ---------------------------------------------------------------- API

void zl_start(const zl_config_t *cfg, const zl_callbacks_t *cb)
{
    if (cfg->mic_channels < 1 || cfg->mic_channels > 2) {
        ESP_LOGE(TAG, "mic_channels %d: 1 or 2", cfg->mic_channels);
        return;
    }
    s_cfg = *cfg;
    s_cfg.url = strdup(cfg->url);
    s_cfg.id = strdup(cfg->id);
    s_cfg.fw = strdup(cfg->fw);
    s_cfg.model = strdup(cfg->model);
    if (cb != NULL) s_cb = *cb;
    s_volume = cfg->volume;
    ((uint8_t *)s_tx)[1] = ZL_KIND_MIC;
    if (cfg->play_rate > 0) {
        const size_t cap = (size_t)cfg->play_rate * (size_t)cfg->play_buffer_ms / 1000;
        int16_t *buf = heap_caps_malloc(cap * sizeof *buf, MALLOC_CAP_SPIRAM);
        if (buf == NULL) {
            ESP_LOGE(TAG, "playback buffer of %u samples: out of PSRAM", (unsigned)cap);
            return;
        }
        zl_playq_init(&s_playq, buf, cap);
    }
    s_lock = xSemaphoreCreateMutex();
    if (s_lock == NULL ||
        xTaskCreatePinnedToCore(link_task, "zakhar_link", TASK_STACK, NULL, TASK_PRIO, &s_task, TASK_CORE) != pdPASS) {
        ESP_LOGE(TAG, "task create failed");
    }
}

void zl_wake(const char *word)
{
    atomic_store(&s_wake_word, word);
    if (s_task == NULL) return;
    atomic_store(&s_wake_pos, s_cfg.mic_position());
    notify(EV_WAKE);
}

void zl_report_volume(int percent)
{
    lock();
    s_volume = percent;
    unlock();
    notify(EV_VOLUME);
}

size_t zl_play_read(int16_t *out, size_t max)
{
    lock();
    const size_t n = zl_playq_read(&s_playq, out, max);
    const bool done = s_playq.ndone > 0;
    unlock();
    if (done) notify(EV_PLAYED);
    return n;
}
