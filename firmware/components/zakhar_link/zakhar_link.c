// The ESP side of zakhar_link (zakhar_link.h) over esp_websocket_client; the protocol logic is in
// the pure core (zl_core.h).
//
// Two tasks. The component task owns the client: it creates it, and when the link is lost
// destroys it and connects a new one after the backoff. It sends everything the device starts:
// the wake, `stop`, the session's and the capture's mic frames, audio_end, wake_peak once a second,
// `played` of finished streams, `volume`, `ambient` and, while the server asks for it, the device log in
// `log` messages. It also owns the microphone position, so a `capture` and a `capture_resume`
// are handed to it; only a follow-up `listening` sets the position on the client's task, under
// the mutex, together with turning the mic on. A capture outlives a lost link: the next hello
// names it, and a `capture_resume` sends it on from the frame the server asks for. The client's
// own task runs the event handler: the hello on connect and every received frame in arrival
// order, so a play_start, its 0x02 frames and its play_end reach the playback queue in the order
// sent; a model is assembled from its 0x03 frames and handed to on_model there too. One mutex
// guards the shared state.
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
#include "mdns.h"
#include "zl_core.h"

static const char *TAG = "zakhar_link";

#define PASS_MS 20             // the component task's loop: microphone frames and timers
#define MAX_FRAMES_PER_PASS 8  // at most 8 microphone frames (256 ms) per pass
// A send that cannot go out in this long fails and the link drops. Long enough to wait out the
// ~1.6 s retransmission timeout after a segment lost on Wi-Fi, short enough that the microphone
// frames waiting meanwhile still fit the device's tap (3.1 s on the Voice PE and the panel; about
// 1 s while a session's 2 s pre-roll is still going out): a longer wait would skip the oldest of
// them, splicing a session's audio and leaving a hole in a capture's, instead of dropping the link,
// which a capture outlives with its frames still in the tap to resume from.
#define SEND_TIMEOUT_MS 3000
#define SILENCE_MS 30000       // nothing heard for this long drops the link
#define RX_CAP 4096            // a text frame larger than this is skipped; fits a long reply
#define PEAK_MS 1000           // wake_peak period
// Audio a stream holds before it plays, and again after it runs dry. A TTS stream arrives in
// bursts with pauses between them; played with no reserve, each pause longer than what is queued
// is a gap of silence mid-word, heard as a click. The wait stays inside the server's 500 ms pacing
// margin (docs/device-protocol.md), so the buffer does not overflow.
#define PLAY_PRIME_MS 300
#define LOG_CHUNK 2048  // the most log text one `log` message carries; one goes out per pass
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
#define EV_CAPTURE_RESUME (1u << 5)
#define EV_STOP (1u << 6)
#define EV_AMBIENT (1u << 7)

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
static uint32_t s_capture_ms, s_capture_new;  // the last `capture`: its length and id
static uint32_t s_resume_id, s_resume_from;   // the last `capture_resume`
// The held capture (s_sess.capture): its id, and whether its frames go to the server on this link
// (from its `capture` or `capture_resume` until the link is lost).
static uint32_t s_capture_id;
static bool s_capture_sending;
static bool s_peak_on;
static bool s_log_on;  // the server asked for the log on this link
static int s_volume;
static double s_ambient;  // the last zl_report_ambient()
static zl_playq_t s_playq;

// The client's task only.
static char s_rx[RX_CAP];
static bool s_rx_skip;   // the text frame being received does not fit s_rx
static uint8_t s_kind;   // the kind byte of the binary frame being received
// The model being received, its bytes counted by s_modelrx. s_model is NULL without on_model or
// when its allocation failed; the bytes are then counted and dropped.
static zl_modelrx_t s_modelrx;
static zl_model_t s_model_kind;
static char *s_model_name;
static uint8_t *s_model;

// The next microphone frame to send: set by wake(), capture_start() and capture_resume() and
// advanced by mic_poll(), all on the component task; set by on_text() on the client's task, under
// s_lock, when a follow-up `listening` turns the mic on again: no mic_poll() reads it then (the mic
// is off from before the `played` this task sent), and the next one reads it after seeing the mic
// on under s_lock.
static uint32_t s_mic_pos;
// The component task only: a microphone frame with its samples always from byte 8, aligned: a
// session's frame starts at byte 7 with its kind byte, a capture's at byte 3 with its kind byte and
// the offset at bytes 4..7.
static int16_t s_tx[4 + ZL_FRAME_FRAMES * 2];
// The component task only: the held capture's first frame and length (set by capture_start()).
static uint32_t s_capture_start, s_capture_frames;
// The server's URL found by mDNS: written by the component task before it creates a client, read
// while that client lives.
static char s_url[sizeof "ws://255.255.255.255:65535/device"];
// The component task only: log text for one `log` message (PSRAM, only with log_read).
static char *s_log;

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
    for (size_t i = 0; i < n; i++) send_json(ws, zl_encode_played(ids[i], true, 0, 0));
}

// ---------------------------------------------------------------- the client's task

// Drops the model being received, if any.
static void model_discard(void)
{
    heap_caps_free(s_model);
    free(s_model_name);
    s_model = NULL;
    s_model_name = NULL;
    s_modelrx.active = false;
}

// A `model`: its bytes follow in 0x03 frames; one still being received is dropped.
static void model_start(const zl_msg_t *m)
{
    model_discard();
    zl_modelrx_start(&s_modelrx, m->size);
    s_model_kind = m->model;
    s_model_name = strdup(m->name);
    if (s_cb.on_model == NULL) return;
    s_model = heap_caps_aligned_alloc(16, m->size, MALLOC_CAP_SPIRAM);
    if (s_model == NULL) ESP_LOGE(TAG, "%s model %s of %u B: out of PSRAM", zl_model_name(m->model), m->name, (unsigned)m->size);
}

// 0x03 payload bytes; bytes past the announced size are dropped. The complete model goes to
// on_model (s_model exists only with it), and its `model_loaded` to the server.
static void model_write(esp_websocket_client_handle_t ws, const uint8_t *p, size_t n)
{
    size_t at;
    bool done;
    const size_t take = zl_modelrx_write(&s_modelrx, n, &at, &done);
    if (s_model != NULL) memcpy(s_model + at, p, take);
    if (!done) return;
    const size_t size = s_modelrx.size;
    uint8_t *data = s_model;
    s_model = NULL;
    const bool ok = data != NULL && s_cb.on_model(s_model_kind, data, size);
    if (ok) {
        ESP_LOGI(TAG, "%s model %s loaded (%u B)", zl_model_name(s_model_kind), s_model_name, (unsigned)size);
    } else {
        ESP_LOGW(TAG, "%s model %s (%u B) not loaded", zl_model_name(s_model_kind), s_model_name, (unsigned)size);
    }
    send_json(ws, zl_encode_model_loaded(s_model_kind, s_model_name, ok));
    model_discard();
}

static void on_connected(esp_websocket_client_handle_t ws)
{
    model_discard();
    lock();
    const int volume = s_volume;
    const uint32_t capture = s_sess.capture ? s_capture_id : 0;
    unlock();
    ESP_LOGI(TAG, "connected to %s", s_url);
    send_json(ws, zl_encode_hello(&s_cfg, volume, capture));
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
    // Where a follow-up's audio starts: read outside the lock, like every s_cfg callback.
    const uint32_t mic_now = s_cfg.mic_position();
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
            if (zl_sess_state(&s_sess, m.state)) s_mic_pos = mic_now;
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
            s_capture_new = m.id;
            break;
        case ZL_MSG_CAPTURE_RESUME:
            s_resume_id = m.id;
            s_resume_from = m.from;
            break;
        case ZL_MSG_DETECT:
            zl_sess_detect(&s_sess, m.on);
            break;
        case ZL_MSG_LOG_STREAM:
            s_log_on = m.on;
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
        ESP_LOGI(TAG, "config: wake %u, vad %u, echo_cancel %d mode %d filter %d, auto_volume %d", m.wake, m.vad,
                 m.echo_cancel, m.echo_mode, m.echo_filter, m.auto_volume);
        if (s_cb.on_config != NULL) s_cb.on_config(m.wake, m.vad);
        if ((m.echo_cancel >= 0 || m.echo_mode >= 0 || m.echo_filter >= 0) && s_cb.on_echo_cancel != NULL) {
            s_cb.on_echo_cancel(m.echo_cancel, m.echo_mode, m.echo_filter);
        }
        if (m.auto_volume >= 0 && s_cb.on_auto_volume != NULL) s_cb.on_auto_volume(m.auto_volume == 1);
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
    case ZL_MSG_VOLUME_BOOST:
        if (s_cb.on_volume_boost != NULL) s_cb.on_volume_boost(m.value);
        break;
    case ZL_MSG_WAKE_PEAK_STREAM:
        ESP_LOGI(TAG, "wake_peak stream %s", m.on ? "on" : "off");
        if (m.on) (void)s_cfg.wake_peak_take();  // the peak counts from now
        break;
    case ZL_MSG_CAPTURE:
        notify(EV_CAPTURE);
        break;
    case ZL_MSG_CAPTURE_RESUME:
        notify(EV_CAPTURE_RESUME);
        break;
    case ZL_MSG_DETECT:
        ESP_LOGI(TAG, "wake word detection %s", m.on ? "on" : "off");
        break;
    case ZL_MSG_LOG_STREAM:
        // The callback first, so this line itself is queued for the server.
        if (s_cb.on_log_stream != NULL) s_cb.on_log_stream(m.on);
        ESP_LOGI(TAG, "log stream %s", m.on ? "on" : "off");
        break;
    case ZL_MSG_MODEL:
        model_start(&m);
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
        if (s_kind == ZL_KIND_MODEL) {
            model_write(d->client, p, n);
            return;
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

// The link is gone (or silent): the client goes, the session and the playback end. A held capture
// stays held and stops sending until a `capture_resume`. The log stops draining but on_log_stream
// is not called: the lines around the drop stay queued for the next link's `log_stream`.
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
    bool was_open;
    uint32_t ids[ZL_PLAYQ_STREAMS];
    zl_played_t done[ZL_PLAYQ_STREAMS];
    lock();
    zl_sess_lost(&s_sess, &was_open);
    s_capture_sending = false;
    s_peak_on = false;
    s_log_on = false;
    (void)zl_playq_flush(&s_playq, ids);
    (void)zl_playq_take_done(&s_playq, done);
    unlock();
    const uint32_t delay = backoff_take();
    ESP_LOGW(TAG, "link lost, reconnecting in %u s", (unsigned)delay);
    if (was_open && s_cb.on_state != NULL) s_cb.on_state(ZL_STATE_ERROR, NULL);
    if (was_online && s_cb.on_connection != NULL) s_cb.on_connection(false);
    return delay;
}

// The first IPv4 address in an mDNS result's address list.
static bool first_ipv4(const mdns_ip_addr_t *a, esp_ip4_addr_t *ip)
{
    for (; a != NULL; a = a->next) {
        if (a->addr.type == ESP_IPADDR_TYPE_V4) {
            *ip = a->addr.u_addr.ip4;
            return true;
        }
    }
    return false;
}

// The address of <host>.local. Asked for with a multicast answer (QM), not mdns_query_a()'s
// unicast one (QU): a server in a Docker bridge network answers through an mDNS repeater on its
// host, which relays multicast only. The browse answer has just multicast this record, and
// python-zeroconf multicasts it again no sooner than 1 s later (RFC 6762 section 14): the answer
// comes 1.02-1.12 s after the question, so the wait is 2 s.
static bool address_find(const char *host, esp_ip4_addr_t *ip)
{
    mdns_result_t *results = NULL;
    if (mdns_query_generic(host, NULL, NULL, MDNS_TYPE_A, MDNS_QUERY_MULTICAST, 2000, 1, &results) != ESP_OK) {
        return false;
    }
    const bool found = results != NULL && first_ipv4(results->addr, ip);
    mdns_query_results_free(results);
    return found;
}

// Finds the server by mDNS into s_url: the first `_zakhar._tcp` answer with a host and port. The
// queries block this task for up to 3 + 2 s, but only while there is no client: offline, nothing
// is being streamed.
static bool server_find(void)
{
    mdns_result_t *results = NULL;
    // One result is enough: the query ends with the first answering server, not at the timeout.
    const esp_err_t err = mdns_query_ptr("_zakhar", "_tcp", 3000, 1, &results);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "mDNS query for the server failed: %s", esp_err_to_name(err));
        return false;
    }
    if (results == NULL) {
        ESP_LOGW(TAG, "no server answers _zakhar._tcp");
        return false;
    }
    bool found = false;
    for (const mdns_result_t *r = results; r != NULL && !found; r = r->next) {
        if (r->hostname == NULL) {  // no SRV record: no host and no port
            ESP_LOGW(TAG, "%s answers without its host and port", r->instance_name ? r->instance_name : "a server");
            continue;
        }
        // The parser keeps an A record only when the SRV record came before it in the packet, and
        // python-zeroconf orders those records at random: then the address is asked for by name.
        esp_ip4_addr_t ip = {0};
        if (!first_ipv4(r->addr, &ip) && !address_find(r->hostname, &ip)) {
            ESP_LOGW(TAG, "no address for %s.local", r->hostname);
            continue;
        }
        snprintf(s_url, sizeof s_url, "ws://" IPSTR ":%u/device", IP2STR(&ip), (unsigned)r->port);
        found = true;
    }
    mdns_query_results_free(results);
    return found;
}

// Starts a client; false when no server was found or the client could not even be created (the
// caller waits the backoff). A client that fails later ends in WEBSOCKET_EVENT_FINISH.
static bool link_connect(void)
{
    if (!server_find()) return false;
    const esp_websocket_client_config_t wc = {
        .uri = s_url,
        .disable_auto_reconnect = true,   // the backoff is ours
        .disable_pingpong_discon = true,  // the silence rule is ours
        // A capture's microphone frame, the larger kind, in one piece.
        .buffer_size = 5 + ZL_FRAME_FRAMES * s_cfg.mic_channels * (int)sizeof(int16_t),
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

static void wake_refused(zl_wake_t why)
{
    ESP_LOGI(TAG, "wake word ignored: %s",
             why == ZL_WAKE_DETECT_OFF ? "detection is off"
             : why == ZL_WAKE_CAPTURE  ? "a capture is running"
                                       : "the session still hears its phrase");
}

static void wake(void)
{
    const char *word = atomic_load(&s_wake_word);
    lock();
    const bool online = s_online;
    const zl_wake_t would = zl_sess_wake_check(&s_sess);
    unlock();
    if (!online) {
        ESP_LOGI(TAG, "wake word ignored: no server");
        return;
    }
    // Refused before the start gate is asked, so can_start sees only wakes that start a session.
    if (would != ZL_WAKE_START) {
        wake_refused(would);
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
    const zl_wake_t w = zl_sess_wake(&s_sess);
    const uint32_t session = s_sess.session;
    if (w == ZL_WAKE_START) {
        s_mic_pos = pos - (pos < preroll ? pos : preroll);
        ndropped = zl_playq_flush(&s_playq, dropped);
    }
    unlock();
    if (w != ZL_WAKE_START) {  // a `detect` off came in while can_start was asked
        wake_refused(w);
        return;
    }
    ESP_LOGI(TAG, "wake word \"%s\": session %u", word, (unsigned)session);
    send_json(s_ws, zl_encode_wake(session, word));
    send_dropped(s_ws, dropped, ndropped);
}

// The stop word: the playback is flushed, and the server stops the open session's run. The
// session itself ends with the server's `state`.
static void stop(void)
{
    uint32_t dropped[ZL_PLAYQ_STREAMS];
    lock();
    const size_t ndropped = zl_playq_flush(&s_playq, dropped);
    const bool online = s_online;
    const bool has_session = online && s_sess.open;
    const uint32_t session = s_sess.session;
    unlock();
    if (has_session) {
        ESP_LOGI(TAG, "stop word: session %u, %u streams dropped", (unsigned)session, (unsigned)ndropped);
    } else {
        ESP_LOGI(TAG, "stop word: no session, %u streams dropped", (unsigned)ndropped);
    }
    if (!online) return;
    send_json(s_ws, zl_encode_stop(has_session, session));
    send_dropped(s_ws, dropped, ndropped);
}

// A `capture` from the server: from now, no pre-roll, in place of one held already; refused while
// a session is open.
static void capture_start(void)
{
    const uint32_t pos = s_cfg.mic_position();
    lock();
    const bool online = s_online;
    const bool started = online && zl_sess_capture(&s_sess);
    const uint32_t ms = s_capture_ms, id = s_capture_new;
    if (started) {
        s_capture_id = id;
        s_capture_sending = true;
        s_mic_pos = pos;
        s_capture_start = pos;
        s_capture_frames = ms * (ZL_MIC_RATE / 1000);
    }
    unlock();
    if (!online) return;
    if (!started) {
        ESP_LOGI(TAG, "capture %u refused: a session is running", (unsigned)id);
        send_json(s_ws, zl_encode_audio_end("busy", id));
        return;
    }
    ESP_LOGI(TAG, "capture %u: %u ms", (unsigned)id, (unsigned)ms);
    if (s_cb.on_capture != NULL) s_cb.on_capture(true);
}

// A `capture_resume` for the held capture sends it on from the frame the server lacks (mic_read()
// skips to the oldest frame still in the tap when that one is gone); any other is ignored.
static void capture_resume(void)
{
    lock();
    const uint32_t id = s_resume_id, from = s_resume_from;
    const bool resumed = s_online && s_sess.capture && id == s_capture_id;
    if (resumed) {
        s_mic_pos = s_capture_start + (from < s_capture_frames ? from : s_capture_frames);
        s_capture_sending = true;
    }
    unlock();
    if (resumed) {
        ESP_LOGI(TAG, "capture %u resumed from frame %u", (unsigned)id, (unsigned)from);
    } else {
        ESP_LOGI(TAG, "capture_resume %u ignored", (unsigned)id);
    }
}

// The capture ends when its audio is sent, not by the clock: a pass that waited in a send must not
// cut off the frames still in the tap. One that is not being sent (its link lost, no
// `capture_resume`) is dropped without audio_end once its audio is leaving the tap.
static void capture_poll(void)
{
    const uint32_t pos = s_cfg.mic_position();
    lock();
    const uint32_t id = s_capture_id;
    const bool held = s_sess.capture, sending = s_capture_sending;
    const bool done = held && sending && zl_capture_done(s_mic_pos, s_capture_start, s_capture_frames);
    const bool expired = held && !sending && zl_capture_expired(pos, s_capture_start, s_capture_frames);
    if (done || expired) s_sess.capture = false;
    unlock();
    if (done) {
        ESP_LOGI(TAG, "capture %u complete", (unsigned)id);
        send_json(s_ws, zl_encode_audio_end("capture", id));
    } else if (expired) {
        ESP_LOGW(TAG, "capture %u dropped: not sent within %d s of its end", (unsigned)id,
                 ZL_CAPTURE_KEEP_FRAMES / ZL_MIC_RATE);
    }
    if ((done || expired) && s_cb.on_capture != NULL) s_cb.on_capture(false);
}

// The session's or the capture's microphone audio: whole frames only, at most
// MAX_FRAMES_PER_PASS per pass, so the pre-roll goes out at that pace and then the live audio.
// A capture's frame carries its offset, taken after the read: mic_read() skips forward past audio
// the tap no longer holds.
static void mic_poll(void)
{
    lock();
    const bool capture = s_sess.capture && s_capture_sending;
    const bool on = s_online && (s_sess.mic || capture);
    unlock();
    if (!on) return;
    const size_t samples = ZL_FRAME_FRAMES * (size_t)s_cfg.mic_channels;
    uint8_t *const frame = (uint8_t *)s_tx + (capture ? 3 : 7);
    frame[0] = capture ? ZL_KIND_CAPTURE : ZL_KIND_MIC;
    const int len = (capture ? 5 : 1) + (int)(samples * sizeof(int16_t));
    for (int i = 0; i < MAX_FRAMES_PER_PASS; i++) {
        if (capture && zl_capture_done(s_mic_pos, s_capture_start, s_capture_frames)) break;
        if (s_cfg.mic_position() - s_mic_pos < ZL_FRAME_FRAMES) break;
        if (s_cfg.mic_read(&s_mic_pos, s_tx + 4, ZL_FRAME_FRAMES) != ZL_FRAME_FRAMES) break;
        if (capture) zl_put_u32le(frame + 1, s_mic_pos - ZL_FRAME_FRAMES - s_capture_start);
        if (esp_websocket_client_send_bin(s_ws, (const char *)frame, len, pdMS_TO_TICKS(SEND_TIMEOUT_MS)) < 0) break;
    }
}

static void peak_poll(void)
{
    lock();
    const bool on = s_online && s_peak_on;
    unlock();
    if (on) send_json(s_ws, zl_encode_wake_peak(zl_peak_percent(s_cfg.wake_peak_take())));
}

// `played` of the finished streams; a stream's gap, counted in samples at play_rate, goes out in ms.
static void send_played(void)
{
    zl_played_t done[ZL_PLAYQ_STREAMS];
    lock();
    const size_t n = zl_playq_take_done(&s_playq, done);
    const bool online = s_online;
    unlock();
    for (size_t i = 0; online && i < n; i++) {
        const uint32_t gap_ms = (uint32_t)((uint64_t)done[i].gap * 1000 / (uint64_t)s_cfg.play_rate);
        send_json(s_ws, zl_encode_played(done[i].id, false, done[i].dry, gap_ms));
    }
}

static void send_volume(void)
{
    lock();
    const bool online = s_online;
    const int volume = s_volume;
    unlock();
    if (online) send_json(s_ws, zl_encode_volume(volume));
}

static void send_ambient(void)
{
    lock();
    const bool online = s_online;
    const double dbfs = s_ambient;
    unlock();
    if (online) send_json(s_ws, zl_encode_ambient(dbfs));
}

// The device log while the server asks for it: up to LOG_CHUNK bytes of whole lines a pass, in
// one `log`. Not through send_json: its failure line carries the message, which would feed the
// log into itself; a failure here logs only a fixed line.
static void log_poll(void)
{
    if (s_log == NULL) return;
    lock();
    const bool on = s_online && s_log_on;
    unlock();
    if (!on) return;
    const size_t n = s_cfg.log_read(s_log, LOG_CHUNK);
    if (n == 0) return;
    char *json = zl_encode_log(s_log, n);
    if (json == NULL ||
        esp_websocket_client_send_text(s_ws, json, (int)strlen(json), pdMS_TO_TICKS(SEND_TIMEOUT_MS)) < 0) {
        ESP_LOGW(TAG, "%u B of log not sent", (unsigned)n);
    }
    zl_free(json);
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
        const bool silent = s_online && zl_silent(now, s_heard, pdMS_TO_TICKS(SILENCE_MS));
        unlock();
        if (silent) ESP_LOGW(TAG, "nothing heard for %d s", SILENCE_MS / 1000);
        if (s_ws != NULL && ((ev & EV_LINK_DOWN) || silent)) connect_at = now + pdMS_TO_TICKS(link_lost() * 1000);
        if (s_ws == NULL && (int32_t)(now - connect_at) >= 0 && !link_connect()) {
            connect_at = now + pdMS_TO_TICKS(backoff_take() * 1000);
        }
        if (ev & EV_CAPTURE) capture_start();
        if (ev & EV_CAPTURE_RESUME) capture_resume();
        // Before a wake of the same pass: its `stop` names the session the wake replaces.
        if (ev & EV_STOP) stop();
        if (ev & EV_WAKE) wake();
        if (ev & EV_PLAYED) send_played();
        if (ev & EV_VOLUME) send_volume();
        if (ev & EV_AMBIENT) send_ambient();
        mic_poll();
        capture_poll();
        if (now - peak_at >= pdMS_TO_TICKS(PEAK_MS)) {
            peak_at = now;
            peak_poll();
        }
        log_poll();
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
    s_cfg.id = strdup(cfg->id);
    s_cfg.fw = strdup(cfg->fw);
    if (cb != NULL) s_cb = *cb;
    s_volume = cfg->volume;
    if (cfg->play_rate > 0) {
        const size_t cap = (size_t)cfg->play_rate * (size_t)cfg->play_buffer_ms / 1000;
        int16_t *buf = heap_caps_malloc(cap * sizeof *buf, MALLOC_CAP_SPIRAM);
        if (buf == NULL) {
            ESP_LOGE(TAG, "playback buffer of %u samples: out of PSRAM", (unsigned)cap);
            return;
        }
        zl_playq_init(&s_playq, buf, cap, (size_t)cfg->play_rate * PLAY_PRIME_MS / 1000);
    }
    if (cfg->log_read != NULL) {
        s_log = heap_caps_malloc(LOG_CHUNK, MALLOC_CAP_SPIRAM);
        if (s_log == NULL) ESP_LOGE(TAG, "log buffer of %d B: out of PSRAM; no `log` messages", LOG_CHUNK);
    }
    const esp_err_t err = mdns_init();
    if (err != ESP_OK) ESP_LOGE(TAG, "mDNS init failed: %s", esp_err_to_name(err));
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

void zl_report_ambient(double dbfs)
{
    lock();
    s_ambient = dbfs;
    unlock();
    notify(EV_AMBIENT);
}

bool zl_mic_busy(void)
{
    if (s_lock == NULL) return false;
    lock();
    const bool busy = s_sess.open || s_sess.capture;
    unlock();
    return busy;
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

bool zl_playing(void)
{
    if (s_lock == NULL || s_cfg.play_rate <= 0) return false;
    lock();
    const bool playing = s_playq.nstreams > 0;
    unlock();
    return playing;
}

void zl_stop(void)
{
    notify(EV_STOP);
}
