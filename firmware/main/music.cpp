// The music player (music.h): a sendspin-cpp client with the player role alone. Music Assistant
// finds it by mDNS (`_sendspin._tcp`, TXT path and name) and connects to its WebSocket server; the
// player's sync task decodes the stream (FLAC or PCM, 48 kHz stereo 16-bit: the I2S bus has no
// other rate) into a ring the speaker task takes it from every block, mixed under zakhar's
// playback. Music Assistant's volume and mute are the music's own digital gain, the DAC volume
// applies to everything. The keypair and pairing records live in the NVS namespace `sendspin`.
// The client's loop, and with it every NVS write, runs on this file's task: its stack is internal.
// The library's tasks sit below the mic task (5) and the speaker task (6), the sync task above the
// two that feed it as the library asks; their stacks are in PSRAM.
#include "music.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "mdns.h"
#include "music_ring.h"
#include "nvs.h"
#include "sendspin/client.h"
#include "sendspin/player_role.h"

static const char *TAG = "music";

#define RATE 48000
#define FRAME_BYTES (2 * sizeof(int16_t))
#define RING_FRAMES (RATE / 4)  // 250 ms
#define GAIN_UNITY 65536u
#define INITIAL_VOLUME 100
#define LOOP_MS 10
#define LOOP_STACK (8 * 1024)
#define STACK_LOG_MS (30 * 1000)  // the loop task's stack use is logged this long after the start
#define LOOP_PRIO 2
#define HTTPD_PRIO 3
#define PROTOCOL_PRIO 3
#define SYNC_PRIO 4

static std::atomic<bool> s_net_up;
static std::atomic<bool> s_active;
static std::atomic<bool> s_muted;
static std::atomic<uint32_t> s_gain{GAIN_UNITY};
static sendspin::PlayerRole *s_player;

// The stream from the sync task (on_audio_write, the one writer) to the speaker task (the one
// reader) through a music ring (music_ring.h) of RING_FRAMES slots. Its head and tail live here as
// atomics: each side copies them in and stores back its own. A stream's end flushes it through the
// reader: the writer is idle then, and the reader drops the frames up to the head it had
// (s_flush_to), which keeps the frames a next stream writes after it.
static int16_t *s_ring;
static std::atomic<uint32_t> s_head, s_tail;
static std::atomic<uint32_t> s_flush_to;
static std::atomic<bool> s_flush;
// Given by the reader once it has taken frames: a writer waiting for space tries again.
static SemaphoreHandle_t s_space;

static size_t ring_write(const uint8_t *in, size_t frames)
{
    music_ring_t r = {s_ring, RING_FRAMES, s_head.load(std::memory_order_relaxed),
                      s_tail.load(std::memory_order_acquire)};
    const size_t n = music_ring_write(&r, in, frames);
    s_head.store(r.head, std::memory_order_release);
    return n;
}

class Player : public sendspin::PlayerRoleListener {
public:
    // The sync task: as much as fits, waiting for the speaker task to make room up to timeout_ms.
    size_t on_audio_write(uint8_t *data, size_t length, uint32_t timeout_ms) override
    {
        const size_t frames = length / FRAME_BYTES;
        const TickType_t start = xTaskGetTickCount(), wait = pdMS_TO_TICKS(timeout_ms);
        size_t done = ring_write(data, frames);
        while (done < frames) {
            const TickType_t spent = xTaskGetTickCount() - start;
            if (spent >= wait || xSemaphoreTake(s_space, wait - spent) != pdTRUE) break;
            done += ring_write(data + done * FRAME_BYTES, frames - done);
        }
        return done * FRAME_BYTES;
    }

    void on_stream_start() override
    {
        ESP_LOGI(TAG, "stream on");
        s_active.store(true, std::memory_order_release);
    }

    void on_stream_end() override
    {
        ESP_LOGI(TAG, "stream off");
        s_flush_to.store(s_head.load(std::memory_order_acquire), std::memory_order_relaxed);
        s_flush.store(true, std::memory_order_release);
        s_active.store(false, std::memory_order_release);
    }

    // The spec's curve: amplitude = (volume / 100)^1.5.
    void on_volume_changed(uint8_t volume) override
    {
        const float v = std::min<uint8_t>(volume, 100) / 100.0f;
        s_gain.store((uint32_t)std::lround(std::pow(v, 1.5f) * GAIN_UNITY), std::memory_order_relaxed);
    }

    void on_mute_changed(bool muted) override
    {
        s_muted.store(muted, std::memory_order_relaxed);
    }
};

class Network : public sendspin::SendspinNetworkProvider {
public:
    bool is_network_ready() override
    {
        return s_net_up.load(std::memory_order_relaxed);
    }
};

// Every key in one NVS namespace, the blob stored as the library hands it over. NVS writes are
// durable when nvs_set_blob returns.
class NvsStore : public sendspin::SendspinPersistenceProvider {
public:
    nvs_handle_t nvs = 0;

    std::optional<std::vector<uint8_t>> load_blob(const std::string &key) override
    {
        size_t len = 0;
        if (nvs_get_blob(nvs, key.c_str(), NULL, &len) != ESP_OK) return std::nullopt;
        std::vector<uint8_t> blob(len);
        if (nvs_get_blob(nvs, key.c_str(), blob.data(), &len) != ESP_OK) return std::nullopt;
        return blob;
    }

    bool save_blob(const std::string &key, const uint8_t *data, size_t len) override
    {
        return nvs_set_blob(nvs, key.c_str(), data, len) == ESP_OK;
    }

    bool commit() override
    {
        return nvs_commit(nvs) == ESP_OK;
    }
};

static void log_internal(const char *when)
{
    ESP_LOGI(TAG, "%s: %u B of internal RAM free, largest block %u B", when,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
}

// Starts the client, then advertises it: the mDNS host is the DHCP host name app_main gave the
// station, the instance and the TXT name `Zakhar <MAC tail>`. Then the client's loop, for good.
static void music_task(void *arg)
{
    (void)arg;
    static Player listener;
    static Network network;
    static NvsStore store;
    esp_err_t err = nvs_open("sendspin", NVS_READWRITE, &store.nvs);
    uint8_t mac[6];
    if (err == ESP_OK) err = esp_read_mac(mac, ESP_MAC_WIFI_STA);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "no player: %s", esp_err_to_name(err));
        vTaskDelete(NULL);
        return;
    }
    char name[16];
    snprintf(name, sizeof name, "Zakhar %02x%02x%02x", mac[3], mac[4], mac[5]);

    sendspin::SendspinClientConfig cfg;
    cfg.name = name;
    // POST /ota's server holds ESP_HTTPD_DEF_CTRL_PORT.
    cfg.httpd_ctrl_port = ESP_HTTPD_DEF_CTRL_PORT + 1;
    cfg.httpd_psram_stack = true;
    cfg.httpd_priority = HTTPD_PRIO;
    cfg.protocol_task_psram_stack = true;
    cfg.protocol_task_priority = PROTOCOL_PRIO;
    // The protocol's JSON documents go to PSRAM: the arena would be internal RAM.
    cfg.json_arena_size = 0;
    auto *client = new sendspin::SendspinClient(std::move(cfg));
    sendspin::PlayerRoleConfig pc;
    pc.audio_formats = {
        {sendspin::SendspinCodecFormat::FLAC, 2, RATE, 16},
        {sendspin::SendspinCodecFormat::PCM, 2, RATE, 16},
    };
    pc.psram_stack = true;
    pc.priority = SYNC_PRIO;
    sendspin::PlayerRole &player = client->add_player(std::move(pc));
    player.set_listener(&listener);
    // The state the first client/state reports, as the gain starts.
    player.update_volume(INITIAL_VOLUME);
    player.update_muted(false);
    s_player = &player;
    client->set_network_provider(&network);
    client->set_persistence_provider(&store);
    // Music Assistant plays without pairing, as on the stock ESPHome firmware; the library starts
    // with this off and never stores it.
    client->set_unpaired_access_enabled(true);
    log_internal("before the client starts");
    if (!client->start()) {
        ESP_LOGE(TAG, "no player: the client did not start");
        vTaskDelete(NULL);
        return;
    }
    const TickType_t started = xTaskGetTickCount();

    const uint16_t port = sendspin::SendspinClientConfig::DEFAULT_SERVER_PORT;
    const char *host = NULL;
    err = esp_netif_get_hostname(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &host);
    if (err == ESP_OK) err = mdns_hostname_set(host);
    mdns_txt_item_t txt[] = {{"path", "/sendspin"}, {"name", name}};
    if (err == ESP_OK) err = mdns_service_add(name, "_sendspin", "_tcp", port, txt, 2);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "not advertised by mDNS: %s", esp_err_to_name(err));
    } else {
        ESP_LOGI(TAG, "Sendspin player \"%s\" on %s.local:%u", name, host, (unsigned)port);
    }
    log_internal("player up");
    bool stack_logged = false;
    for (;;) {
        client->loop();
        if (!stack_logged && xTaskGetTickCount() - started >= pdMS_TO_TICKS(STACK_LOG_MS)) {
            ESP_LOGI(TAG, "music task: %u B of its stack never used", (unsigned)uxTaskGetStackHighWaterMark(NULL));
            stack_logged = true;
        }
        vTaskDelay(pdMS_TO_TICKS(LOOP_MS));
    }
}

void music_set_network(bool up)
{
    s_net_up.store(up, std::memory_order_relaxed);
}

esp_err_t music_start(void)
{
    s_ring = (int16_t *)heap_caps_malloc(RING_FRAMES * FRAME_BYTES, MALLOC_CAP_SPIRAM);
    s_space = xSemaphoreCreateBinary();
    if (s_ring == NULL || s_space == NULL) {
        ESP_LOGE(TAG, "music ring: out of memory");
        return ESP_ERR_NO_MEM;
    }
    // Core 0, with the network, below zakhar_link.
    if (xTaskCreatePinnedToCore(music_task, "music", LOOP_STACK, NULL, LOOP_PRIO, NULL, 0) != pdPASS) {
        ESP_LOGE(TAG, "music task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

bool music_active(void)
{
    return s_active.load(std::memory_order_acquire);
}

size_t music_read(int16_t *out, size_t max_frames)
{
    if (!s_active.load(std::memory_order_acquire)) return 0;
    // The head is loaded after the flush is taken, so it is at least the head the flush recorded.
    const bool flush = s_flush.exchange(false, std::memory_order_acq_rel);
    const uint32_t head = s_head.load(std::memory_order_acquire);
    music_ring_t r = {s_ring, RING_FRAMES, head, s_tail.load(std::memory_order_relaxed)};
    if (flush) music_ring_flush(&r, s_flush_to.load(std::memory_order_relaxed));
    const size_t n = music_ring_read(&r, out, max_frames);
    s_tail.store(r.tail, std::memory_order_release);
    if (n > 0 || flush) xSemaphoreGive(s_space);
    return n;
}

void music_played(size_t frames, int64_t finish_us)
{
    s_player->notify_audio_played((uint32_t)frames, finish_us);
}

uint32_t music_gain(void)
{
    return s_muted.load(std::memory_order_relaxed) ? 0 : s_gain.load(std::memory_order_relaxed);
}
