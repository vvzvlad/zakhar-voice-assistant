// The Voice PE firmware: Wi-Fi, then the board in the stock order (the XMOS, then the DAC, then the
// amplifier), then the zakhar WebSocket device protocol through zakhar_link.
#include <stdio.h>

#include "aic3204.h"
#include "audio.h"
#include "board.h"
#include "controls.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_app_desc.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_ota_ops.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "leds.h"
#include "net_log.h"
#include "nvs_flash.h"
#include "ota_http.h"
#include "volume.h"
#include "wake_word.h"
#include "xmos.h"
#include "zakhar_link.h"

static const char *TAG = "voicepe";

#define NET_UP BIT0  // the first IP

// A new image is kept once it has run this long with an IP (ESPHome's safe_mode
// boot_is_good_after), and restarts into the previous one when it gets no IP this long.
#define VALID_AFTER_MS (60 * 1000)
#define NEW_IMAGE_IP_WAIT_MS (120 * 1000)

static EventGroupHandle_t s_net;

static void on_net_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) leds_set_network(false);
    if (base == WIFI_EVENT && (id == WIFI_EVENT_STA_START || id == WIFI_EVENT_STA_DISCONNECTED)) {
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *ev = data;
        ESP_LOGI(TAG, "IP " IPSTR, IP2STR(&ev->ip_info.ip));
        leds_set_network(true);
        xEventGroupSetBits(s_net, NET_UP);
    }
}

static void wifi_start(void)
{
    s_net = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    const wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, on_net_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_net_event, NULL));
    wifi_config_t wc = {
        .sta = {
            .ssid = CONFIG_ZAKHAR_WIFI_SSID,
            .password = CONFIG_ZAKHAR_WIFI_PASSWORD,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
}

// ---------------------------------------------------------------- zakhar_link callbacks

static void on_state(zl_state_t state, const char *text)
{
    (void)text;
    switch (state) {
    case ZL_STATE_LISTENING:
        leds_set_phase(LED_PHASE_LISTENING);
        break;
    case ZL_STATE_TRANSCRIBING:
    case ZL_STATE_THINKING:
        leds_set_phase(LED_PHASE_THINKING);
        break;
    case ZL_STATE_SPEAKING:
        leds_set_phase(LED_PHASE_REPLYING);
        break;
    case ZL_STATE_IDLE:
        leds_set_phase(LED_PHASE_IDLE);
        break;
    case ZL_STATE_ERROR:
        leds_set_phase(LED_PHASE_ERROR);
        break;
    }
}

// Asked on a wake word that starts a session; from then until the server's verdict the ring shows
// nothing.
static bool can_start(void)
{
    leds_set_phase(LED_PHASE_WAITING);
    return true;
}

static void on_detect(void)
{
    zl_wake(WAKE_WORD_NAME);
}

// ---------------------------------------------------------------- boot

// True while the running image is new and unconfirmed: the bootloader rolls it back on a restart.
static bool image_pending(void)
{
    esp_ota_img_states_t state;
    return esp_ota_get_state_partition(esp_ota_get_running_partition(), &state) == ESP_OK &&
           state == ESP_OTA_IMG_PENDING_VERIFY;
}

// A task of its own, so the image is marked valid however app_main ends. Marking confirms the
// active otadata entry, so it is skipped once an upload has pointed the boot at the next image.
static void mark_valid_task(void *arg)
{
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(VALID_AFTER_MS));
    if (image_pending() && esp_ota_get_boot_partition() == esp_ota_get_running_partition() &&
        esp_ota_mark_app_valid_cancel_rollback() == ESP_OK) {
        ESP_LOGI(TAG, "a minute with an IP: the image is valid");
    }
    vTaskDelete(NULL);
}

// The bootloader rolls a new image back when it restarts before the image is marked valid. So a
// new image that gets no IP restarts, and one that does is marked valid a minute later: a crash
// within that minute is a restart too.
static void wait_network(void)
{
    const TickType_t wait = image_pending() ? pdMS_TO_TICKS(NEW_IMAGE_IP_WAIT_MS) : portMAX_DELAY;
    if (!(xEventGroupWaitBits(s_net, NET_UP, pdFALSE, pdTRUE, wait) & NET_UP)) {
        ESP_LOGE(TAG, "a new image with no IP in %d s: restarting into the previous one", NEW_IMAGE_IP_WAIT_MS / 1000);
        esp_restart();
    }
    if (xTaskCreate(mark_valid_task, "mark_valid", 4 * 1024, NULL, 1, NULL) != pdPASS) {
        ESP_LOGE(TAG, "mark_valid task create failed: the image stays unconfirmed");
    }
}

void app_main(void)
{
    const gpio_config_t amp = {.pin_bit_mask = 1ULL << PIN_AMP_EN, .mode = GPIO_MODE_OUTPUT};
    ESP_ERROR_CHECK(gpio_config(&amp));
    gpio_set_level(PIN_AMP_EN, 0);
    ESP_ERROR_CHECK(nvs_flash_init());
    volume_init();
    leds_start();
    wifi_start();

    // POST /ota and the network log come up with the first IP, before the board: nothing below
    // aborts, a failed bring-up step is logged and ends app_main, leaving Wi-Fi, POST /ota and the
    // network log up. A new image is marked valid a minute after the IP (wait_network).
    wait_network();
    ota_http_start();
    net_log_start();

    const i2c_master_bus_config_t bc = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = PIN_I2C_SDA,
        .scl_io_num = PIN_I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
    };
    i2c_master_bus_handle_t bus;
    wake_word_set_detect_cb(on_detect);
    esp_err_t err = i2c_new_master_bus(&bc, &bus);
    if (err == ESP_OK) err = xmos_boot(bus);
    // The I2S clocks run once the XMOS answers; a slave channel started before that fails.
    if (err == ESP_OK) err = audio_start();
    if (err == ESP_OK) err = aic3204_init(bus, volume_get());
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "audio bring-up failed: %s", esp_err_to_name(err));
        return;
    }
    gpio_set_level(PIN_AMP_EN, 1);
    ESP_LOGI(TAG, "amplifier on");

    char fw[64];
    snprintf(fw, sizeof fw, "voicepe %s", esp_app_get_description()->version);
    const zl_config_t cfg = {
        .url = CONFIG_ZAKHAR_SERVER_URL,
        .id = CONFIG_ZAKHAR_DEVICE_ID,
        .fw = fw,
        .model = WAKE_WORD_MODEL_VERSION,
        .screen = false,
        .mic_channels = 2,
        .preroll_ms = 2000,
        .mic_position = audio_mic_position,
        .mic_read = audio_mic_read,
        .wake_peak_take = wake_word_take_peak,
        .play_rate = 48000,
        .play_buffer_ms = 4000,
        .volume = volume_get(),
    };
    const zl_callbacks_t cb = {
        .on_connection = leds_set_link,
        .on_state = on_state,
        .on_config = wake_word_set_cutoffs,
        .on_set_volume = controls_request_volume,
        .on_capture = leds_set_capture,
        .can_start = can_start,
    };
    zl_start(&cfg, &cb);
    // zl_start() reports no failure; its task exists only when the start got through (the playback
    // buffer, the lock, the task), and zl_play_read() and zl_report_volume() must not run without
    // it.
    if (xTaskGetHandle("zakhar_link") == NULL) {
        ESP_LOGE(TAG, "zakhar_link did not start");
        return;
    }
    err = audio_start_playback();
    if (err == ESP_OK) err = controls_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "bring-up failed: %s", esp_err_to_name(err));
        return;
    }
    ESP_LOGI(TAG, "%s up as %s", fw, CONFIG_ZAKHAR_DEVICE_ID);
}
