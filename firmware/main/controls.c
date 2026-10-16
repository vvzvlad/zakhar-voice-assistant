// The dial is decoded on every edge of A or B (ui_core's dial_edge, the stock decoder). The stock
// control_volume script steps the volume per count, restarting on a count that comes within
// 16 ms; here the counts of one 16 ms tick make one step in their direction (dial_volume). The
// mute switch is read every tick.
#include "controls.h"

#include <stdatomic.h>

#include "board.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "leds.h"
#include "ui_core.h"
#include "volume.h"

static const char *TAG = "controls";

#define TICK_MS 16

static dial_t s_dial;               // the dial interrupt only
static _Atomic int s_counts;        // dial counts since the last tick
static _Atomic int s_request = -1;  // a set_volume not applied yet

static void dial_isr(void *arg)
{
    (void)arg;
    const int counts = dial_edge(&s_dial, gpio_get_level(PIN_DIAL_A), gpio_get_level(PIN_DIAL_B));
    if (counts != 0) atomic_fetch_add(&s_counts, counts);
}

static void controls_task(void *arg)
{
    (void)arg;
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(TICK_MS));
        leds_set_muted(gpio_get_level(PIN_MUTE) == 1);
        const int request = atomic_exchange(&s_request, -1);
        if (request >= 0) volume_set(request);
        const int counts = atomic_exchange(&s_counts, 0);
        if (counts != 0) volume_set(dial_volume(volume_get(), counts));
    }
}

esp_err_t controls_start(void)
{
    // External pull-ups on all three (R98, R106, R116).
    const gpio_config_t mute = {.pin_bit_mask = 1ULL << PIN_MUTE, .mode = GPIO_MODE_INPUT};
    const gpio_config_t dial = {
        .pin_bit_mask = (1ULL << PIN_DIAL_A) | (1ULL << PIN_DIAL_B),
        .mode = GPIO_MODE_INPUT,
        .intr_type = GPIO_INTR_ANYEDGE,
    };
    esp_err_t err = gpio_config(&mute);
    if (err == ESP_OK) err = gpio_config(&dial);
    if (err == ESP_OK) err = gpio_install_isr_service(0);
    if (err == ESP_OK) err = gpio_isr_handler_add(PIN_DIAL_A, dial_isr, NULL);
    if (err == ESP_OK) err = gpio_isr_handler_add(PIN_DIAL_B, dial_isr, NULL);
    if (err != ESP_OK) return err;
    if (xTaskCreatePinnedToCore(controls_task, "controls", 3 * 1024, NULL, 2, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

void controls_request_volume(int percent)
{
    atomic_store(&s_request, percent);
}
