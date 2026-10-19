// The dial is decoded on every edge of A or B (ui_core's dial_edge, the stock decoder). The stock
// control_volume script steps the volume per count, restarting on a count that comes within
// 16 ms; here the counts of one 16 ms tick make one step in their direction (dial_volume). The
// mute switch is read every tick. The center button is polled on a task of its own (hold_step):
// the controls task starts only once the board and zakhar_link are up, the button must work before.
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
#define BUTTON_TICK_MS 50

static dial_t s_dial;               // the dial interrupt only
static _Atomic int s_counts;        // dial counts since the last tick
static _Atomic int s_request = -1;  // a set_volume not applied yet
static _Atomic int s_boost = -1;    // a volume_boost not applied yet
static void (*s_on_hold)(void);     // set before the button task starts

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
        const int boost = atomic_exchange(&s_boost, -1);
        if (boost >= 0) volume_set_boost(boost);
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

void controls_request_boost(int points)
{
    atomic_store(&s_boost, points);
}

static void button_task(void *arg)
{
    (void)arg;
    hold_t hold = {0};
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(BUTTON_TICK_MS));
        if (hold_step(&hold, gpio_get_level(PIN_BUTTON) == 0, pdTICKS_TO_MS(xTaskGetTickCount()))) s_on_hold();
    }
}

esp_err_t controls_button_start(void (*on_hold)(void))
{
    // The stock config reads GPIO0 with the internal pull-up off (the strapping pin is pulled up on
    // the board); here it is on as well, harmlessly.
    const gpio_config_t button = {
        .pin_bit_mask = 1ULL << PIN_BUTTON,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
    };
    const esp_err_t err = gpio_config(&button);
    if (err != ESP_OK) return err;
    s_on_hold = on_hold;
    if (xTaskCreate(button_task, "button", 3 * 1024, NULL, 1, NULL) != pdPASS) {
        ESP_LOGE(TAG, "button task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
