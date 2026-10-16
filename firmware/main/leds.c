// The stock indication of home-assistant-voice.yaml @26.5.0 redrawn without ESPHome: the effects
// of its voice_assistant_leds light ("Listening For Command", "Replying", "Error", "Muted or
// Silent", the red "Twinkle" of no connection, the warm white "Twinkle" of Improv BLE, here shown
// while BLE provisioning waits for a network) and the zakhar config's "Thinking White" and capture
// blink, with ESPHome's light pipeline under them (light/addressable_light_effect.h,
// light/esp_color_correction.h, core/color.h at 2026.5.3): each LED is
// gamma_2.8(value * brightness), the ring is remapped (logical i = physical (i + 7) % 12) and the
// LED power follows the stock power_supply (on 20 ms before the first lit frame, off 10 s after
// the last LED goes dark).
//
// Which indication wins is ui_core's ind_choose; its inputs live under one spinlock, set from any
// task and read by the leds task once per tick.
#include "leds.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "board.h"
#include "driver/gpio.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "led_strip.h"

#define N 12
#define TICK_MS 16           // ESPHome's loop period, at which the stock effects run
#define POWER_ON_MS 20       // stock power_supply enable time
#define POWER_KEEP_MS 10000  // stock power_supply keep-on time
#define BLINK_MS 300         // the capture blink, on and off

typedef struct {
    uint8_t r, g, b;
} rgb_t;

// voice_assistant_leds brightness on ESPHome's uint8 scale, roundf(x * 255): the user ring's
// initial 66 % through the scripts' max(66 %, 20 %), the error script's 10 % more, "Thinking White"
// at 100 %, the capture blink at 80 %.
#define BRIGHT_RING 168
#define BRIGHT_ERROR 194
#define BRIGHT_FULL 255
#define BRIGHT_CAPTURE 204

// The user ring's initial colour 9.4 / 73.3 / 94.9 % as the effects sample it, uint8(x * 255);
// WARM is the Improv BLE script's 100 / 89 / 71 % the same way.
static const rgb_t RING = {23, 186, 241};
static const rgb_t WARM = {255, 226, 181};
static const rgb_t RED = {255, 0, 0};
static const rgb_t WHITE = {255, 255, 255};
static const rgb_t BLACK = {0, 0, 0};

static portMUX_TYPE s_lock = portMUX_INITIALIZER_UNLOCKED;
static ind_state_t s_ind;  // under s_lock

// The leds task only.
static led_strip_handle_t s_strip;
static uint8_t s_gamma[256];
static ind_t s_fx = IND_OFF;
static bool s_initial;         // the effect's first run (ESPHome's initial_run)
static uint32_t s_started;     // when the effect started
static uint32_t s_last_run;    // the effect's last update
static uint8_t s_anim;         // global_led_animation_index, shared by the effects as in the stock config
static uint8_t s_step;         // the pulse effects' brightness_step
static bool s_decreasing;
static uint8_t s_twinkle[N];   // the twinkle's per-LED effect data
static uint32_t s_progress;    // the twinkle's last_progress
static rgb_t s_px[N];          // the logical ring
static uint8_t s_bright;
static uint8_t s_shown[N][3];  // the last frame written, physical order
static bool s_power, s_off_pending;
static uint32_t s_off_at;

static uint32_t now_ms(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000);
}

// Color * uint8 (esp_scale8 per channel).
static rgb_t scale(rgb_t c, uint8_t s)
{
    return (rgb_t){(uint8_t)(c.r * (1 + s) / 256), (uint8_t)(c.g * (1 + s) / 256), (uint8_t)(c.b * (1 + s) / 256)};
}

static void fill(rgb_t c)
{
    for (int i = 0; i < N; i++) s_px[i] = c;
}

// sin16_c / half_sin8 of addressable_light_effect.h.
static int16_t sin16(uint16_t theta)
{
    static const uint16_t BASE[] = {0, 6393, 12539, 18204, 23170, 27245, 30273, 32137};
    static const uint8_t SLOPE[] = {49, 48, 44, 38, 31, 23, 14, 4};
    uint16_t offset = (theta & 0x3FFF) >> 3;
    if (theta & 0x4000) offset = 2047 - offset;
    const uint8_t section = offset / 256;
    const uint8_t secoffset8 = (uint8_t)offset / 2;
    const int16_t y = (int16_t)(SLOPE[section] * secoffset8 + BASE[section]);
    return (theta & 0x8000) ? -y : y;
}

static uint8_t half_sin8(uint8_t v)
{
    return (uint8_t)(sin16((uint16_t)(v * 128u)) >> 8);
}

// ESPHome's lambda effects run on their first call and then once update_interval has passed.
static bool due(uint32_t now, uint32_t interval)
{
    if (!s_initial && now - s_last_run < interval) return false;
    s_last_run = now;
    return true;
}

// "Listening For Command": two comet heads opposite each other, spinning clockwise.
static void listening(void)
{
    const int a = s_anim;
    for (int i = 0; i < N; i++) {
        if (i == a % N || i == (a + 6) % N) {
            s_px[i] = RING;
        } else if (i == (a + 11) % N || i == (a + 5) % N) {
            s_px[i] = scale(RING, 192);
        } else if (i == (a + 10) % N || i == (a + 4) % N) {
            s_px[i] = scale(RING, 128);
        } else {
            s_px[i] = BLACK;
        }
    }
    s_anim = (s_anim + 1) % N;
}

// "Replying": the same, anticlockwise.
static void replying(void)
{
    s_anim = (N + s_anim - 1) % N;
    const int a = s_anim;
    for (int i = 0; i < N; i++) {
        if (i == a % N || i == (a + 6) % N) {
            s_px[i] = RING;
        } else if (i == (a + 1) % N || i == (a + 7) % N) {
            s_px[i] = scale(RING, 192);
        } else if (i == (a + 2) % N || i == (a + 8) % N) {
            s_px[i] = scale(RING, 128);
        } else {
            s_px[i] = BLACK;
        }
    }
}

// "Thinking White" (two opposite LEDs where the spin stopped) and "Error" (the whole ring): a
// brightness pulse in 10 steps.
static void pulse(rgb_t color, bool two)
{
    if (s_initial) {
        s_step = 0;
        s_decreasing = true;
    }
    const rgb_t c = scale(color, (uint8_t)(255 / 10 * (10 - s_step)));
    for (int i = 0; i < N; i++) {
        s_px[i] = (!two || i == s_anim % N || i == (s_anim + 6) % N) ? c : BLACK;
    }
    if (s_decreasing) {
        s_step++;
    } else {
        s_step--;
    }
    if (s_step == 0 || s_step == 10) s_decreasing = !s_decreasing;
}

// "Muted or Silent" with the user ring off: red next to the microphones when muted, red next to
// the speaker grill at volume 0.
static void muted(const ind_state_t *in)
{
    fill(BLACK);
    if (in->muted) {
        s_px[3] = RED;
        s_px[9] = RED;
    }
    if (in->volume == 0) s_px[6] = RED;
}

// addressable_twinkle, twinkle_probability 50 %, progress_interval 4 ms: run every loop.
static void twinkle(rgb_t color, uint32_t now)
{
    uint8_t add = 0;
    if (now - s_progress > 4) {
        const uint32_t a = (now - s_progress) / 4;
        add = (uint8_t)a;
        s_progress += a * 4;
    }
    for (int i = 0; i < N; i++) {
        if (s_twinkle[i] == 0) {
            s_px[i] = BLACK;
            continue;
        }
        s_px[i] = scale(color, half_sin8(s_twinkle[i]));
        const uint8_t next = (uint8_t)(s_twinkle[i] + add);
        s_twinkle[i] = next < s_twinkle[i] ? 0 : next;
    }
    while ((float)esp_random() / (float)UINT32_MAX < 0.5f) {
        const uint32_t pos = esp_random() % N;
        if (s_twinkle[pos] == 0) s_twinkle[pos] = 1;
    }
}

static void draw(ind_t fx, const ind_state_t *in, uint32_t now)
{
    if (fx != s_fx) {
        s_fx = fx;
        s_initial = true;
        s_started = now;
        s_progress = now;
        memset(s_twinkle, 0, sizeof s_twinkle);
    }
    switch (fx) {
    case IND_OFF:
        fill(BLACK);
        break;
    case IND_PROVISIONING:
        s_bright = BRIGHT_RING;
        twinkle(WARM, now);
        break;
    case IND_NO_SERVER:
        s_bright = BRIGHT_RING;
        twinkle(RED, now);
        break;
    case IND_CAPTURE:
        s_bright = BRIGHT_CAPTURE;
        fill((now - s_started) / BLINK_MS % 2 == 0 ? RED : BLACK);
        break;
    case IND_LISTENING:
        s_bright = BRIGHT_RING;
        if (due(now, 50)) listening();
        break;
    case IND_THINKING:
        s_bright = BRIGHT_FULL;
        if (due(now, 10)) pulse(WHITE, true);
        break;
    case IND_REPLYING:
        s_bright = BRIGHT_RING;
        if (due(now, 50)) replying();
        break;
    case IND_ERROR:
        s_bright = BRIGHT_ERROR;
        if (due(now, 10)) pulse(RED, false);
        break;
    case IND_MUTED:
        s_bright = BRIGHT_RING;
        if (due(now, 16)) muted(in);
        break;
    }
    s_initial = false;
}

static void show(uint32_t now)
{
    uint8_t out[N][3];
    bool any = false;
    for (int i = 0; i < N; i++) {
        uint8_t *o = out[(i + 7) % N];
        // esp_scale8_twice(v, 255, brightness), then the gamma table.
        o[0] = s_gamma[(s_px[i].r * (1 + s_bright)) >> 8];
        o[1] = s_gamma[(s_px[i].g * (1 + s_bright)) >> 8];
        o[2] = s_gamma[(s_px[i].b * (1 + s_bright)) >> 8];
        any = any || o[0] != 0 || o[1] != 0 || o[2] != 0;
    }
    bool fresh = false;
    if (any) {
        s_off_pending = false;
        if (!s_power) {
            gpio_set_level(PIN_LED_POWER, 1);
            s_power = true;
            fresh = true;
            vTaskDelay(pdMS_TO_TICKS(POWER_ON_MS));
        }
    } else if (s_power && !s_off_pending) {
        s_off_pending = true;
        s_off_at = now + POWER_KEEP_MS;
    }
    if (s_power && (fresh || memcmp(out, s_shown, sizeof out) != 0)) {
        for (int i = 0; i < N; i++) led_strip_set_pixel(s_strip, i, out[i][0], out[i][1], out[i][2]);
        led_strip_refresh(s_strip);
        memcpy(s_shown, out, sizeof out);
    }
    if (s_off_pending && (int32_t)(now - s_off_at) >= 0) {
        gpio_set_level(PIN_LED_POWER, 0);
        s_power = false;
        s_off_pending = false;
    }
}

static void leds_task(void *arg)
{
    (void)arg;
    TickType_t wake = xTaskGetTickCount();
    for (;;) {
        // The clock is read inside the lock, so an error set meanwhile is never newer than `now`.
        taskENTER_CRITICAL(&s_lock);
        const uint32_t now = now_ms();
        const ind_t ind = ind_choose(&s_ind, now);
        const ind_state_t in = s_ind;
        taskEXIT_CRITICAL(&s_lock);
        draw(ind, &in, now);
        show(now);
        vTaskDelayUntil(&wake, pdMS_TO_TICKS(TICK_MS));
    }
}

void leds_start(void)
{
    // ESPHome's gamma_correct 2.8 table: uint16 round((i / 255) ^ 2.8 * 65535), at least 1 for i > 0,
    // read back as (v + 128) / 257.
    for (int i = 1; i < 256; i++) {
        long v = lround(pow(i / 255.0, 2.8) * 65535.0);
        if (v < 1) v = 1;
        s_gamma[i] = (uint8_t)((v + 128) / 257);
    }
    const gpio_config_t power = {.pin_bit_mask = 1ULL << PIN_LED_POWER, .mode = GPIO_MODE_OUTPUT};
    ESP_ERROR_CHECK(gpio_config(&power));
    gpio_set_level(PIN_LED_POWER, 0);
    const led_strip_config_t sc = {
        .strip_gpio_num = PIN_LED_DATA,
        .max_leds = N,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
    };
    const led_strip_rmt_config_t rc = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .mem_block_symbols = 192,  // the stock rmt_symbols
    };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&sc, &rc, &s_strip));
    xTaskCreatePinnedToCore(leds_task, "leds", 3 * 1024, NULL, 2, NULL, 1);
}

// The setters touch s_ind only inside the spinlock, for a few instructions.
#define LOCKED(stmt)                  \
    do {                              \
        taskENTER_CRITICAL(&s_lock);  \
        stmt;                         \
        taskEXIT_CRITICAL(&s_lock);   \
    } while (0)

void leds_set_provisioning(bool on)
{
    LOCKED(s_ind.provisioning = on);
}

void leds_set_network(bool up)
{
    LOCKED(s_ind.network = up);
}

void leds_set_link(bool up)
{
    LOCKED(s_ind.link = up);
}

void leds_set_phase(led_phase_t phase)
{
    LOCKED(ind_set_phase(&s_ind, phase, now_ms()));
}

void leds_set_capture(bool on)
{
    LOCKED(s_ind.capture = on);
}

void leds_set_playing(bool on)
{
    LOCKED(s_ind.playing = on);
}

void leds_set_muted(bool on)
{
    LOCKED(s_ind.muted = on);
}

void leds_set_volume(int percent)
{
    LOCKED(s_ind.volume = percent);
}
