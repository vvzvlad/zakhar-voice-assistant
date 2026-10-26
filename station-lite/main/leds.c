// The Voice PE's indication, chosen by ui_core.c's ind_choose from the inputs firmware/main/leds.c
// and app_main.c give it, drawn on the board's three RGB zones (/sys/class/leds/micro:<colour>:ring<n>)
// in the Voice PE's colours. Wi-Fi is the board's own business: the network is always up here, and
// there is no provisioning, mute switch or firmware update to show. A thread redraws about 30 times
// a second and writes a file only when its value changes.
#include "leds.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "zakhar_link.h"

static const char *TAG = "leds";

#define ZONES 3
#define TICK_MS 33
#define BLINK_MS 300        // the capture blink, on and off (the Voice PE's)
#define ZONE_MS 200         // listening: a turn of the ring in 600 ms, as the Voice PE's spin
#define PULSE_MS 640        // thinking and error: one pulse, bright to dark and back
#define SLOW_PULSE_MS 2000  // no server
#define DIM 32              // muted: red at an eighth

typedef struct {
    uint8_t r, g, b;
} rgb_t;

// The Voice PE's colours (firmware/main/leds.c).
static const rgb_t RING = {23, 186, 241};
static const rgb_t RED = {255, 0, 0};
static const rgb_t WHITE = {255, 255, 255};
static const rgb_t BLACK = {0, 0, 0};

static const char *const COLOURS[3] = {"red", "green", "blue"};

static pthread_mutex_t s_lock = PTHREAD_MUTEX_INITIALIZER;
static ind_state_t s_ind = {.network = true};  // under s_lock

// The ring's thread only: the brightness files and the values last written to them.
static int s_fd[ZONES][3];
static int s_shown[ZONES][3];

static uint32_t now_ms(void)
{
    return (uint32_t)(esp_timer_get_time() / 1000);
}

static rgb_t scale(rgb_t c, unsigned s)
{
    return (rgb_t){(uint8_t)(c.r * s / 255), (uint8_t)(c.g * s / 255), (uint8_t)(c.b * s / 255)};
}

// 255 at t = 0, down to 0 at half the period and back up.
static unsigned pulse(uint32_t t, uint32_t period)
{
    const uint32_t half = period / 2, p = t % period;
    return p < half ? 255 - p * 255 / half : (p - half) * 255 / half;
}

// The zones at `t` ms into the indication.
static void draw(ind_t ind, uint32_t t, rgb_t out[ZONES])
{
    rgb_t all = BLACK;
    switch (ind) {
    case IND_NO_SERVER:
        all = scale(RED, pulse(t, SLOW_PULSE_MS));
        break;
    case IND_CAPTURE:
        all = t / BLINK_MS % 2 == 0 ? RED : BLACK;
        break;
    case IND_THINKING:
        all = scale(WHITE, pulse(t, PULSE_MS));
        break;
    case IND_REPLYING:
        all = RING;
        break;
    case IND_ERROR:
        all = scale(RED, pulse(t, PULSE_MS));
        break;
    case IND_MUTED:
        all = scale(RED, DIM);
        break;
    default:  // IND_OFF, and IND_LISTENING below
        break;
    }
    for (int z = 0; z < ZONES; z++) out[z] = all;
    if (ind == IND_LISTENING) out[t / ZONE_MS % ZONES] = RING;
}

static void show(const rgb_t zones[ZONES])
{
    for (int z = 0; z < ZONES; z++) {
        const int v[3] = {zones[z].r, zones[z].g, zones[z].b};
        for (int c = 0; c < 3; c++) {
            if (s_fd[z][c] < 0 || v[c] == s_shown[z][c]) continue;
            char text[8];
            const int n = snprintf(text, sizeof text, "%d", v[c]);
            if (pwrite(s_fd[z][c], text, (size_t)n, 0) == n) s_shown[z][c] = v[c];
        }
    }
}

static void *leds_main(void *arg)
{
    (void)arg;
    ind_t shown = IND_OFF;
    uint32_t started = now_ms();
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    for (;;) {
        const bool playing = zl_playing();
        // The clock is read inside the lock, so an error set meanwhile is never newer than `now`.
        pthread_mutex_lock(&s_lock);
        const uint32_t now = now_ms();
        s_ind.playing = playing;
        const ind_t ind = ind_choose(&s_ind, now);
        pthread_mutex_unlock(&s_lock);
        if (ind != shown) {
            shown = ind;
            started = now;
        }
        rgb_t zones[ZONES];
        draw(ind, now - started, zones);
        show(zones);
        next.tv_nsec += TICK_MS * 1000000L;
        if (next.tv_nsec >= 1000000000L) {
            next.tv_sec++;
            next.tv_nsec -= 1000000000L;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    }
    return NULL;
}

void leds_start(void)
{
    for (int z = 0; z < ZONES; z++) {
        for (int c = 0; c < 3; c++) {
            char path[64];
            snprintf(path, sizeof path, "/sys/class/leds/micro:%s:ring%d/brightness", COLOURS[c], z + 1);
            s_fd[z][c] = open(path, O_WRONLY | O_CLOEXEC);
            s_shown[z][c] = -1;
            if (s_fd[z][c] < 0) ESP_LOGW(TAG, "%s: %s", path, strerror(errno));
        }
    }
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_t t;
    const int err = pthread_create(&t, &a, leds_main, NULL);
    pthread_attr_destroy(&a);
    if (err != 0) ESP_LOGE(TAG, "thread: %s", strerror(err));
}

void leds_set_link(bool up)
{
    pthread_mutex_lock(&s_lock);
    s_ind.link = up;
    pthread_mutex_unlock(&s_lock);
}

void leds_set_phase(led_phase_t phase)
{
    pthread_mutex_lock(&s_lock);
    ind_set_phase(&s_ind, phase, now_ms());
    pthread_mutex_unlock(&s_lock);
}

void leds_set_capture(bool on)
{
    pthread_mutex_lock(&s_lock);
    s_ind.capture = on;
    pthread_mutex_unlock(&s_lock);
}

void leds_set_volume(int percent)
{
    pthread_mutex_lock(&s_lock);
    s_ind.volume = percent;
    pthread_mutex_unlock(&s_lock);
}
