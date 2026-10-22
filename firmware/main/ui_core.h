// The pure decisions of the Voice PE's indication and controls: no ESP headers, host-tested
// (make fw-voicepe-host-test). Which indication the LED ring shows, the dial and the center
// button's short press and hold.
#pragma once

#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------- indication

typedef enum {
    LED_PHASE_IDLE,
    LED_PHASE_LISTENING,
    LED_PHASE_THINKING,   // transcribing and thinking
    LED_PHASE_REPLYING,
    LED_PHASE_ERROR,      // shown for IND_ERROR_MS, then idle
} led_phase_t;

typedef enum {
    IND_OFF,
    IND_PROVISIONING,  // warm white twinkle, the stock Improv BLE indication
    IND_NO_SERVER,     // red twinkle
    IND_CAPTURE,    // red blink
    IND_LISTENING,
    IND_THINKING,   // "Thinking White"
    IND_REPLYING,
    IND_ERROR,
    IND_MUTED,      // "Muted or Silent"
    IND_OTA,        // white fill, the share of a firmware update received
} ind_t;

#define IND_ERROR_MS 1000  // stock on_end keeps the error phase a second

typedef struct {
    bool provisioning;  // waiting for a Wi-Fi network over BLE provisioning
    bool network;       // Wi-Fi has an IP
    bool link;     // zakhar_link is connected
    bool capture;
    bool playing;  // playback audio is going out
    bool muted;    // the mute switch
    int volume;
    led_phase_t phase;
    uint32_t error_at;  // ms, when the phase became LED_PHASE_ERROR
    bool ota;           // a firmware update is being received
    int ota_percent;    // its share received, 0..100
} ind_state_t;

void ind_set_phase(ind_state_t *s, led_phase_t phase, uint32_t now_ms);

// The indication at now_ms: a firmware update over everything, then the stock control_leds order:
// provisioning (the stock Improv BLE), no server, the capture, the session phase, an announcement
// playing, mute or volume 0, dark. An error phase older than IND_ERROR_MS turns idle first.
ind_t ind_choose(ind_state_t *s, uint32_t now_ms);

// How many of `leds` LEDs show `percent`: rounded up, so the first percent already lights one.
int ind_fill(int percent, int leds);

// ---------------------------------------------------------------- the dial

#define DIAL_STEP 5  // the stock volume_increment 0.05

typedef struct {
    uint16_t state;
    bool started;
} dial_t;  // zero-initialised

// One edge of the dial, given the levels of A and B after it; returns the counts it makes (+1, -1
// or 0). ESPHome's rotary_encoder decoder at `resolution: 2`: two counts per A-B cycle, +1 when A
// leads; the first edge only sets the state.
int dial_edge(dial_t *d, bool a, bool b);

// The volume after one tick that collected `counts`: one DIAL_STEP in their direction, within
// 0..100.
int dial_volume(int volume, int counts);

// ---------------------------------------------------------------- the center button

#define FORGET_HOLD_MS 10000  // a hold this long forgets the stored Wi-Fi network

typedef enum {
    BUTTON_NONE,
    BUTTON_SHORT,  // a press released before its hold
    BUTTON_HOLD,   // a press held FORGET_HOLD_MS
} button_event_t;

typedef struct {
    bool down;
    bool fired;
    uint32_t since;  // ms, when the press began
} button_t;  // zero-initialised

// One look at the button at now_ms: BUTTON_HOLD once per press, when it has been held
// FORGET_HOLD_MS; BUTTON_SHORT at the release of a press that did not get there. A held press
// gives nothing at its release.
button_event_t button_step(button_t *b, bool pressed, uint32_t now_ms);
