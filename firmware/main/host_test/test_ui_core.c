// Host test of the Voice PE firmware's pure indication, dial and button decisions (ui_core.c): the
// stock control_leds priority under a firmware update, the error hold, the update's fill of the
// ring, the stock dial decoder with one volume step per tick, and the center button's short press
// and the hold that forgets the Wi-Fi network.
// `make fw-voicepe-host-test` builds it.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "ui_core.h"

static ind_state_t online(void)
{
    return (ind_state_t){.network = true, .link = true, .volume = 50, .phase = LED_PHASE_IDLE};
}

static void test_priority(void)
{
    ind_state_t s = online();
    s.capture = s.playing = s.muted = true;
    s.phase = LED_PHASE_LISTENING;
    s.network = false;
    s.provisioning = true;
    assert(ind_choose(&s, 0) == IND_PROVISIONING);  // waiting for a network beats everything
    s.network = true;
    assert(ind_choose(&s, 0) == IND_PROVISIONING);  // an IP alone does not end it
    s.provisioning = false;
    s.network = false;
    assert(ind_choose(&s, 0) == IND_NO_SERVER);  // then no Wi-Fi
    s.network = true;
    s.link = false;
    assert(ind_choose(&s, 0) == IND_NO_SERVER);  // no server too
    s.link = true;
    assert(ind_choose(&s, 0) == IND_CAPTURE);    // then the capture
    s.capture = false;
    assert(ind_choose(&s, 0) == IND_LISTENING);  // then the session, over playback and mute
    s.phase = LED_PHASE_THINKING;
    assert(ind_choose(&s, 0) == IND_THINKING);
    s.phase = LED_PHASE_REPLYING;
    assert(ind_choose(&s, 0) == IND_REPLYING);
    s.phase = LED_PHASE_IDLE;
    assert(ind_choose(&s, 0) == IND_REPLYING);   // an announcement playing, over mute
    s.playing = false;
    assert(ind_choose(&s, 0) == IND_MUTED);
    s.muted = false;
    assert(ind_choose(&s, 0) == IND_OFF);
    s.volume = 0;
    assert(ind_choose(&s, 0) == IND_MUTED);      // volume 0 shows as silent
    s.provisioning = s.capture = true;
    s.network = false;
    s.ota = true;
    assert(ind_choose(&s, 0) == IND_OTA);        // a firmware update beats everything
}

static void test_fill(void)
{
    assert(ind_fill(0, 12) == 0);
    assert(ind_fill(1, 12) == 1);    // the first percent shows
    assert(ind_fill(8, 12) == 1);
    assert(ind_fill(9, 12) == 2);
    assert(ind_fill(50, 12) == 6);
    assert(ind_fill(99, 12) == 12);
    assert(ind_fill(100, 12) == 12);
}

static void test_error_hold(void)
{
    ind_state_t s = online();
    ind_set_phase(&s, LED_PHASE_ERROR, 1000);
    assert(ind_choose(&s, 1000) == IND_ERROR);
    assert(ind_choose(&s, 1999) == IND_ERROR);
    assert(ind_choose(&s, 2000) == IND_OFF && s.phase == LED_PHASE_IDLE);

    // A new session within the second replaces the error, and the hold does not touch it.
    s = online();
    ind_set_phase(&s, LED_PHASE_ERROR, 0);
    assert(ind_choose(&s, 400) == IND_ERROR);
    ind_set_phase(&s, LED_PHASE_LISTENING, 500);
    assert(ind_choose(&s, 1500) == IND_LISTENING && s.phase == LED_PHASE_LISTENING);

    // While the server is away the error waits; it ends at the first look after.
    s = online();
    ind_set_phase(&s, LED_PHASE_ERROR, 0);
    s.link = false;
    assert(ind_choose(&s, 5000) == IND_NO_SERVER && s.phase == LED_PHASE_ERROR);
    s.link = true;
    assert(ind_choose(&s, 5001) == IND_OFF && s.phase == LED_PHASE_IDLE);

    // Across the wrap of the millisecond clock.
    s = online();
    ind_set_phase(&s, LED_PHASE_ERROR, UINT32_MAX - 100);
    assert(ind_choose(&s, 898) == IND_ERROR);
    assert(ind_choose(&s, 899) == IND_OFF);
}

// Feeds (A, B) levels edge by edge and checks the counts of each edge.
static void edges(dial_t *d, const int (*ab)[2], const int *want, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const int got = dial_edge(d, ab[i][0], ab[i][1]);
        if (got != want[i]) {
            fprintf(stderr, "edge %zu (A=%d B=%d): %d, want %d\n", i, ab[i][0], ab[i][1], got, want[i]);
            assert(0);
        }
    }
}

static void test_dial(void)
{
    // Two cycles A leading (00 10 11 01), then two B leading: two counts per cycle, each way.
    const int ab[16][2] = {{1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0},
                           {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}};
    const int want[16] = {0, 1, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1};
    dial_t d = {0};
    edges(&d, ab, want, 16);

    // At rest both pins read high (pull-ups): the first cycle settles the state, then the same.
    const int ab11[16][2] = {{0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1},
                             {1, 0}, {0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}, {0, 1}, {1, 1}};
    const int want11[16] = {0, 0, 0, 1, 0, 1, 0, 1, 0, -1, 0, -1, 0, -1, 0, -1};
    dial_t d11 = {0};
    edges(&d11, ab11, want11, 16);

    // The counts of one tick make one step of 5 in their direction, within 0..100.
    assert(dial_volume(50, 1) == 55);
    assert(dial_volume(50, 4) == 55);
    assert(dial_volume(50, -3) == 45);
    assert(dial_volume(50, 0) == 50);
    assert(dial_volume(98, 2) == 100 && dial_volume(100, 1) == 100);
    assert(dial_volume(3, -1) == 0 && dial_volume(0, -2) == 0);
}

static void test_button(void)
{
    // A press gives the hold once, when it has lasted FORGET_HOLD_MS, and nothing at its release.
    button_t b = {0};
    assert(button_step(&b, false, 0) == BUTTON_NONE);
    assert(button_step(&b, true, 1000) == BUTTON_NONE);
    assert(button_step(&b, true, 1000 + FORGET_HOLD_MS - 1) == BUTTON_NONE);
    assert(button_step(&b, true, 1000 + FORGET_HOLD_MS) == BUTTON_HOLD);
    assert(button_step(&b, true, 1000 + 2 * FORGET_HOLD_MS) == BUTTON_NONE);
    assert(button_step(&b, false, 30000) == BUTTON_NONE);

    // A short press, once, at its release.
    assert(button_step(&b, true, 30050) == BUTTON_NONE);
    assert(button_step(&b, true, 30100) == BUTTON_NONE);
    assert(button_step(&b, false, 30150) == BUTTON_SHORT);
    assert(button_step(&b, false, 30200) == BUTTON_NONE);

    // Across the hold boundary: last seen down a millisecond short of it, a short press; seen down
    // at it, the hold. The release starts the count again.
    assert(button_step(&b, true, 40000) == BUTTON_NONE);
    assert(button_step(&b, true, 40000 + FORGET_HOLD_MS - 1) == BUTTON_NONE);
    assert(button_step(&b, false, 40000 + FORGET_HOLD_MS) == BUTTON_SHORT);
    assert(button_step(&b, true, 60000) == BUTTON_NONE);
    assert(button_step(&b, true, 60000 + FORGET_HOLD_MS - 1) == BUTTON_NONE);
    assert(button_step(&b, true, 60000 + FORGET_HOLD_MS) == BUTTON_HOLD);
    assert(button_step(&b, false, 60000 + FORGET_HOLD_MS + 50) == BUTTON_NONE);

    // Across the wrap of the millisecond clock.
    b = (button_t){0};
    assert(button_step(&b, true, UINT32_MAX - 100) == BUTTON_NONE);
    assert(button_step(&b, true, FORGET_HOLD_MS - 102) == BUTTON_NONE);
    assert(button_step(&b, true, FORGET_HOLD_MS - 101) == BUTTON_HOLD);
}

int main(void)
{
    test_priority();
    test_error_hold();
    test_fill();
    test_dial();
    test_button();
    printf("ui_core: all tests passed\n");
    return 0;
}
