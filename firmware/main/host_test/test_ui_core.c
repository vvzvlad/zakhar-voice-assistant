// Host test of the Voice PE firmware's pure indication and dial decisions (ui_core.c): the stock
// control_leds priority, the error hold, and the stock dial decoder with one volume step per tick.
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
    assert(ind_choose(&s, 0) == IND_NO_SERVER);  // no Wi-Fi beats everything
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
    s.phase = LED_PHASE_WAITING;
    assert(ind_choose(&s, 0) == IND_OFF);        // nothing from the wake to the verdict
    s.phase = LED_PHASE_IDLE;
    assert(ind_choose(&s, 0) == IND_REPLYING);   // an announcement playing, over mute
    s.playing = false;
    assert(ind_choose(&s, 0) == IND_MUTED);
    s.muted = false;
    assert(ind_choose(&s, 0) == IND_OFF);
    s.volume = 0;
    assert(ind_choose(&s, 0) == IND_MUTED);      // volume 0 shows as silent
}

static void test_error_hold(void)
{
    ind_state_t s = online();
    ind_set_phase(&s, LED_PHASE_ERROR, 1000);
    assert(ind_choose(&s, 1000) == IND_ERROR);
    assert(ind_choose(&s, 1999) == IND_ERROR);
    assert(ind_choose(&s, 2000) == IND_OFF && s.phase == LED_PHASE_IDLE);

    // A new wake within the second replaces the error, and the hold does not touch it.
    s = online();
    ind_set_phase(&s, LED_PHASE_ERROR, 0);
    assert(ind_choose(&s, 400) == IND_ERROR);
    ind_set_phase(&s, LED_PHASE_WAITING, 500);
    assert(ind_choose(&s, 1500) == IND_OFF && s.phase == LED_PHASE_WAITING);
    ind_set_phase(&s, LED_PHASE_LISTENING, 1600);
    assert(ind_choose(&s, 1600) == IND_LISTENING);

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

int main(void)
{
    test_priority();
    test_error_hold();
    test_dial();
    printf("ui_core: all tests passed\n");
    return 0;
}
