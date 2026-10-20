#include "ui_core.h"

void ind_set_phase(ind_state_t *s, led_phase_t phase, uint32_t now_ms)
{
    if (phase == LED_PHASE_ERROR) s->error_at = now_ms;
    s->phase = phase;
}

ind_t ind_choose(ind_state_t *s, uint32_t now_ms)
{
    if (s->ota) return IND_OTA;
    if (s->provisioning) return IND_PROVISIONING;
    if (!s->network || !s->link) return IND_NO_SERVER;
    if (s->capture) return IND_CAPTURE;
    if (s->phase == LED_PHASE_ERROR && now_ms - s->error_at >= IND_ERROR_MS) s->phase = LED_PHASE_IDLE;
    switch (s->phase) {
    case LED_PHASE_LISTENING:
        return IND_LISTENING;
    case LED_PHASE_THINKING:
        return IND_THINKING;
    case LED_PHASE_REPLYING:
        return IND_REPLYING;
    case LED_PHASE_ERROR:
        return IND_ERROR;
    case LED_PHASE_IDLE:
        break;
    }
    if (s->playing) return IND_REPLYING;
    if (s->muted || s->volume == 0) return IND_MUTED;
    return IND_OFF;
}

int ind_fill(int percent, int leds)
{
    return (percent * leds + 99) / 100;
}

// ESPHome's rotary_encoder state table (2026.5.3 rotary_encoder.cpp, after QEIx4): index
// = state (S0-S3 and the direction) | A | B << 1; each entry is the next state with the counter
// increment or decrement it makes at resolutions 1, 2 and 4.
#define STATE_MASK 0x1C
#define S0 0x00
#define S1 0x04
#define S2 0x08
#define S3 0x0C
#define CCW 0x00
#define CW 0x10
#define INC_1 0x0100
#define INC_2 0x0300
#define INC_4 0x0700
#define DEC_1 0x1000
#define DEC_2 0x3000
#define DEC_4 0x7000
#define RESOLUTION_2 0x2200  // ROTARY_ENCODER_2_PULSES_PER_CYCLE

static const uint16_t LUT[32] = {
    CCW | S0,         CW | S1 | INC_1,  CCW | S0,         CCW | S3 | DEC_4,  // S0, CCW
    CCW | S1,         CCW | S1,         CCW | S0 | DEC_1, CW | S2 | INC_4,   // S1, CCW
    CCW | S1 | DEC_2, CCW | S2,         CW | S3 | INC_1,  CCW | S2,          // S2, CCW
    CW | S0 | INC_2,  CCW | S2 | DEC_1, CCW | S3,         CCW | S3,          // S3, CCW
    CW | S0,          CW | S1 | INC_1,  CW | S0,          CCW | S3 | DEC_4,  // S0, CW
    CW | S1,          CW | S1,          CCW | S0 | DEC_1, CW | S2 | INC_4,   // S1, CW
    CCW | S1 | DEC_2, CW | S2,          CW | S3 | INC_1,  CW | S2,           // S2, CW
    CW | S0 | INC_2,  CCW | S2 | DEC_1, CW | S3,          CW | S3,           // S3, CW
};

int dial_edge(dial_t *d, bool a, bool b)
{
    const uint16_t next = LUT[(d->state & STATE_MASK) | (a ? 1 : 0) | (b ? 2 : 0)];
    int counts = 0;
    if (next & RESOLUTION_2 & INC_4) counts = 1;
    if (next & RESOLUTION_2 & DEC_4) counts = -1;
    const bool first = !d->started;
    d->started = true;
    d->state = next;
    return first ? 0 : counts;
}

int dial_volume(int volume, int counts)
{
    const int v = volume + (counts > 0 ? DIAL_STEP : counts < 0 ? -DIAL_STEP : 0);
    return v < 0 ? 0 : v > 100 ? 100 : v;
}

bool hold_step(hold_t *h, bool pressed, uint32_t now_ms)
{
    if (!pressed) {
        h->down = false;
        return false;
    }
    if (!h->down) {
        h->down = true;
        h->fired = false;
        h->since = now_ms;
    }
    if (h->fired || now_ms - h->since < FORGET_HOLD_MS) return false;
    h->fired = true;
    return true;
}
