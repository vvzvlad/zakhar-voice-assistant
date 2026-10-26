// The speaker volume curve: percent 1..100 is -39..0 dB, linear in dB; 0 mutes. Pure: host-tested
// (make station-host-test).
#pragma once

#include <stdint.h>

// The Q16 gain (65536 = unity) of zakhar's playback at the volume `base` percent and the server's
// `volume_boost` in points: the reply plays at base + boost within 1..100 (1, not 0: a boost must
// not silence a reply), and a base of 0 mutes it.
uint32_t volume_gain(int base, int boost);
