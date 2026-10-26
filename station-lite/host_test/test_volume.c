// Host test of the speaker volume curve (station-lite/main/volume.c). `make station-host-test`.
#include <assert.h>
#include <stdio.h>

#include "volume.h"

int main(void)
{
    // -39 dB at 1 %, 0 dB at 100 %, linear in dB between.
    assert(volume_gain(1, 0) == 735);
    assert(volume_gain(50, 0) == 6786);
    assert(volume_gain(99, 0) == 62630);
    assert(volume_gain(100, 0) == 65536);
    for (int v = 2; v <= 100; v++) assert(volume_gain(v, 0) > volume_gain(v - 1, 0));
    // 0 mutes, whatever the boost.
    assert(volume_gain(0, 0) == 0);
    assert(volume_gain(0, 50) == 0);
    // The reply plays at base + boost, kept within 1..100.
    assert(volume_gain(40, 10) == volume_gain(50, 0));
    assert(volume_gain(60, -10) == volume_gain(50, 0));
    assert(volume_gain(50, 60) == 65536);
    assert(volume_gain(50, -60) == 735);
    puts("volume: ok");
    return 0;
}
