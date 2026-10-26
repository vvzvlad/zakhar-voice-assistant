#include "volume.h"

#include <math.h>

#define VOLUME_MIN_DB -39.0  // at 1 %

uint32_t volume_gain(int base, int boost)
{
    if (base <= 0) return 0;
    const int v = base + boost;
    const int reply = v < 1 ? 1 : v > 100 ? 100 : v;
    const double db = VOLUME_MIN_DB * (100 - reply) / 99.0;
    return (uint32_t)lround(65536.0 * pow(10.0, db / 20.0));
}
