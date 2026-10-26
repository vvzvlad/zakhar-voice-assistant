#include "mic_chain.h"

#include <math.h>

// Channel 1's gain, four times the Voice PE's 4 (firmware/main/audio.c): at 4 a quiet room's noise
// reaches the wake-word engine at a few LSB, and the model fires on it in bursts that only its VAD
// holds back.
#define WAKE_GAIN 16

static int16_t to_s16(float v)
{
    const float s = v * 32768.0f;
    return s >= 32767.0f ? 32767 : s <= -32768.0f ? -32768 : (int16_t)lrintf(s);
}

bool mic_chain_process(const int16_t *mics, const int16_t *spk, int16_t *frames, int16_t *wake)
{
    static float mic[APM_IN_FRAMES], ref_l[APM_IN_FRAMES], ref_r[APM_IN_FRAMES], clean[APM_OUT_FRAMES],
        agc[APM_OUT_FRAMES];
    static const float *const ref[2] = {ref_l, ref_r};
    for (int i = 0; i < APM_IN_FRAMES; i++) {
        const int16_t *m = &mics[i * MIC_CHANNELS];
        mic[i] = (float)(m[0] + m[2] + m[4] + m[6]) / (4.0f * 32768.0f);
        ref_l[i] = (float)spk[2 * i] / 32768.0f;
        ref_r[i] = (float)spk[2 * i + 1] / 32768.0f;
    }
    if (!apm_process(ref, mic, clean, agc)) return false;
    for (int i = 0; i < APM_OUT_FRAMES; i++) {
        wake[i] = to_s16(clean[i] * WAKE_GAIN);
        frames[2 * i] = to_s16(agc[i]);
        frames[2 * i + 1] = wake[i];
    }
    return true;
}
