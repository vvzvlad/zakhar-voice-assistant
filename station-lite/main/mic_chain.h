// The board's microphone processing for one 10 ms period, shared by the capture thread (audio.c)
// and zakhar-replay (station-lite/tools/replay.c), which runs it over a recording: the four
// microphones' average and the speaker's own signal through the audio processing (apm.h), out as
// the device's tap gives it, channel 1 also alone for the wake-word engine.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "apm.h"

// hw:0,1: the four microphones on channels 0, 2, 4 and 6, each repeated on the next one.
#define MIC_CHANNELS 8

// `mics` is APM_IN_FRAMES frames of MIC_CHANNELS, `spk` APM_IN_FRAMES frames of the speaker's two
// channels, both 48 kHz. `frames` gets APM_OUT_FRAMES frames of two 16 kHz channels: 0 the AGC
// output, 1 the cleaned output at the wake gain; `wake` gets channel 1 alone. After apm_init(), from
// one thread only. False when the processing refused the period (logged).
bool mic_chain_process(const int16_t *mics, const int16_t *spk, int16_t *frames, int16_t *wake);
