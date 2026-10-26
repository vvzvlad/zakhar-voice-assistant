// The microphone processing, two WebRTC audio processing modules (webrtc-audio-processing 2.1) on
// float samples in [-1, 1]. The first takes the microphones' average at 48 kHz with the speaker's
// signal at 48 kHz as the echo reference: high-pass filter, echo cancellation (AEC3), noise
// suppression, out at 16 kHz. The second runs AGC2's adaptive digital gain on a copy of that.
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APM_IN_FRAMES 480   // 10 ms at 48 kHz
#define APM_OUT_FRAMES 160  // 10 ms at 16 kHz

bool apm_init(void);
// One 10 ms frame: `ref` is the speaker's two channels (APM_IN_FRAMES each), `mic` the microphones;
// `clean` gets the first module's output, `agc` the second's. False when a module refused a frame
// (logged).
bool apm_process(const float *const ref[2], const float *mic, float *clean, float *agc);

#ifdef __cplusplus
}
#endif
