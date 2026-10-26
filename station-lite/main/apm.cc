#include "apm.h"

#include "esp_log.h"
#include "modules/audio_processing/include/audio_processing.h"

static const char *TAG = "apm";

// Called from the capture thread only, as the modules require.
static rtc::scoped_refptr<webrtc::AudioProcessing> s_clean, s_agc;

extern "C" bool apm_init(bool noise_suppression)
{
    webrtc::AudioProcessing::Config c;
    c.high_pass_filter.enabled = true;
    c.echo_canceller.enabled = true;
    c.noise_suppression.enabled = noise_suppression;
    s_clean = webrtc::AudioProcessingBuilder().SetConfig(c).Create();
    webrtc::AudioProcessing::Config a;
    a.gain_controller2.enabled = true;
    a.gain_controller2.adaptive_digital.enabled = true;
    s_agc = webrtc::AudioProcessingBuilder().SetConfig(a).Create();
    if (!s_clean || !s_agc) {
        ESP_LOGE(TAG, "audio processing not created");
        return false;
    }
    return true;
}

extern "C" bool apm_process(const float *const ref[2], const float *mic, float *clean, float *agc)
{
    static const webrtc::StreamConfig REF(48000, 2), MIC(48000, 1), OUT(16000, 1);
    int err = s_clean->AnalyzeReverseStream(ref, REF);
    if (err == webrtc::AudioProcessing::kNoError) err = s_clean->ProcessStream(&mic, MIC, OUT, &clean);
    if (err != webrtc::AudioProcessing::kNoError) {
        ESP_LOGE(TAG, "echo cancellation and noise suppression failed: %d", err);
        return false;
    }
    err = s_agc->ProcessStream(&clean, OUT, OUT, &agc);
    if (err != webrtc::AudioProcessing::kNoError) {
        ESP_LOGE(TAG, "gain control failed: %d", err);
        return false;
    }
    return true;
}
