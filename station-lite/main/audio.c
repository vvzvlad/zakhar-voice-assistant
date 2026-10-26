// The capture thread reads 10 ms of the microphones (hw:0,1, 8 channels) and of the speaker's own
// signal (hw:0,0 capture, 2 channels) in lockstep: the two streams are linked, so they start
// together and recover from an overrun together. mic_chain.h turns the two into the tap's two 16
// kHz channels: 1 the cleaned output, 0 the same after AGC2, as the Voice PE's XMOS gives its
// noise-suppressed and its AGC channel. Both go to the tap, channel 1 also to the wake-word engine.
// The speaker thread plays zakhar's playback (mono, 48 kHz) at the volume's gain on both channels of
// hw:0,0, and silence while there is none.
// tinyalsa opens, sets up, links and prepares the streams; the frames go through the READI and
// WRITEI ioctls here, not pcm_readi()/pcm_writei(). Before each transfer those read the stream's
// state with SNDRV_PCM_IOCTL_SYNC_PTR without its APPL flag, which hands the kernel their own stale
// copy of the application pointer (a 32-bit process cannot mmap the status page of a 64-bit
// kernel, so tinyalsa keeps that copy): the reads came back at once, 300 of 10 ms in 0.05 s.
#include "audio.h"

#include <errno.h>
#include <pthread.h>
#include <sound/asound.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>

#include "apm.h"
#include "audio_core.h"
#include "esp_log.h"
#include "mic_chain.h"
#include "tinyalsa/pcm.h"
#include "wake_word.h"
#include "zakhar_link.h"

static const char *TAG = "audio";

#define CARD 0
#define DEV_SPEAKER 0  // the playback, and the capture of what it plays
#define DEV_MICS 1
#define RATE 48000
#define FRAMES APM_IN_FRAMES  // the period: 10 ms
// The microphones take at most 2048 frames of buffer at 8 channels: 4 periods, for every stream.
#define PERIODS 4
#define MIC_RATE 16000
// The tap as the Voice PE sizes it (firmware/main/audio.c): 8.192 s, readers 1 s away from the end
// being overwritten; what is left holds the 2000 ms pre-roll plus zakhar_link's 3000 ms send timeout.
#define TAP_FRAMES (1u << 17)
#define TAP_MARGIN MIC_RATE
_Static_assert(TAP_FRAMES - TAP_MARGIN >= (2000 + 3000) * MIC_RATE / 1000,
               "the tap must hold the pre-roll plus zakhar_link's send timeout");
// The capture thread runs WebRTC's processing, written for desktop thread stacks, and the TFLite
// inference: well above musl's default 128 KB.
#define CAPTURE_STACK (1024 * 1024)

static tap_t s_tap;
static int16_t s_tap_buf[TAP_FRAMES * 2];
static struct pcm *s_mics, *s_ref, *s_spk;
static _Atomic uint32_t s_gain = 65536;

// A period of `channels` frames in or out (SNDRV_PCM_IOCTL_READI_FRAMES or WRITEI_FRAMES). An xrun
// prepares the stream (both linked ones) and goes on, as tinyalsa's own transfers do; any other
// failure ends the process: the device is no use without the stream.
static void transfer(struct pcm *pcm, unsigned long request, int16_t *buf, unsigned channels, const char *what)
{
    for (unsigned done = 0; done < FRAMES;) {
        struct snd_xferi x = {.buf = buf + done * channels, .frames = FRAMES - done};
        if (ioctl(pcm_get_file_descriptor(pcm), request, &x) == 0) {
            done += (unsigned)x.result;
        } else if (errno == EPIPE) {
            ESP_LOGW(TAG, "%s: xrun", what);
            if (pcm_prepare(pcm) != 0) {
                ESP_LOGE(TAG, "%s: prepare after the xrun failed: %s; exiting", what, pcm_get_error(pcm));
                exit(1);
            }
        } else if (errno != EINTR) {
            ESP_LOGE(TAG, "%s: %s; exiting", what, strerror(errno));
            exit(1);
        }
    }
}

static void *capture_main(void *arg)
{
    (void)arg;
    static int16_t mics[FRAMES * MIC_CHANNELS], spk[FRAMES * 2], frames[APM_OUT_FRAMES * 2], wake[APM_OUT_FRAMES];
    ESP_LOGI(TAG, "capture thread running: microphones hw:%d,%d, speaker reference hw:%d,%d", CARD, DEV_MICS, CARD,
             DEV_SPEAKER);
    for (;;) {
        transfer(s_mics, SNDRV_PCM_IOCTL_READI_FRAMES, mics, MIC_CHANNELS, "microphones");
        transfer(s_ref, SNDRV_PCM_IOCTL_READI_FRAMES, spk, 2, "speaker reference");
        if (!mic_chain_process(mics, spk, frames, wake)) exit(1);
        tap_write(&s_tap, frames, APM_OUT_FRAMES);
        wake_word_feed(wake, APM_OUT_FRAMES);
    }
    return NULL;
}

static void *playback_main(void *arg)
{
    (void)arg;
    static int16_t mono[FRAMES], stereo[FRAMES * 2];
    ESP_LOGI(TAG, "playback thread running: speaker hw:%d,%d", CARD, DEV_SPEAKER);
    for (;;) {
        const size_t n = zl_play_read(mono, FRAMES);
        memset(mono + n, 0, (FRAMES - n) * sizeof mono[0]);
        gain_block(mono, n, atomic_load_explicit(&s_gain, memory_order_relaxed));
        for (int i = 0; i < FRAMES; i++) stereo[2 * i] = stereo[2 * i + 1] = mono[i];
        transfer(s_spk, SNDRV_PCM_IOCTL_WRITEI_FRAMES, stereo, 2, "speaker");
    }
    return NULL;
}

static struct pcm *open_pcm(unsigned device, unsigned flags, unsigned channels, const char *what)
{
    const struct pcm_config cfg = {
        .channels = channels,
        .rate = RATE,
        .period_size = FRAMES,
        .period_count = PERIODS,
        .format = PCM_FORMAT_S16_LE,
        // An overrun or underrun as soon as the buffer is full or empty: tinyalsa's default for a
        // capture is ten buffers, and a 4.9 kernel then lets a read that fell behind by up to that
        // much copy the ring's stale periods again instead of failing with EPIPE.
        .stop_threshold = PERIODS * FRAMES,
    };
    struct pcm *pcm = pcm_open(CARD, device, flags, &cfg);
    if (!pcm_is_ready(pcm)) {
        ESP_LOGE(TAG, "%s (hw:%d,%u): %s", what, CARD, device, pcm_get_error(pcm));
        pcm_close(pcm);
        return NULL;
    }
    return pcm;
}

static bool start_thread(void *(*fn)(void *), size_t stack, const char *what)
{
    pthread_attr_t a;
    pthread_attr_init(&a);
    if (stack > 0) pthread_attr_setstacksize(&a, stack);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_t t;
    const int err = pthread_create(&t, &a, fn, NULL);
    pthread_attr_destroy(&a);
    if (err != 0) ESP_LOGE(TAG, "%s thread: %s", what, strerror(err));
    return err == 0;
}

bool audio_start_capture(void)
{
    tap_init(&s_tap, s_tap_buf, TAP_FRAMES, TAP_MARGIN, 2);
    if (!apm_init(true)) return false;
    s_mics = open_pcm(DEV_MICS, PCM_IN, MIC_CHANNELS, "microphones");
    s_ref = open_pcm(DEV_SPEAKER, PCM_IN, 2, "speaker reference");
    if (s_mics == NULL || s_ref == NULL) return false;
    // Linked first: the prepare and the start of one are then the pair's.
    if (pcm_link(s_mics, s_ref) != 0 || pcm_prepare(s_mics) != 0) {
        ESP_LOGE(TAG, "microphones and speaker reference not linked: %s", pcm_get_error(s_mics));
        return false;
    }
    return start_thread(capture_main, CAPTURE_STACK, "capture");
}

bool audio_start_playback(void)
{
    s_spk = open_pcm(DEV_SPEAKER, PCM_OUT, 2, "speaker");
    if (s_spk == NULL) return false;
    if (pcm_prepare(s_spk) != 0) {
        ESP_LOGE(TAG, "speaker not prepared: %s", pcm_get_error(s_spk));
        return false;
    }
    return start_thread(playback_main, 0, "playback");
}

void audio_set_gain(uint32_t gain)
{
    atomic_store_explicit(&s_gain, gain, memory_order_relaxed);
}

uint32_t audio_mic_position(void)
{
    return tap_position(&s_tap);
}

size_t audio_mic_read(uint32_t *pos, int16_t *out, size_t max_frames)
{
    return tap_read(&s_tap, pos, out, max_frames);
}
