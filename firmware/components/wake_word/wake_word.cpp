// The micro_wake_word engine, ported out of the micro_wake_word component of the
// zakhar-voice-assistant ESPHome fork (esphome/zakhar-preroll) by way of the espanel panel's
// wake_word component: the same frontend settings (preprocessor_settings.h), feature scaling,
// streaming inference and detection rule (StreamingModel / WakeWordModel / VADModel in
// streaming_model.cpp, MicroWakeWord::process_probabilities_ in micro_wake_word.cpp). The audio
// comes from the caller (wake_word_feed), and so do the models (wake_word_load); the microphone
// and the model source are the board's business.
#include "wake_word.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "frontend.h"
#include "frontend_util.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "wake_word";

static const int SAMPLE_RATE = 16000;
static const int FEATURES = 40;  // PREPROCESSOR_FEATURE_SIZE
static const int WINDOW = 5;     // sliding_window_size, all three manifests
static const int MIN_SLICES_BEFORE_DETECTION = 100;
// The stop model runs from this many 10 ms slices of playback on: the stock firmware's
// activate_stop_word_once enables it 1 s after the answer starts.
static const int STOP_ARM_SLICES = 100;
static const size_t VAR_ARENA_SIZE = 1024;  // STREAMING_MODEL_VARIABLE_ARENA_SIZE
// Each model's tensor arena, allocated once and reused by every model loaded into it: well above
// the manifests' tensor_arena_size (Zakhar v31 48000, the stock VAD 16772), so a retrained model
// fits without a firmware change.
static const size_t TENSOR_ARENA_SIZE = 128 * 1024;

struct Model {
  explicit Model(uint8_t c) : cutoff(c) {}
  uint8_t *data = nullptr;   // the loaded model's flatbuffer, owned
  uint8_t *var = nullptr;    // VAR_ARENA_SIZE, for the resource variables
  uint8_t *arena = nullptr;  // TENSOR_ARENA_SIZE
  // int(probability_cutoff * 255), as the fork's __init__.py quantizes it; wake_word_set_cutoffs
  // writes it from another task.
  std::atomic<uint8_t> cutoff;
  tflite::MicroMutableOpResolver<20> resolver;
  tflite::MicroInterpreter *interp = nullptr;  // nullptr: no model loaded
  int stride = 0, step = 0, idx = 0, ignore = -MIN_SLICES_BEFORE_DETECTION;
  uint8_t probs[WINDOW] = {0};
  bool fresh = false;  // a probability arrived since the last detection check
};

// The wake and VAD cutoffs come from the server's `config` (wake_word_set_cutoffs); until the first
// one (s_cutoffs_set) neither the wake word nor the stop word, which the VAD gates too, is detected.
static Model s_wake(0);
static Model s_vad(0);
static std::atomic<bool> s_cutoffs_set{false};
// The stop model's cutoff is compiled in: stop.json's probability_cutoff 0.5, trunc(0.5 * 255).
static Model s_stop(127);

static FrontendState s_fs;
static FrontendState s_fs_stop;  // the stop model's own input, when wake_word_feed() gets one
// Held by wake_word_feed() while it runs the models and by wake_word_load() while it swaps one.
static SemaphoreHandle_t s_lock;
static std::atomic<uint8_t> s_peak{0};  // the highest wake probability since wake_word_take_peak()
static std::atomic<void (*)(void)> s_detect_cb{nullptr};
static std::atomic<bool (*)(void)> s_playing{nullptr};
static std::atomic<void (*)(void)> s_stop_cb{nullptr};
// Under s_lock: the consecutive slices fed while playing, whether the stop model runs, and the
// stop model's highest window sum since it was armed.
static int s_playing_slices = 0;
static bool s_stop_armed = false;
static uint32_t s_stop_peak = 0;

struct Lock {
  Lock() { xSemaphoreTake(s_lock, portMAX_DELAY); }
  ~Lock() { xSemaphoreGive(s_lock); }
};

// The op resolver and both arenas, once.
static bool model_init(Model &m) {
  auto &r = m.resolver;
  r.AddCallOnce(); r.AddVarHandle(); r.AddReshape(); r.AddReadVariable(); r.AddStridedSlice();
  r.AddConcatenation(); r.AddAssignVariable(); r.AddConv2D(); r.AddMul(); r.AddAdd(); r.AddMean();
  r.AddFullyConnected(); r.AddLogistic(); r.AddQuantize(); r.AddDepthwiseConv2D();
  r.AddAveragePool2D(); r.AddMaxPool2D(); r.AddPad(); r.AddPack(); r.AddSplitV();
  const uint32_t caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
  m.var = (uint8_t *) heap_caps_aligned_alloc(16, VAR_ARENA_SIZE, caps);
  m.arena = (uint8_t *) heap_caps_aligned_alloc(16, TENSOR_ARENA_SIZE, caps);
  return m.var != nullptr && m.arena != nullptr;
}

static void model_unload(Model &m) {
  delete m.interp;
  m.interp = nullptr;
  heap_caps_free(m.data);
  m.data = nullptr;
}

// Under s_lock. The model in `data` (owned from here on) replaces the current one, on the same
// arenas; false, with no model loaded, when it does not load.
static bool model_load(Model &m, uint8_t *data) {
  model_unload(m);
  m.data = data;
  auto *mrv = tflite::MicroResourceVariables::Create(tflite::MicroAllocator::Create(m.var, VAR_ARENA_SIZE), 20);
  m.interp = new tflite::MicroInterpreter(tflite::GetModel(m.data), m.resolver, m.arena, TENSOR_ARENA_SIZE, mrv);
  if (m.interp->AllocateTensors() != kTfLiteOk) {
    model_unload(m);
    return false;
  }
  m.stride = m.interp->input(0)->dims->data[1];
  m.step = 0;
  m.idx = 0;
  memset(m.probs, 0, sizeof(m.probs));
  m.fresh = false;
  m.ignore = -MIN_SLICES_BEFORE_DETECTION;
  return true;
}

// One feature slice in; an inference every `stride` slices (StreamingModel::perform_streaming_inference).
static bool model_feed(Model &m, const int8_t *features) {
  m.step %= m.stride;
  memcpy(m.interp->input(0)->data.int8 + FEATURES * m.step, features, FEATURES);
  if (++m.step >= m.stride) {
    if (m.interp->Invoke() != kTfLiteOk) return false;
    m.idx = (m.idx + 1) % WINDOW;
    m.probs[m.idx] = m.interp->output(0)->data.uint8[0];
    m.fresh = true;
    if (&m == &s_wake) {
      uint8_t peak = s_peak.load(std::memory_order_relaxed);
      while (m.probs[m.idx] > peak &&
             !s_peak.compare_exchange_weak(peak, m.probs[m.idx], std::memory_order_relaxed)) {
      }
    }
  }
  // The cool-off after a detection: only slices below the cutoff count it down.
  if (m.probs[m.idx] < m.cutoff.load(std::memory_order_relaxed)) m.ignore = std::min(m.ignore + 1, 0);
  return true;
}

static uint32_t window_sum(const Model &m) {
  uint32_t sum = 0;
  for (uint8_t p : m.probs) sum += p;
  return sum;
}

static bool model_above_cutoff(const Model &m) {
  return window_sum(m) > (uint32_t) m.cutoff.load(std::memory_order_relaxed) * WINDOW;
}

// The next feature slice of fs from *samples (advanced past what it used, *n counted down), as
// the models take it; false when the samples ran out before a slice completed.
static bool next_slice(FrontendState *fs, const int16_t **samples, size_t *n, int8_t *features) {
  while (*n > 0) {
    size_t used = 0;
    FrontendOutput out = FrontendProcessSamples(fs, *samples, *n, &used);
    *samples += used;
    *n -= used;
    if (out.size == 0) continue;
    for (size_t i = 0; i < out.size; i++) {
      int32_t v = ((out.values[i] * 256) + 333) / 666 - 128;  // 666 = 25.6 * 26.0, rounded
      features[i] = (int8_t) std::clamp<int32_t>(v, INT8_MIN, INT8_MAX);
    }
    return true;
  }
  return false;
}

// Under s_lock: one slice into the stop model while it is armed, and its detection by the same
// rule as "Zakhar" in wake_word_feed().
static void stop_feed(const int8_t *features) {
  if (!s_stop_armed || s_stop.interp == nullptr) return;
  if (!model_feed(s_stop, features)) {
    ESP_LOGW(TAG, "invoke failed");
    return;
  }
  s_stop_peak = std::max(s_stop_peak, window_sum(s_stop));
  if (s_vad.interp == nullptr || !s_cutoffs_set.load(std::memory_order_acquire) || !s_stop.fresh ||
      s_stop.ignore < 0) {
    return;
  }
  s_stop.fresh = false;
  if (!model_above_cutoff(s_stop)) return;
  if (!model_above_cutoff(s_vad)) {
    ESP_LOGI(TAG, "\"Stop\" blocked by VAD");
    return;
  }
  ESP_LOGI(TAG, "\"Stop\" detected");
  if (void (*cb)(void) = s_stop_cb.load(std::memory_order_acquire)) cb();
  memset(s_stop.probs, 0, sizeof(s_stop.probs));
  s_stop.ignore = -MIN_SLICES_BEFORE_DETECTION;
}

extern "C" bool wake_word_init(void) {
  FrontendConfig fc;
  fc.window.size_ms = 30;
  fc.window.step_size_ms = 10;
  fc.filterbank.num_channels = FEATURES;
  fc.filterbank.lower_band_limit = 125.0;
  fc.filterbank.upper_band_limit = 7500.0;
  fc.noise_reduction.smoothing_bits = 10;
  fc.noise_reduction.even_smoothing = 0.025;
  fc.noise_reduction.odd_smoothing = 0.06;
  fc.noise_reduction.min_signal_remaining = 0.05;
  fc.pcan_gain_control.enable_pcan = 1;
  fc.pcan_gain_control.strength = 0.95;
  fc.pcan_gain_control.offset = 80.0;
  fc.pcan_gain_control.gain_bits = 21;
  fc.log_scale.enable_log = 1;
  fc.log_scale.scale_shift = 6;
  s_lock = xSemaphoreCreateMutex();
  if (s_lock == nullptr || !FrontendPopulateState(&fc, &s_fs, SAMPLE_RATE) ||
      !FrontendPopulateState(&fc, &s_fs_stop, SAMPLE_RATE) || !model_init(s_wake) || !model_init(s_vad) ||
      !model_init(s_stop)) {
    ESP_LOGE(TAG, "frontend/arena init failed");
    return false;
  }
  ESP_LOGI(TAG, "ready, no model loaded");
  return true;
}

extern "C" bool wake_word_load(wake_word_model_t kind, uint8_t *data, size_t size) {
  Model &m = kind == WAKE_WORD_MODEL_WAKE ? s_wake : kind == WAKE_WORD_MODEL_VAD ? s_vad : s_stop;
  const char *name = kind == WAKE_WORD_MODEL_WAKE ? "wake" : kind == WAKE_WORD_MODEL_VAD ? "VAD" : "stop";
  bool ok;
  size_t used = 0;
  {
    Lock lock;
    ok = model_load(m, data);
    if (ok) used = m.interp->arena_used_bytes();
  }
  if (ok) {
    ESP_LOGI(TAG, "%s model of %u B loaded, tensor arena %u of %u B used", name, (unsigned) size, (unsigned) used,
             (unsigned) TENSOR_ARENA_SIZE);
  }
  return ok;
}

extern "C" void wake_word_feed(const int16_t *wake, const int16_t *stop, size_t n) {
  // Asked before s_lock is taken: it takes the playback owner's lock.
  bool (*playing_cb)(void) = s_playing.load(std::memory_order_acquire);
  const bool playing = playing_cb != nullptr && playing_cb();
  int8_t features[FEATURES];
  const int16_t *samples = wake;
  size_t left = n;
  while (next_slice(&s_fs, &samples, &left, features)) {
    Lock lock;
    if (!playing) {
      if (s_stop_armed) {
        ESP_LOGI(TAG, "\"Stop\" off with the playback: its highest window mean %.2f, cutoff %.2f",
                 s_stop_peak / (WINDOW * 255.0), s_stop.cutoff.load(std::memory_order_relaxed) / 255.0);
      }
      s_playing_slices = 0;
      s_stop_armed = false;
    } else if (!s_stop_armed && ++s_playing_slices >= STOP_ARM_SLICES) {
      // As ESPHome's enable_model (reset_probabilities): the stop model starts with an empty
      // window and the cool-off.
      s_stop_armed = true;
      memset(s_stop.probs, 0, sizeof(s_stop.probs));
      s_stop.idx = 0;
      s_stop.fresh = false;
      s_stop.ignore = -MIN_SLICES_BEFORE_DETECTION;
      s_stop_peak = 0;
    }
    const bool wake_on = s_wake.interp != nullptr, vad = s_vad.interp != nullptr;
    if ((wake_on && !model_feed(s_wake, features)) || (vad && !model_feed(s_vad, features))) {
      ESP_LOGW(TAG, "invoke failed");
      continue;
    }
    if (stop == nullptr) stop_feed(features);
    if (!wake_on || !vad || !s_cutoffs_set.load(std::memory_order_acquire)) continue;
    // As WakeWordModel::determine_detected: during the cool-off the probability stays unprocessed,
    // so the window is checked on the very slice the cool-off ends.
    if (!s_wake.fresh || s_wake.ignore < 0) continue;
    s_wake.fresh = false;
    if (!model_above_cutoff(s_wake)) continue;
    if (!model_above_cutoff(s_vad)) {
      ESP_LOGI(TAG, "\"Zakhar\" blocked by VAD");
      continue;
    }
    ESP_LOGI(TAG, "\"Zakhar\" detected");
    if (void (*cb)(void) = s_detect_cb.load(std::memory_order_acquire)) cb();
    memset(s_wake.probs, 0, sizeof(s_wake.probs));
    s_wake.ignore = -MIN_SLICES_BEFORE_DETECTION;
  }
  // The stop input's own frontend runs on every sample given, armed or not, so its noise
  // estimate is current when the model is armed.
  samples = stop;
  left = stop != nullptr ? n : 0;
  while (next_slice(&s_fs_stop, &samples, &left, features)) {
    Lock lock;
    stop_feed(features);
  }
}

extern "C" void wake_word_set_cutoffs(uint8_t wake, uint8_t vad) {
  s_wake.cutoff.store(wake, std::memory_order_relaxed);
  s_vad.cutoff.store(vad, std::memory_order_relaxed);
  s_cutoffs_set.store(true, std::memory_order_release);
}

extern "C" uint8_t wake_word_take_peak(void) { return s_peak.exchange(0, std::memory_order_relaxed); }

extern "C" void wake_word_set_detect_cb(void (*cb)(void)) { s_detect_cb.store(cb, std::memory_order_release); }

extern "C" void wake_word_set_stop(bool (*playing)(void), void (*cb)(void)) {
  s_playing.store(playing, std::memory_order_release);
  s_stop_cb.store(cb, std::memory_order_release);
}
