// The micro_wake_word engine, ported out of the micro_wake_word component of the
// zakhar-voice-assistant ESPHome fork (esphome/zakhar-preroll) by way of the espanel panel's
// wake_word component: the same frontend settings (preprocessor_settings.h), feature scaling,
// streaming inference and detection rule (StreamingModel / WakeWordModel / VADModel in
// streaming_model.cpp, MicroWakeWord::process_probabilities_ in micro_wake_word.cpp). The audio
// comes from the caller (wake_word_feed); the microphone is the board's business.
#include "wake_word.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "frontend.h"
#include "frontend_util.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "wake_word";

// models.S
extern "C" const uint8_t wake_word_zakhar_model[];
extern "C" const uint8_t wake_word_vad_model[];

static const int SAMPLE_RATE = 16000;
static const int FEATURES = 40;  // PREPROCESSOR_FEATURE_SIZE
static const int WINDOW = 5;     // sliding_window_size, both manifests
static const int MIN_SLICES_BEFORE_DETECTION = 100;
static const size_t VAR_ARENA_SIZE = 1024;  // STREAMING_MODEL_VARIABLE_ARENA_SIZE

struct Model {
  Model(const uint8_t *d, size_t a, uint8_t c) : data(d), arena_size(a), cutoff(c) {}
  const uint8_t *data;
  size_t arena_size;  // tensor_arena_size from the manifest
  // int(probability_cutoff * 255), as the fork's __init__.py quantizes it; wake_word_set_cutoffs
  // writes it from another task.
  std::atomic<uint8_t> cutoff;
  tflite::MicroMutableOpResolver<20> resolver;
  tflite::MicroInterpreter *interp = nullptr;
  int stride = 0, step = 0, idx = 0, ignore = -MIN_SLICES_BEFORE_DETECTION;
  uint8_t probs[WINDOW] = {0};
  bool fresh = false;  // a probability arrived since the last detection check
};

// Zakhar v31 manifest: tensor_arena_size 48000 (v31 is v28 fine-tuned: the same graph plus one
// int32 bias constant; v28 ran in 45000). VAD (esphome/micro-wake-word-models
// models/v2/vad.json): 16772. The cutoffs are the device defaults until the server's `config`:
// wake 80 % and VAD 30 %, as trunc(percent / 100 * 255).
static Model s_wake(wake_word_zakhar_model, 48000, 204);
static Model s_vad(wake_word_vad_model, 16772, 76);

static FrontendState s_fs;
static std::atomic<uint8_t> s_peak{0};  // the highest wake probability since wake_word_take_peak()
static std::atomic<void (*)(void)> s_detect_cb{nullptr};

static bool model_load(Model &m) {
  auto &r = m.resolver;
  r.AddCallOnce(); r.AddVarHandle(); r.AddReshape(); r.AddReadVariable(); r.AddStridedSlice();
  r.AddConcatenation(); r.AddAssignVariable(); r.AddConv2D(); r.AddMul(); r.AddAdd(); r.AddMean();
  r.AddFullyConnected(); r.AddLogistic(); r.AddQuantize(); r.AddDepthwiseConv2D();
  r.AddAveragePool2D(); r.AddMaxPool2D(); r.AddPad(); r.AddPack(); r.AddSplitV();
  const uint32_t caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
  const size_t size = (m.arena_size + 15) & ~size_t(15);
  auto *var = (uint8_t *) heap_caps_aligned_alloc(16, VAR_ARENA_SIZE, caps);
  auto *arena = (uint8_t *) heap_caps_aligned_alloc(16, size, caps);
  if (var == nullptr || arena == nullptr) return false;
  auto *mrv = tflite::MicroResourceVariables::Create(tflite::MicroAllocator::Create(var, VAR_ARENA_SIZE), 20);
  m.interp = new tflite::MicroInterpreter(tflite::GetModel(m.data), m.resolver, arena, size, mrv);
  if (m.interp->AllocateTensors() != kTfLiteOk) return false;
  m.stride = m.interp->input(0)->dims->data[1];
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

static bool model_above_cutoff(const Model &m) {
  uint32_t sum = 0;
  for (uint8_t p : m.probs) sum += p;
  return sum > (uint32_t) m.cutoff.load(std::memory_order_relaxed) * WINDOW;
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
  if (!FrontendPopulateState(&fc, &s_fs, SAMPLE_RATE) || !model_load(s_wake) || !model_load(s_vad)) {
    ESP_LOGE(TAG, "frontend/model init failed");
    return false;
  }
  ESP_LOGI(TAG, "listening");
  return true;
}

extern "C" void wake_word_feed(const int16_t *samples, size_t n) {
  int8_t features[FEATURES];
  while (n > 0) {
    size_t used = 0;
    FrontendOutput out = FrontendProcessSamples(&s_fs, samples, n, &used);
    samples += used;
    n -= used;
    if (out.size == 0) continue;
    for (size_t i = 0; i < out.size; i++) {
      int32_t v = ((out.values[i] * 256) + 333) / 666 - 128;  // 666 = 25.6 * 26.0, rounded
      features[i] = (int8_t) std::clamp<int32_t>(v, INT8_MIN, INT8_MAX);
    }
    if (!model_feed(s_wake, features) || !model_feed(s_vad, features)) {
      ESP_LOGW(TAG, "invoke failed");
      continue;
    }
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
}

extern "C" void wake_word_set_cutoffs(uint8_t wake, uint8_t vad) {
  s_wake.cutoff.store(wake, std::memory_order_relaxed);
  s_vad.cutoff.store(vad, std::memory_order_relaxed);
}

extern "C" uint8_t wake_word_take_peak(void) { return s_peak.exchange(0, std::memory_order_relaxed); }

extern "C" void wake_word_set_detect_cb(void (*cb)(void)) { s_detect_cb.store(cb, std::memory_order_release); }
