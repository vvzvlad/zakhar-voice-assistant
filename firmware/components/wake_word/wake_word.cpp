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
static const int WINDOW = 5;     // sliding_window_size, both manifests
static const int MIN_SLICES_BEFORE_DETECTION = 100;
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

// The cutoffs are the device defaults until the server's `config`: wake 80 % and VAD 30 %, as
// trunc(percent / 100 * 255).
static Model s_wake(204);
static Model s_vad(76);

static FrontendState s_fs;
// Held by wake_word_feed() while it runs the models and by wake_word_load() while it swaps one.
static SemaphoreHandle_t s_lock;
static std::atomic<uint8_t> s_peak{0};  // the highest wake probability since wake_word_take_peak()
static std::atomic<void (*)(void)> s_detect_cb{nullptr};

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
  s_lock = xSemaphoreCreateMutex();
  if (s_lock == nullptr || !FrontendPopulateState(&fc, &s_fs, SAMPLE_RATE) || !model_init(s_wake) ||
      !model_init(s_vad)) {
    ESP_LOGE(TAG, "frontend/arena init failed");
    return false;
  }
  ESP_LOGI(TAG, "ready, no model loaded");
  return true;
}

extern "C" bool wake_word_load(wake_word_model_t kind, uint8_t *data, size_t size) {
  Model &m = kind == WAKE_WORD_MODEL_WAKE ? s_wake : s_vad;
  bool ok;
  size_t used = 0;
  {
    Lock lock;
    ok = model_load(m, data);
    if (ok) used = m.interp->arena_used_bytes();
  }
  if (ok) {
    ESP_LOGI(TAG, "%s model of %u B loaded, tensor arena %u of %u B used", kind == WAKE_WORD_MODEL_WAKE ? "wake" : "VAD",
             (unsigned) size, (unsigned) used, (unsigned) TENSOR_ARENA_SIZE);
  }
  return ok;
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
    Lock lock;
    const bool wake = s_wake.interp != nullptr, vad = s_vad.interp != nullptr;
    if ((wake && !model_feed(s_wake, features)) || (vad && !model_feed(s_vad, features))) {
      ESP_LOGW(TAG, "invoke failed");
      continue;
    }
    if (!wake || !vad) continue;
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
