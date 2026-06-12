#pragma once

#ifdef USE_ESP32

#include "preprocessor_settings.h"

#include "esphome/core/preferences.h"

#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

// ZAKHAR FORK (additive): <atomic> is needed for the gated Wake-Probability peak
// tracker (stream_peak_) below — it is written from the inference FreeRTOS task and
// read from the main loop, so it must be a std::atomic (mirrors the existing
// cross-task vad_state_ pattern in micro_wake_word.h). VERIFY ON REAL HARDWARE.
#include <atomic>

namespace esphome::micro_wake_word {

static const uint8_t MIN_SLICES_BEFORE_DETECTION = 100;
static const uint32_t STREAMING_MODEL_VARIABLE_ARENA_SIZE = 1024;

struct DetectionEvent {
  std::string *wake_word;
  bool detected;
  bool partially_detection;  // Set if the most recent probability exceed the threshold, but the sliding window average
                             // hasn't yet
  uint8_t max_probability;
  uint8_t average_probability;
  bool blocked_by_vad = false;
};

class StreamingModel {
 public:
  virtual void log_model_config() = 0;
  virtual DetectionEvent determine_detected() = 0;

  // Performs inference on the given features.
  //  - If the model is enabled but not loaded, it will load it
  //  - If the model is disabled but loaded, it will unload it
  // Returns true if sucessful or false if there is an error
  bool perform_streaming_inference(const int8_t features[PREPROCESSOR_FEATURE_SIZE]);

  /// @brief Sets all recent_streaming_probabilities to 0 and resets the ignore window count
  void reset_probabilities();

  /// @brief Destroys the TFLite interpreter and frees the tensor and variable arenas' memory
  void unload_model();

  /// @brief Enable the model. The next performing_streaming_inference call will load it.
  virtual void enable() { this->enabled_ = true; }

  /// @brief Disable the model. The next performing_streaming_inference call will unload it.
  virtual void disable() { this->enabled_ = false; }

  /// @brief Return true if the model is enabled.
  bool is_enabled() const { return this->enabled_; }

  bool get_unprocessed_probability_status() const { return this->unprocessed_probability_status_; }

  // Quantized probability cutoffs mapping 0.0 - 1.0 to 0 - 255
  uint8_t get_default_probability_cutoff() const { return this->default_probability_cutoff_; }
  uint8_t get_probability_cutoff() const { return this->probability_cutoff_; }
  void set_probability_cutoff(uint8_t probability_cutoff) { this->probability_cutoff_ = probability_cutoff; }

  // ZAKHAR FORK (additive, non-destructive): a gated peak-probability tap.
  // The stock component never exposes the raw per-inference probability publicly, so
  // we keep a running MAX of it (0..255) since the last main-loop read, gated by a
  // switch so it costs nothing when nobody is watching. The server flips the gate on
  // when the device modal opens and off when it closes; while ON, the 1s panel sensor
  // reports the PEAK Wake Probability over that second. VERIFY ON REAL HARDWARE.
  /// @brief Enable/disable the gated peak tracker. Disabling also clears the peak so a
  /// stale value can't be read after the panel closes.
  void set_stream_enabled(bool enabled) {
    this->stream_enabled_ = enabled;
    if (!enabled) {
      this->stream_peak_.store(0, std::memory_order_relaxed);
    }
  }
  bool get_stream_enabled() const { return this->stream_enabled_; }
  /// @brief Read the running peak (0..255) and atomically reset it to 0 (main-loop side).
  uint8_t read_and_reset_stream_peak() { return this->stream_peak_.exchange(0); }

 protected:
  /// @brief Allocates tensor and variable arenas and sets up the model interpreter
  /// @return True if successful, false otherwise
  bool load_model_();
  /// @brief Probes the actual required tensor arena size by trial allocation.
  /// Tries the manifest size first, then 2x if that fails.
  /// @return The required arena size rounded up to 16-byte alignment, or 0 on failure.
  size_t probe_arena_size_();
  /// @brief Returns true if successfully registered the streaming model's TensorFlow operations
  bool register_streaming_ops_(tflite::MicroMutableOpResolver<20> &op_resolver);

  tflite::MicroMutableOpResolver<20> streaming_op_resolver_;

  bool loaded_{false};
  bool enabled_{true};
  bool tensor_arena_size_probed_{false};
  bool unprocessed_probability_status_{false};
  uint8_t current_stride_step_{0};
  int16_t ignore_windows_{-MIN_SLICES_BEFORE_DETECTION};

  uint8_t default_probability_cutoff_;
  uint8_t probability_cutoff_;
  size_t sliding_window_size_;

  size_t last_n_index_{0};
  size_t tensor_arena_size_;
  std::vector<uint8_t> recent_streaming_probabilities_;

  // ZAKHAR FORK (additive): gated Wake-Probability peak tracker. stream_peak_ holds the
  // running MAX raw probability (0..255) since the last main-loop read; stream_enabled_
  // gates the whole feature so it's free when nobody is watching. stream_peak_ is atomic
  // because perform_streaming_inference() writes it on the inference FreeRTOS task while
  // read_and_reset_stream_peak() reads it on the main loop. VERIFY ON REAL HARDWARE.
  std::atomic<uint8_t> stream_peak_{0};
  bool stream_enabled_{false};

  const uint8_t *model_start_;
  uint8_t *tensor_arena_{nullptr};
  uint8_t *var_arena_{nullptr};
  std::unique_ptr<tflite::MicroInterpreter> interpreter_;
  tflite::MicroResourceVariables *mrv_{nullptr};
  tflite::MicroAllocator *ma_{nullptr};
};

class WakeWordModel final : public StreamingModel {
 public:
  /// @brief Constructs a wake word model object
  /// @param id (std::string) identifier for this model
  /// @param model_start (const uint8_t *) pointer to the start of the model's TFLite FlatBuffer
  /// @param default_probability_cutoff (uint8_t) probability cutoff for acceping the wake word has been said
  /// @param sliding_window_average_size (size_t) the length of the sliding window computing the mean rolling
  ///                                    probability
  /// @param wake_word (std::string) Friendly name of the wake word
  /// @param tensor_arena_size (size_t) Size in bytes for allocating the tensor arena
  /// @param default_enabled (bool) If true, it will be enabled by default on first boot
  /// @param internal_only (bool) If true, the model will not be exposed to HomeAssistant as an available model
  WakeWordModel(const std::string &id, const uint8_t *model_start, uint8_t default_probability_cutoff,
                size_t sliding_window_average_size, const std::string &wake_word, size_t tensor_arena_size,
                bool default_enabled, bool internal_only);

  void log_model_config() override;

  /// @brief Checks for the wake word by comparing the mean probability in the sliding window with the probability
  /// cutoff
  /// @return True if wake word is detected, false otherwise
  DetectionEvent determine_detected() override;

  const std::string &get_id() const { return this->id_; }
  const std::string &get_wake_word() const { return this->wake_word_; }

  void add_trained_language(const std::string &language) { this->trained_languages_.push_back(language); }
  const std::vector<std::string> &get_trained_languages() const { return this->trained_languages_; }

  /// @brief Enable the model and save to flash. The next performing_streaming_inference call will load it.
  void enable() override;

  /// @brief Disable the model and save to flash. The next performing_streaming_inference call will unload it.
  void disable() override;

  bool get_internal_only() { return this->internal_only_; }

 protected:
  std::string id_;
  std::string wake_word_;
  std::vector<std::string> trained_languages_;

  bool internal_only_;

  ESPPreferenceObject pref_;
};

class VADModel final : public StreamingModel {
 public:
  VADModel(const uint8_t *model_start, uint8_t default_probability_cutoff, size_t sliding_window_size,
           size_t tensor_arena_size);

  void log_model_config() override;

  /// @brief Checks for voice activity by comparing the max probability in the sliding window with the probability
  /// cutoff
  /// @return True if voice activity is detected, false otherwise
  DetectionEvent determine_detected() override;
};

}  // namespace esphome::micro_wake_word

#endif
