# The micro_wake_word engine (wake_word.cpp). The "Zakhar" model is the deployed one at the
# repository root, microWakeWord/v31/model/zakhar.tflite; models/vad.tflite is the stock VAD model
# (esphome/micro-wake-word-models models/v2/vad.tflite). The TFLite Micro, esp-nn and frontend
# libraries come in through idf_component.yml.
idf_component_register(
    SRCS "wake_word.cpp" "models.S"
    INCLUDE_DIRS "include"
)

idf_build_get_property(project_dir PROJECT_DIR)
set(zakhar_dir "${project_dir}/../microWakeWord/v31/model")
set(vad_dir "${CMAKE_CURRENT_SOURCE_DIR}/models")
# .incbin resolves the model files through the assembler's include path.
target_compile_options(${COMPONENT_LIB} PRIVATE
    "$<$<COMPILE_LANGUAGE:ASM>:-Wa,-I${zakhar_dir}>"
    "$<$<COMPILE_LANGUAGE:ASM>:-Wa,-I${vad_dir}>")
set_property(SOURCE models.S APPEND PROPERTY OBJECT_DEPENDS
             "${zakhar_dir}/zakhar.tflite" "${vad_dir}/vad.tflite")
