// The two TFLite Micro models, embedded 16-byte aligned (IDF's EMBED_FILES emits no alignment,
// and the flatbuffer is read in place). .incbin finds them through -Wa,-I in CMakeLists.txt.
    .section .rodata.wake_word_models, "a"

    .balign 16
    .global wake_word_zakhar_model
wake_word_zakhar_model:
    .incbin "zakhar.tflite"

    .balign 16
    .global wake_word_vad_model
wake_word_vad_model:
    .incbin "vad.tflite"
