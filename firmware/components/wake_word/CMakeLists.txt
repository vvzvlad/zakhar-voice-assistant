# The micro_wake_word engine (wake_word.cpp). It embeds no model: the wake, stop and VAD models
# arrive from the server at run time (wake_word_load). The TFLite Micro, esp-nn and frontend libraries
# come in through idf_component.yml.
idf_component_register(
    SRCS "wake_word.cpp"
    INCLUDE_DIRS "include"
)
