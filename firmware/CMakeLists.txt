# The Home Assistant Voice PE firmware: plain ESP-IDF, speaking docs/device-protocol.md through
# components/zakhar_link. Built by `make fw-build`, which passes ZAKHAR_GIT_VERSION (git describe).
cmake_minimum_required(VERSION 3.16)

# sdkconfig.secrets (gitignored) carries the Wi-Fi credentials and the device settings of one
# speaker; it is applied over sdkconfig.defaults when present.
set(SDKCONFIG_DEFAULTS "sdkconfig.defaults")
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/sdkconfig.secrets")
    list(APPEND SDKCONFIG_DEFAULTS "sdkconfig.secrets")
endif()

# hello.fw is "voicepe <PROJECT_VER>".
set(PROJECT_VER "0.1.0 (${ZAKHAR_GIT_VERSION})")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(voicepe)
