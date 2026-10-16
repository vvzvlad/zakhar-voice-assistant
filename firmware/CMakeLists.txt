# The Home Assistant Voice PE firmware: plain ESP-IDF, speaking docs/device-protocol.md through
# components/zakhar_link. Built by `make fw-build`, which passes ZAKHAR_GIT_VERSION (git describe).
cmake_minimum_required(VERSION 3.16)

# sdkconfig.secrets (gitignored) carries a Wi-Fi network and goes only into the one-time seed image:
# `make fw-build-seed` passes ZAKHAR_WIFI_SEED=1 and the file is applied over sdkconfig.defaults.
# Every other image has no network compiled in and joins over BLE provisioning (main/app_main.c).
set(SDKCONFIG_DEFAULTS "sdkconfig.defaults")
if(ZAKHAR_WIFI_SEED)
    list(APPEND SDKCONFIG_DEFAULTS "sdkconfig.secrets")
endif()

# network_provisioning (main/idf_component.yml) replaces IDF 5.x's own wifi_provisioning.
set(EXCLUDE_COMPONENTS wifi_provisioning)

# hello.fw is "voicepe <PROJECT_VER>".
set(PROJECT_VER "0.1.0 (${ZAKHAR_GIT_VERSION})")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(voicepe)
