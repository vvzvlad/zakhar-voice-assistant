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

# At most 24 compiles at once, however the build is started (make fw-build, a bare idf.py build):
# idf.py starts ninja with a job per CPU, and on cirunner's 72 CPUs the ~74 parallel cc1plus of
# esp-tflite-micro (~430 MB each) run its 19 GB out of memory. Set before project(), which creates
# the component targets that take the pool.
set_property(GLOBAL PROPERTY JOB_POOLS compile=24)
set(CMAKE_JOB_POOL_COMPILE compile)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(voicepe)
