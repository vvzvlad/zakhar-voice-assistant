# audio_core.c and ui_core.c are the pure parts (host test: make fw-voicepe-host-test); led_strip
# and network_provisioning come in through idf_component.yml.
idf_component_register(
    SRCS "app_main.c" "aic3204.c" "audio.c" "audio_core.c" "controls.c" "leds.c" "net_log.c"
         "ota_http.c" "ui_core.c" "volume.c" "xmos.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES zakhar_link wake_word esp_driver_i2s esp_driver_i2c esp_driver_gpio esp_wifi
                  esp_netif esp_event esp_timer esp_ringbuf lwip nvs_flash app_update
                  esp_http_server esp_app_format
)
