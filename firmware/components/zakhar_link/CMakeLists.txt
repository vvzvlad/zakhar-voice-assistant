# zakhar_link: the device side of docs/device-protocol.md. zl_core.c is the pure core (host
# test: make fw-host-test); zakhar_link.c the ESP transport. esp_websocket_client and mdns come in
# through idf_component.yml.
idf_component_register(
    SRCS "zakhar_link.c" "zl_core.c"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES json esp_timer esp_wifi esp_event esp_driver_tsens
)
