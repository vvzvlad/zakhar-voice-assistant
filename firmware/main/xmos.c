// The XMOS control protocol over I2C, as ESPHome's voice_kit component speaks it (the XMOS
// firmware's servicer.c): a write is [resid, cmd, len, payload...]; a read writes
// [resid, cmd | 0x80, len], then reads len bytes [status, data...], status 0 meaning done.
#include "xmos.h"

#include "board.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "xmos";

#define RESID_DFU 240
#define RESID_CONFIG 241
#define CMD_GETVERSION 88
#define CMD_READ 0x80
#define CMD_CH0_STAGE 0x30
#define CMD_CH1_STAGE 0x40
#define STAGE_NS 3
#define STAGE_AGC 4
#define STATUS_DONE 0

#define I2C_TIMEOUT_MS 100
#define POLL_MS 200  // GETVERSION retries while the XMOS boots

static bool read_version(i2c_master_dev_handle_t dev, uint8_t version[3])
{
    const uint8_t req[] = {RESID_DFU, CMD_GETVERSION | CMD_READ, 4};
    uint8_t resp[4];
    if (i2c_master_transmit(dev, req, sizeof req, I2C_TIMEOUT_MS) != ESP_OK) return false;
    if (i2c_master_receive(dev, resp, sizeof resp, I2C_TIMEOUT_MS) != ESP_OK) return false;
    if (resp[0] != STATUS_DONE) return false;
    version[0] = resp[1];
    version[1] = resp[2];
    version[2] = resp[3];
    return true;
}

static esp_err_t write_stage(i2c_master_dev_handle_t dev, uint8_t cmd, uint8_t stage)
{
    const uint8_t req[] = {RESID_CONFIG, cmd, 1, stage};
    const esp_err_t err = i2c_master_transmit(dev, req, sizeof req, I2C_TIMEOUT_MS);
    if (err != ESP_OK) ESP_LOGE(TAG, "stage write 0x%02X failed", cmd);
    return err;
}

esp_err_t xmos_boot(i2c_master_bus_handle_t bus)
{
    const gpio_config_t reset = {.pin_bit_mask = 1ULL << PIN_XMOS_RESET, .mode = GPIO_MODE_OUTPUT};
    esp_err_t err = gpio_config(&reset);
    if (err != ESP_OK) return err;
    gpio_set_level(PIN_XMOS_RESET, 1);
    esp_rom_delay_us(1000);
    gpio_set_level(PIN_XMOS_RESET, 0);

    const i2c_device_config_t dc = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = I2C_ADDR_XMOS,
        .scl_speed_hz = 400000,
    };
    i2c_master_dev_handle_t dev;
    err = i2c_master_bus_add_device(bus, &dc, &dev);
    if (err != ESP_OK) return err;

    ESP_LOGI(TAG, "waiting for the XMOS to boot");
    uint8_t v[3];
    while (!read_version(dev, v)) vTaskDelay(pdMS_TO_TICKS(POLL_MS));
    if (v[0] == 1 && v[1] == 3 && v[2] == 1) {
        ESP_LOGI(TAG, "firmware %u.%u.%u", v[0], v[1], v[2]);
    } else {
        ESP_LOGW(TAG, "firmware %u.%u.%u, not the 1.3.1 this firmware is made for", v[0], v[1], v[2]);
    }

    err = write_stage(dev, CMD_CH0_STAGE, STAGE_AGC);
    if (err == ESP_OK) err = write_stage(dev, CMD_CH1_STAGE, STAGE_NS);
    return err;
}
