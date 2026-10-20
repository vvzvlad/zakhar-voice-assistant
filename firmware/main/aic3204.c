// The register sequence of ESPHome's aic3204 component (2026.5.3, aic3204.cpp setup()), register
// meanings from TI SLAA557. Every write selects its page through register 0 first in the list.
#include "aic3204.h"

#include "audio_core.h"
#include "board.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "aic3204";

#define I2C_TIMEOUT_MS 100
#define REG_PAGE 0x00
#define REG_DAC_CH_SET2 0x40  // page 0: mute bits 2-3
#define REG_DACL_VOL 0x41     // page 0
#define REG_DACR_VOL 0x42     // page 0

typedef struct {
    uint8_t reg, val;
} regval_t;

static const regval_t RESET[] = {
    {REG_PAGE, 0x00},
    {0x01, 0x01},  // software reset
};

static const regval_t SETUP[] = {
    {0x0B, 0x82},  // NDAC powered, /2
    {0x0C, 0x82},  // MDAC powered, /2
    {0x0E, 0x80},  // DOSR 128: 24.576 MHz / (2 * 2 * 128) = 48 kHz
    {0x1B, 0x30},  // I2S, 32-bit words, BCLK and WCLK inputs
    {0x38, 0x02},  // SCLK/MFP3 pin: secondary data input
    {0x1F, 0x01},  // secondary data input from SCLK/MFP3
    {0x20, 0x01},  // the DAC takes the secondary data input
    {0x3C, 0x01},  // DAC processing block PRB_P1
    {REG_PAGE, 0x01},
    {0x02, 0x09},  // AVDD LDO on
    {0x01, 0x08},  // crude AVdd off
    {0x02, 0x01},  // master analog power on
    {0x0A, 0x40},  // common mode 0.75 V (the LDO gives 1.72 V)
    {0x03, 0x00},  // left DAC PTM_P3/4, class AB
    {0x04, 0x00},  // right DAC PTM_P3/4, class AB
    {0x7B, 0x01},  // reference charging 40 ms
    {0x14, 0x25},  // headphone soft stepping against the power-up pop
    {0x0C, 0x08},  // HPL <- left DAC
    {0x0D, 0x08},  // HPR <- right DAC
    {0x0E, 0x08},  // LOL <- left DAC
    {0x0F, 0x08},  // LOR <- right DAC
    {0x10, 0x3E},  // HPL -2 dB, unmuted (for the 0.75 V common mode)
    {0x11, 0x3E},  // HPR -2 dB, unmuted
    // LOL and LOR feed the speaker amplifier (TPA6211A1, gain 40k / 15k from each, both summed
    // into IN-). At 0 dB a full-scale DAC gives it 2.0 Vrms out of the ~2.7 Vrms it drives at 1 %
    // THD from 5 V; +3 dB takes that headroom.
    {0x12, 0x03},  // LOL +3 dB, unmuted
    {0x13, 0x03},  // LOR +3 dB, unmuted
    {0x09, 0x3C},  // HPL, HPR, LOL, LOR powered up
};

static const regval_t DAC_ON[] = {
    {REG_PAGE, 0x00},
    {0x3F, 0xD4},  // both DACs up, left data to left, right to right, soft step 1 per WCLK
};

static i2c_master_dev_handle_t s_dev;

static esp_err_t write_list(const regval_t *list, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        const uint8_t buf[] = {list[i].reg, list[i].val};
        const esp_err_t err = i2c_master_transmit(s_dev, buf, sizeof buf, I2C_TIMEOUT_MS);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "write 0x%02X <- 0x%02X failed: %s", list[i].reg, list[i].val, esp_err_to_name(err));
            return err;
        }
    }
    return ESP_OK;
}

esp_err_t aic3204_set_volume(int volume)
{
    const uint8_t b = volume_dac_byte(volume);
    const regval_t list[] = {
        {REG_PAGE, 0x00},
        {REG_DACL_VOL, b},
        {REG_DACR_VOL, b},
        {REG_DAC_CH_SET2, volume_dac_muted(volume) ? 0x0C : 0x00},
    };
    return write_list(list, sizeof list / sizeof list[0]);
}

esp_err_t aic3204_init(i2c_master_bus_handle_t bus, int volume)
{
    const i2c_device_config_t dc = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = I2C_ADDR_AIC3204,
        .scl_speed_hz = 400000,
    };
    esp_err_t err = i2c_master_bus_add_device(bus, &dc, &s_dev);
    if (err == ESP_OK) err = write_list(RESET, sizeof RESET / sizeof RESET[0]);
    if (err != ESP_OK) return err;
    vTaskDelay(pdMS_TO_TICKS(2));  // the reset takes up to 1 ms (SLAA557 3.2); two ticks are at least that
    err = write_list(SETUP, sizeof SETUP / sizeof SETUP[0]);
    if (err != ESP_OK) return err;
    vTaskDelay(pdMS_TO_TICKS(2500));  // the soft stepping settles before the DACs power up
    err = write_list(DAC_ON, sizeof DAC_ON / sizeof DAC_ON[0]);
    if (err == ESP_OK) err = aic3204_set_volume(volume);
    if (err == ESP_OK) ESP_LOGI(TAG, "up, volume %d %%", volume);
    return err;
}
