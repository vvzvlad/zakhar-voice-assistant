// The TLV320AIC3204 DAC: playback from its secondary data input (the ESP32's I2S0 DOUT), 48 kHz
// from the XMOS MCLK, out to the headphone jack and the line outputs that feed the speaker amp.
#pragma once

#include "driver/i2c_master.h"
#include "esp_err.h"

// The stock ESPHome aic3204 setup, with the 1 ms the reset needs, then the 2.5 s the headphone
// soft-stepping needs before the DACs power up, then the volume. Blocks for those 2.5 s. Run it
// after the XMOS has booted: the XMOS holds the DAC in reset until then.
esp_err_t aic3204_init(i2c_master_bus_handle_t bus, int volume);

// Volume in percent through the DAC registers (audio_core's volume mapping).
esp_err_t aic3204_set_volume(int volume);
