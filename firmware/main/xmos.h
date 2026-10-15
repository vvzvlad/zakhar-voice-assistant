// The XMOS XU316 audio processor. It boots by itself from its own flash and clocks both I2S buses;
// every boot of it also pulses the AIC3204's hardware reset.
#pragma once

#include "driver/i2c_master.h"

// Pulses the XMOS reset, waits until it answers GETVERSION (logging a version other than the 1.3.1
// this firmware is made for) and selects the processing stage of each microphone channel:
// ch0 = AGC, ch1 = NS. Blocks until the XMOS answers; an error is a failed pin or I2C setup or a
// failed stage write.
esp_err_t xmos_boot(i2c_master_bus_handle_t bus);
