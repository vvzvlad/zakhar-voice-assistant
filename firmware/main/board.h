// The Voice PE board (schematic v1.0, 2024-10-09), as the stock home-assistant-voice.yaml @26.5.0
// wires it. GPIO9, GPIO11, GPIO12 and GPIO41 are driven by the XMOS and stay untouched inputs.
#pragma once

// Internal I2C bus: the XMOS XU316 and the TLV320AIC3204 DAC; 4.7k pull-ups on the board.
#define PIN_I2C_SDA 5
#define PIN_I2C_SCL 6
#define I2C_ADDR_XMOS 0x42
#define I2C_ADDR_AIC3204 0x18

#define PIN_XMOS_RESET 4  // high holds the XMOS in reset

// Speaker I2S (I2S0, clocked by the XMOS): DOUT feeds the DAC's secondary data input and the XMOS
// echo reference.
#define PIN_SPK_LRCLK 7
#define PIN_SPK_BCLK 8
#define PIN_SPK_DOUT 10
// Microphone I2S (I2S1, clocked by the XMOS).
#define PIN_MIC_BCLK 13
#define PIN_MIC_LRCLK 14
#define PIN_MIC_DIN 15

#define PIN_AMP_EN 47  // high: the speaker amplifier on (a plugged jack forces it off in hardware)

// LED ring: 12 x WS2812 GRB, data through a buffer powered from the switched LED 5 V.
#define PIN_LED_DATA 21
#define PIN_LED_POWER 45  // high: LED power on

#define PIN_MUTE 3  // high: the mute switch is on (it also cuts the microphones in hardware)
// The dial: active low, external pull-ups.
#define PIN_DIAL_A 16
#define PIN_DIAL_B 18
