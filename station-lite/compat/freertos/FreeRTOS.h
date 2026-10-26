// FreeRTOS over POSIX threads: a tick is a millisecond of CLOCK_MONOTONIC.
#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef uint32_t TickType_t;
typedef int BaseType_t;
typedef unsigned int UBaseType_t;

#define portMAX_DELAY ((TickType_t)0xffffffffu)
#define pdMS_TO_TICKS(ms) ((TickType_t)(ms))
#define pdFALSE 0
#define pdTRUE 1
#define pdPASS pdTRUE
#define pdFAIL pdFALSE
