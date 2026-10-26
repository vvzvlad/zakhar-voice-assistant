// FreeRTOS mutexes over pthread mutexes (not recursive, like xSemaphoreCreateMutex's).
#pragma once

#include "freertos/FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct semaphore *SemaphoreHandle_t;

SemaphoreHandle_t xSemaphoreCreateMutex(void);
BaseType_t xSemaphoreTake(SemaphoreHandle_t sem, TickType_t ticks);
BaseType_t xSemaphoreGive(SemaphoreHandle_t sem);

#ifdef __cplusplus
}
#endif
