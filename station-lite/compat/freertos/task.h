// Tasks are threads with the libc's default stack; their stack size, priority and core are not
// used. Each task has the one notification value of FreeRTOS, set with bits (eSetBits) and waited
// for with xTaskNotifyWait.
#pragma once

#include "freertos/FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct task *TaskHandle_t;
typedef void (*TaskFunction_t)(void *arg);

typedef enum {
    eSetBits,
} eNotifyAction;

// *out is set before the task runs.
BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char *name, uint32_t stack, void *arg,
                                   UBaseType_t priority, TaskHandle_t *out, BaseType_t core);
// The task created with this name, NULL when there is none.
TaskHandle_t xTaskGetHandle(const char *name);
BaseType_t xTaskNotify(TaskHandle_t task, uint32_t value, eNotifyAction action);
// Called from a task created by xTaskCreatePinnedToCore.
BaseType_t xTaskNotifyWait(uint32_t clear_on_entry, uint32_t clear_on_exit, uint32_t *value, TickType_t ticks);
TickType_t xTaskGetTickCount(void);

#ifdef __cplusplus
}
#endif
