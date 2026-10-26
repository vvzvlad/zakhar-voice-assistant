// FreeRTOS tasks, task notifications and mutexes over POSIX threads.
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "esp_timer.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

struct task {
    TaskFunction_t fn;
    void *arg;
    char name[32];
    pthread_t thread;
    pthread_mutex_t lock;  // the notification
    pthread_cond_t cond;
    uint32_t value;
    bool pending;
    struct task *next;
};

struct semaphore {
    pthread_mutex_t mutex;
};

static pthread_mutex_t s_tasks_lock = PTHREAD_MUTEX_INITIALIZER;
static struct task *s_tasks;
static __thread struct task *s_self;

// CLOCK_MONOTONIC `ms` from now, for pthread_cond_timedwait.
static struct timespec after_ms(uint32_t ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_sec += ms / 1000;
    ts.tv_nsec += (long)(ms % 1000) * 1000000;
    if (ts.tv_nsec >= 1000000000) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000;
    }
    return ts;
}

static void *task_main(void *arg)
{
    s_self = arg;
    s_self->fn(s_self->arg);
    return NULL;
}

BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char *name, uint32_t stack, void *arg,
                                   UBaseType_t priority, TaskHandle_t *out, BaseType_t core)
{
    (void)stack;
    (void)priority;
    (void)core;
    struct task *t = calloc(1, sizeof *t);
    if (t == NULL) return pdFAIL;
    t->fn = fn;
    t->arg = arg;
    strncpy(t->name, name, sizeof t->name - 1);
    pthread_condattr_t ca;
    pthread_condattr_init(&ca);
    pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
    pthread_mutex_init(&t->lock, NULL);
    pthread_cond_init(&t->cond, &ca);
    pthread_condattr_destroy(&ca);
    if (out != NULL) *out = t;
    if (pthread_create(&t->thread, NULL, task_main, t) != 0) {
        if (out != NULL) *out = NULL;
        free(t);
        return pdFAIL;
    }
    pthread_mutex_lock(&s_tasks_lock);
    t->next = s_tasks;
    s_tasks = t;
    pthread_mutex_unlock(&s_tasks_lock);
    return pdPASS;
}

TaskHandle_t xTaskGetHandle(const char *name)
{
    pthread_mutex_lock(&s_tasks_lock);
    struct task *t = s_tasks;
    while (t != NULL && strcmp(t->name, name) != 0) t = t->next;
    pthread_mutex_unlock(&s_tasks_lock);
    return t;
}

BaseType_t xTaskNotify(TaskHandle_t task, uint32_t value, eNotifyAction action)
{
    (void)action;
    pthread_mutex_lock(&task->lock);
    task->value |= value;
    task->pending = true;
    pthread_cond_signal(&task->cond);
    pthread_mutex_unlock(&task->lock);
    return pdPASS;
}

// As FreeRTOS: the bits to clear on entry go only when nothing is pending; *value gets the value
// whether or not a notification came, and the bits to clear on exit go only when one did.
BaseType_t xTaskNotifyWait(uint32_t clear_on_entry, uint32_t clear_on_exit, uint32_t *value, TickType_t ticks)
{
    struct task *t = s_self;
    pthread_mutex_lock(&t->lock);
    if (!t->pending) {
        t->value &= ~clear_on_entry;
        const struct timespec until = after_ms(ticks);
        while (!t->pending && ticks > 0) {
            if (ticks == portMAX_DELAY) {
                pthread_cond_wait(&t->cond, &t->lock);
            } else if (pthread_cond_timedwait(&t->cond, &t->lock, &until) == ETIMEDOUT) {
                break;
            }
        }
    }
    if (value != NULL) *value = t->value;
    const bool got = t->pending;
    if (got) {
        t->value &= ~clear_on_exit;
        t->pending = false;
    }
    pthread_mutex_unlock(&t->lock);
    return got ? pdTRUE : pdFALSE;
}

TickType_t xTaskGetTickCount(void)
{
    return (TickType_t)(esp_timer_get_time() / 1000);
}

SemaphoreHandle_t xSemaphoreCreateMutex(void)
{
    struct semaphore *s = malloc(sizeof *s);
    if (s != NULL) pthread_mutex_init(&s->mutex, NULL);
    return s;
}

BaseType_t xSemaphoreTake(SemaphoreHandle_t sem, TickType_t ticks)
{
    if (ticks == portMAX_DELAY) return pthread_mutex_lock(&sem->mutex) == 0 ? pdTRUE : pdFALSE;
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += ticks / 1000;
    until.tv_nsec += (long)(ticks % 1000) * 1000000;
    if (until.tv_nsec >= 1000000000) {
        until.tv_sec++;
        until.tv_nsec -= 1000000000;
    }
    return pthread_mutex_timedlock(&sem->mutex, &until) == 0 ? pdTRUE : pdFALSE;
}

BaseType_t xSemaphoreGive(SemaphoreHandle_t sem)
{
    return pthread_mutex_unlock(&sem->mutex) == 0 ? pdTRUE : pdFALSE;
}
