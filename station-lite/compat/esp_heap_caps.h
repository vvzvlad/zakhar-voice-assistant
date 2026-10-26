// heap_caps over the C heap. There is no PSRAM: its total is 0. The free size, the lowest free size
// seen and the largest free block of any other memory are the board's MemAvailable.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MALLOC_CAP_8BIT (1u << 2)
#define MALLOC_CAP_SPIRAM (1u << 10)
#define MALLOC_CAP_INTERNAL (1u << 11)

void *heap_caps_malloc(size_t size, uint32_t caps);
void *heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps);
void heap_caps_free(void *ptr);
size_t heap_caps_get_free_size(uint32_t caps);
size_t heap_caps_get_minimum_free_size(uint32_t caps);
size_t heap_caps_get_largest_free_block(uint32_t caps);
size_t heap_caps_get_total_size(uint32_t caps);

#ifdef __cplusplus
}
#endif
