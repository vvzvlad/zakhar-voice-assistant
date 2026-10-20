// Every C++ allocation of the image (operator new, so the music player's library, tflite-micro,
// NVS) goes to PSRAM first and to internal RAM only when PSRAM has no room. With
// CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL a block under 16 KB would try internal RAM first, and the
// player's small objects alone took the internal RAM that the link's WebSocket task stack needs at
// every connection. C allocations (malloc) keep that rule. operator delete stays the library's,
// free(), which frees either.
#include <cstdlib>
#include <new>

#include "esp_heap_caps.h"

static void *alloc(std::size_t n)
{
    return heap_caps_malloc_prefer(n ? n : 1, 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT, MALLOC_CAP_DEFAULT);
}

// The image is built without C++ exceptions: a failed allocation aborts, as the library's would.
void *operator new(std::size_t n)
{
    void *p = alloc(n);
    if (p == nullptr) abort();
    return p;
}

void *operator new[](std::size_t n)
{
    return operator new(n);
}

void *operator new(std::size_t n, const std::nothrow_t &) noexcept
{
    return alloc(n);
}

void *operator new[](std::size_t n, const std::nothrow_t &) noexcept
{
    return alloc(n);
}
