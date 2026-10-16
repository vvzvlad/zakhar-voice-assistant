// The speaker volume, percent: kept in NVS, applied through the DAC, reported to the server.
#pragma once

// Loads the stored volume (50, the stock initial volume, when none is stored). Call after
// nvs_flash_init(), before the DAC setup reads volume_get().
void volume_init(void);
int volume_get(void);
// 0..100: applies, stores and reports it (zl_report_volume). After zl_start() went through, from
// one task.
void volume_set(int percent);
