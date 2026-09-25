/* board.h -- the few board-specific services the harness needs: a cycle counter, a
 * microsecond clock, the CPU clock, free heap, an identification string, and whether
 * a second core exists for the TCN task. Everything algorithmic is board-independent.
 */
#ifndef BOARD_H
#define BOARD_H

#include <Arduino.h>
#include <stdint.h>

#if defined(ESP32)
#include "esp_timer.h"
#define BOARD_HAS_SECOND_CORE 1
static inline uint32_t board_cycles(void)   { return (uint32_t)xthal_get_ccount(); }
static inline int64_t  board_micros(void)   { return (int64_t)esp_timer_get_time(); }
static inline uint32_t board_cpu_mhz(void)  { return (uint32_t)getCpuFrequencyMhz(); }
static inline uint32_t board_free_heap(void){ return (uint32_t)ESP.getFreeHeap(); }
static inline void board_serial_setup(void) { Serial.setRxBufferSize(8192); }
static inline void board_info(char *buf, size_t n)
{
    snprintf(buf, n, "board=ESP32 chip=%s rev=%d cores=%d flash=%uKB@%uMHz",
             ESP.getChipModel(), ESP.getChipRevision(), ESP.getChipCores(),
             (unsigned)(ESP.getFlashChipSize() / 1024),
             (unsigned)(ESP.getFlashChipSpeed() / 1000000u));
}

#elif defined(__IMXRT1062__)
/* Teensy 4.0 / 4.1: Cortex-M7, DWT cycle counter enabled by the core at startup */
#define BOARD_HAS_SECOND_CORE 0
extern "C" {
extern unsigned long _heap_end;
extern char *__brkval;
}
static inline uint32_t board_cycles(void)   { return ARM_DWT_CYCCNT; }
static inline int64_t  board_micros(void)   { return (int64_t)micros(); }
static inline uint32_t board_cpu_mhz(void)  { return (uint32_t)(F_CPU / 1000000u); }
static inline uint32_t board_free_heap(void){ return (uint32_t)((char *)&_heap_end - __brkval); }
static inline void board_serial_setup(void) { /* native USB, no baud, buffered by the stack */ }
static inline void board_info(char *buf, size_t n)
{
    snprintf(buf, n, "board=Teensy4.1 chip=IMXRT1062 cores=1 ram1=512KB(ITCM/DTCM) ram2=512KB(OCRAM) flash=8MB");
}

#else
#error "board.h: add a section for this target"
#endif

#endif /* BOARD_H */
