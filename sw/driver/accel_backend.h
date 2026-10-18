/* SPDX-License-Identifier: MIT
 * accel_backend.h - primitives every backend (rtl / hw) must provide.
 */
#ifndef HETERO_ACCEL_BACKEND_H
#define HETERO_ACCEL_BACKEND_H

#include "accel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    int      (*init)(void);
    void     (*close)(void);
    void     (*reg_write)(AccelDev dev, uint32_t off, uint32_t val);
    uint32_t (*reg_read)(AccelDev dev, uint32_t off);
    /* Stream tx_bytes from tx (may be NULL/0) into the device and collect
     * rx_bytes (may be NULL/0) from it, then wait until the device reports
     * STATUS.done. The device command must already have been started. */
    void     (*stream)(AccelDev dev, const void *tx, size_t tx_bytes, void *rx, size_t rx_bytes);
    void    *(*alloc)(size_t bytes);
    void     (*free)(void *p);
} AccelBackend;

extern const AccelBackend accel_backend_rtl;  /* accel_rtl.cpp (Verilated SV RTL) */
extern const AccelBackend accel_backend_sim;  /* accel_sim.c   (cycle-accurate C ISA/RTL sim) */
extern const AccelBackend accel_backend_hw;   /* accel_hw.c    (PYNQ-Z2 /dev/mem + AXI DMA) */

#ifdef __cplusplus
}
#endif
#endif /* HETERO_ACCEL_BACKEND_H */
