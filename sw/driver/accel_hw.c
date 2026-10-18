/* SPDX-License-Identifier: MIT
 * accel_hw.c - PYNQ-Z2 hardware backend (/dev/mem + Xilinx AXI DMA + CMA arena).
 *
 * Physical memory map matches hw/vivado/create_bd.tcl:
 *   0x43C0_0000 : npu_top AXI4-Lite slave (64 KB)
 *   0x43C1_0000 : gpu_top AXI4-Lite slave (64 KB)
 *   0x4040_0000 : axi_dma_npu (MM2S -> npu_top.s_axis, npu_top.m_axis -> S2MM) on S_AXI_HP0
 *   0x4041_0000 : axi_dma_gpu (MM2S -> gpu_top.s_axis, gpu_top.m_axis -> S2MM) on S_AXI_HP1
 *   0x1800_0000 : Physically contiguous DMA arena (default 32 MB, configurable via HETERO_CMA_PHYS)
 */
#include "accel_backend.h"
#include "accel_hal.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define NPU_BASE_PHYS      0x43C00000u
#define GPU_BASE_PHYS      0x43C10000u
#define DMA_NPU_BASE_PHYS  0x40400000u
#define DMA_GPU_BASE_PHYS  0x40410000u
#define REG_MAP_BYTES      0x00010000u

#define DEFAULT_CMA_PHYS   0x18000000u
#define DEFAULT_CMA_BYTES  (32u * 1024u * 1024u)
#define BOUNCE_BYTES       (256u * 1024u)

static int               g_mem_fd = -1;
static volatile uint8_t *g_reg[DEV_COUNT] = {NULL, NULL};
static volatile uint8_t *g_dma[DEV_COUNT] = {NULL, NULL};

static uint8_t *g_cma_virt  = NULL;
static uint32_t g_cma_phys  = DEFAULT_CMA_PHYS;
static size_t   g_cma_bytes = DEFAULT_CMA_BYTES;
static size_t   g_cma_off   = 0;

/* Bounce buffers at start of CMA for small stack/heap buffers */
static uint8_t *g_tx_bounce_virt = NULL;
static uint32_t g_tx_bounce_phys = 0;
static uint8_t *g_rx_bounce_virt = NULL;
static uint32_t g_rx_bounce_phys = 0;

static inline void dma_wr(AccelDev dev, uint32_t off, uint32_t val) {
    *(volatile uint32_t *)(g_dma[dev] + off) = val;
}

static inline uint32_t dma_rd(AccelDev dev, uint32_t off) {
    return *(volatile uint32_t *)(g_dma[dev] + off);
}

static void dma_reset_channel(AccelDev dev) {
    dma_wr(dev, AXI_DMA_MM2S_DMACR, 0x04u);
    dma_wr(dev, AXI_DMA_S2MM_DMACR, 0x04u);
    for (int i = 0; i < 10000; i++) {
        if (((dma_rd(dev, AXI_DMA_MM2S_DMACR) | dma_rd(dev, AXI_DMA_S2MM_DMACR)) & 0x04u) == 0)
            break;
    }
    dma_wr(dev, AXI_DMA_MM2S_DMACR, 0x01u); /* RS = 1 */
    dma_wr(dev, AXI_DMA_S2MM_DMACR, 0x01u); /* RS = 1 */
}

static int hw_init(void) {
    const char *env_phys = getenv("HETERO_CMA_PHYS");
    const char *env_size = getenv("HETERO_CMA_BYTES");
    if (env_phys && *env_phys) g_cma_phys  = (uint32_t)strtoul(env_phys, NULL, 0);
    if (env_size && *env_size) g_cma_bytes = (size_t)strtoull(env_size, NULL, 0);

    g_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_mem_fd < 0) {
        perror("[accel_hw] open(/dev/mem) failed (run as root on PYNQ-Z2 or use '-b rtl'/'-b sim')");
        return -1;
    }

    g_reg[DEV_NPU] = (volatile uint8_t *)mmap(NULL, REG_MAP_BYTES, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, g_mem_fd, NPU_BASE_PHYS);
    g_reg[DEV_GPU] = (volatile uint8_t *)mmap(NULL, REG_MAP_BYTES, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, g_mem_fd, GPU_BASE_PHYS);
    g_dma[DEV_NPU] = (volatile uint8_t *)mmap(NULL, REG_MAP_BYTES, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, g_mem_fd, DMA_NPU_BASE_PHYS);
    g_dma[DEV_GPU] = (volatile uint8_t *)mmap(NULL, REG_MAP_BYTES, PROT_READ | PROT_WRITE,
                                              MAP_SHARED, g_mem_fd, DMA_GPU_BASE_PHYS);
    g_cma_virt     = (uint8_t *)mmap(NULL, g_cma_bytes, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, g_mem_fd, g_cma_phys);

    if (g_reg[DEV_NPU] == MAP_FAILED || g_reg[DEV_GPU] == MAP_FAILED ||
        g_dma[DEV_NPU] == MAP_FAILED || g_dma[DEV_GPU] == MAP_FAILED ||
        g_cma_virt == MAP_FAILED) {
        perror("[accel_hw] mmap failed");
        return -1;
    }

    g_tx_bounce_virt = g_cma_virt;
    g_tx_bounce_phys = g_cma_phys;
    g_rx_bounce_virt = g_cma_virt + BOUNCE_BYTES;
    g_rx_bounce_phys = g_cma_phys + BOUNCE_BYTES;
    g_cma_off        = 2u * BOUNCE_BYTES;

    dma_reset_channel(DEV_NPU);
    dma_reset_channel(DEV_GPU);
    return 0;
}

static void hw_close(void) {
    for (int d = 0; d < DEV_COUNT; d++) {
        if (g_reg[d] && g_reg[d] != MAP_FAILED) munmap((void *)g_reg[d], REG_MAP_BYTES);
        if (g_dma[d] && g_dma[d] != MAP_FAILED) munmap((void *)g_dma[d], REG_MAP_BYTES);
        g_reg[d] = NULL;
        g_dma[d] = NULL;
    }
    if (g_cma_virt && g_cma_virt != MAP_FAILED) {
        munmap(g_cma_virt, g_cma_bytes);
        g_cma_virt = NULL;
    }
    if (g_mem_fd >= 0) {
        close(g_mem_fd);
        g_mem_fd = -1;
    }
}

static void hw_reg_write(AccelDev dev, uint32_t off, uint32_t val) {
    *(volatile uint32_t *)(g_reg[dev] + off) = val;
}

static uint32_t hw_reg_read(AccelDev dev, uint32_t off) {
    return *(volatile uint32_t *)(g_reg[dev] + off);
}

static inline int is_in_cma(const void *p, uint32_t *phys_out) {
    const uint8_t * b = (const uint8_t *)p;
    if (b >= g_cma_virt && b < g_cma_virt + g_cma_bytes) {
        *phys_out = g_cma_phys + (uint32_t)(b - g_cma_virt);
        return 1;
    }
    return 0;
}

static void hw_stream(AccelDev dev, const void *tx, size_t tx_bytes, void *rx, size_t rx_bytes) {
    uint32_t tx_phys = 0, rx_phys = 0;
    int      rx_bounced = 0;
    size_t   tx_aligned = (tx_bytes + 7u) & ~7u;
    size_t   rx_aligned = (rx_bytes + 7u) & ~7u;

    /* Arm S2MM first so output FIFO never stalls */
    if (rx_bytes > 0) {
        if (!is_in_cma(rx, &rx_phys) || (rx_bytes & 7u) != 0) {
            rx_phys    = g_rx_bounce_phys;
            rx_bounced = 1;
        }
        dma_wr(dev, AXI_DMA_S2MM_DMASR, 0x7000u); /* clear IOC/err bits */
        dma_wr(dev, AXI_DMA_S2MM_DA, rx_phys);
        dma_wr(dev, AXI_DMA_S2MM_LENGTH, (uint32_t)rx_aligned);
    }

    if (tx_bytes > 0) {
        if (!is_in_cma(tx, &tx_phys) || (tx_bytes & 7u) != 0) {
            memset(g_tx_bounce_virt, 0, tx_aligned);
            memcpy(g_tx_bounce_virt, tx, tx_bytes);
            tx_phys = g_tx_bounce_phys;
        }
        dma_wr(dev, AXI_DMA_MM2S_DMASR, 0x7000u);
        dma_wr(dev, AXI_DMA_MM2S_SA, tx_phys);
        dma_wr(dev, AXI_DMA_MM2S_LENGTH, (uint32_t)tx_aligned);
    }

    if (tx_bytes > 0) {
        while ((dma_rd(dev, AXI_DMA_MM2S_DMASR) & 0x02u) == 0) {}
    }
    if (rx_bytes > 0) {
        while ((dma_rd(dev, AXI_DMA_S2MM_DMASR) & 0x02u) == 0) {}
        if (rx_bounced) {
            memcpy(rx, g_rx_bounce_virt, rx_bytes);
        }
    }

    /* Poll accelerator STATUS.done (bit 1 at offset 0x04) */
    while ((hw_reg_read(dev, 0x04u) & 0x02u) == 0) {}
}

static void *hw_alloc(size_t bytes) {
    size_t aligned = (bytes + 63u) & ~63u;
    if (g_cma_off + aligned > g_cma_bytes) {
        fprintf(stderr, "[accel_hw] CMA arena exhausted (%zu + %zu > %zu)\n",
                g_cma_off, aligned, g_cma_bytes);
        return NULL;
    }
    void *p = g_cma_virt + g_cma_off;
    g_cma_off += aligned;
    return p;
}

static void hw_free(void *p) {
    (void)p; /* Bump allocator inside CMA arena */
}

const AccelBackend accel_backend_hw = {
    "hw",
    hw_init,
    hw_close,
    hw_reg_write,
    hw_reg_read,
    hw_stream,
    hw_alloc,
    hw_free
};
