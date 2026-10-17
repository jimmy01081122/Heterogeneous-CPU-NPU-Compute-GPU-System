/* SPDX-License-Identifier: MIT
 * accel_hal.h - Hardware Abstraction Layer (HAL) for NPU and Compute GPU.
 *
 * Register offsets and command encodings match:
 *   - hw/rtl/npu/npu_pkg.sv  (NPU_MAGIC_ID = 0x4E505531 "NPU1")
 *   - hw/rtl/gpu/gpu_pkg.sv  (GPU_MAGIC_ID = 0x47505531 "GPU1")
 */
#ifndef HETERO_ACCEL_HAL_H
#define HETERO_ACCEL_HAL_H

#include <stdint.h>
#include <stddef.h>

/* ---- NPU Register Map (byte offsets from NPU AXI4-Lite base) ---- */
#define NPU_REG_CTRL            0x00u
#define NPU_REG_STATUS          0x04u
#define NPU_REG_DIM_D           0x08u
#define NPU_REG_DIM_N           0x0Cu
#define NPU_REG_GROUP_SIZE      0x10u
#define NPU_REG_TILE_ROWS       0x14u
#define NPU_REG_PERF_CYCLES     0x20u
#define NPU_REG_PERF_COMPUTE    0x24u
#define NPU_REG_PERF_STALL_IN   0x28u
#define NPU_REG_PERF_STALL_OUT  0x2Cu
#define NPU_REG_PERF_BYTES_IN   0x30u
#define NPU_REG_PERF_BYTES_OUT  0x34u
#define NPU_REG_ID              0x38u

#define NPU_MAGIC_ID            0x4E505531u

#define NPU_CMD_LOAD_X          1u
#define NPU_CMD_LOAD_XS         2u
#define NPU_CMD_RUN_GEMV        3u
#define NPU_CMD_CLEAR_PERF      4u

#define NPU_OUT_GROUP_PSUM      0u
#define NPU_OUT_ROW_ACC         1u
#define NPU_OUT_ROW_FP32        2u

static inline uint32_t npu_ctrl_word(uint32_t cmd, uint32_t out_mode) {
    return 1u | ((cmd & 0xFu) << 2) | ((out_mode & 0x3u) << 6);
}

/* ---- Compute GPU Register Map (byte offsets from GPU AXI4-Lite base) ---- */
#define GPU_REG_CTRL            0x00u
#define GPU_REG_STATUS          0x04u
#define GPU_REG_VEC_LEN         0x08u
#define GPU_REG_DMA_ADDR        0x0Cu
#define GPU_REG_DMA_LEN         0x10u
#define GPU_REG_IMEM_ADDR       0x14u
#define GPU_REG_IMEM_DATA       0x18u
#define GPU_REG_PARAM0          0x1Cu
#define GPU_REG_PARAM1          0x20u
#define GPU_REG_PERF_CYCLES     0x24u
#define GPU_REG_PERF_INSTS      0x28u
#define GPU_REG_PERF_STALL      0x2Cu
#define GPU_REG_PERF_BYTES_IN   0x30u
#define GPU_REG_PERF_BYTES_OUT  0x34u
#define GPU_REG_ID              0x38u

#define GPU_MAGIC_ID            0x47505531u

#define GPU_CMD_DMA_H2D         1u
#define GPU_CMD_DMA_D2H         2u
#define GPU_CMD_RUN_KERNEL      3u
#define GPU_CMD_CLEAR_PERF      4u

static inline uint32_t gpu_ctrl_word(uint32_t cmd) {
    return 1u | ((cmd & 0xFu) << 2);
}

/* ---- Xilinx AXI DMA Direct Register Mode Offsets ---- */
#define AXI_DMA_MM2S_DMACR      0x00u
#define AXI_DMA_MM2S_DMASR      0x04u
#define AXI_DMA_MM2S_SA         0x18u
#define AXI_DMA_MM2S_LENGTH     0x28u
#define AXI_DMA_S2MM_DMACR      0x30u
#define AXI_DMA_S2MM_DMASR      0x34u
#define AXI_DMA_S2MM_DA         0x48u
#define AXI_DMA_S2MM_LENGTH     0x58u

#endif /* HETERO_ACCEL_HAL_H */
