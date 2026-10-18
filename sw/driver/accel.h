/* SPDX-License-Identifier: MIT
 * accel.h - backend-agnostic driver API for the NPU and Compute GPU.
 *
 * Two backends implement the same low-level primitives (accel_backend.h):
 *   "rtl" : the actual SystemVerilog RTL, Verilated and clocked in-process
 *           (host only; cycle-exact accelerator timing, ideal 64-bit/cycle DMA)
 *   "hw"  : PYNQ-Z2 board, /dev/mem register access + Xilinx AXI DMA + CMA buffers
 *
 * The command sequencing (this file + accel.c) is shared, so the code path
 * validated against RTL is the same one that drives the board.
 */
#ifndef HETERO_ACCEL_H
#define HETERO_ACCEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { DEV_NPU = 0, DEV_GPU = 1, DEV_COUNT = 2 } AccelDev;

typedef struct {
    uint64_t cycles[DEV_COUNT];     /* rtl: simulated clock cycles spent per device (incl. AXI-Lite) */
    uint64_t calls[DEV_COUNT];      /* stream/kernel commands issued */
    uint64_t bytes_in[DEV_COUNT];   /* bytes streamed host -> device */
    uint64_t bytes_out[DEV_COUNT];  /* bytes streamed device -> host */
    uint64_t reg_writes[DEV_COUNT];
    uint64_t reg_reads[DEV_COUNT];
} AccelStats;

/* backend: "rtl" or "hw". Returns 0 on success. */
int  accel_init(const char *backend);
void accel_close(void);
AccelStats *accel_stats(void);

/* DMA-capable memory (hw: physically contiguous CMA; rtl: malloc). */
void *accel_alloc(size_t bytes);
void  accel_free(void *p);

/* ---- NPU ---------------------------------------------------------------- */
/* Load int8 activation vector x[0..n-1] into the NPU activation buffer. */
void npu_load_x(const int8_t *xq, int n);
/* W(d,n) int8 @ x -> exact INT32 per-group partial sums psum[d][n/gs] (row-major).
 * w must reside in accel_alloc() memory. tile_rows=1: W in llama2.c row-major
 * layout; tile_rows=R>1: W pre-packed with npu_pack_tiled(). */
void npu_gemv_psum(const int8_t *w, int d, int n, int gs, int tile_rows, int32_t *psum);
/* Repack a row-major W(d,n) into the NPU tiled stream layout. Returns bytes written. */
size_t npu_pack_tiled(const int8_t *w, int d, int n, int gs, int tile_rows, int8_t *dst);
size_t npu_packed_size(int d, int n);

/* ---- Compute GPU -------------------------------------------------------- */
void gpu_load_image(const uint32_t *words, int n);              /* once at start-up */
void gpu_write_spm(uint32_t word_addr, const void *src, int n_words);
void gpu_read_spm(uint32_t word_addr, void *dst, int n_words);
void gpu_run(uint32_t start_pc, uint32_t vec_len, uint32_t param0, uint32_t param1);

#ifdef __cplusplus
}
#endif
#endif /* HETERO_ACCEL_H */
