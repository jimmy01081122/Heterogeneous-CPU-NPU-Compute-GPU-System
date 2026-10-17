// SPDX-License-Identifier: MIT
// Compute GPU Scratchpad Memory (gpu_scratchpad).
//
// 4096 x 32-bit words (16 KB) on-chip scratchpad memory.
// Supports:
//   * N_LANES parallel read/write ports for SIMD/SIMT Load/Store Unit (LSU)
//     execution (supports arbitrary per-lane addresses including contiguous
//     vectors and gather/scatter `VLD_ABS`/`VST_ABS` for RoPE).
//   * 64-bit (2-word) DMA port for host-to-device (H2D) and device-to-host
//     (D2H) AXI-Stream transfers when the kernel controller is idle.

`timescale 1ns / 1ps

module gpu_scratchpad #(
    parameter int N_LANES   = 4,
    parameter int SPM_WORDS = 4096,
    parameter int AW        = $clog2(SPM_WORDS)
) (
    input  logic                        clk,
    // LSU Port (active during kernel execution)
    input  logic [N_LANES-1:0]          lsu_we,
    input  logic [N_LANES-1:0][AW-1:0]  lsu_addr,
    input  logic [N_LANES-1:0][31:0]    lsu_wdata,
    output logic [N_LANES-1:0][31:0]    lsu_rdata,
    // DMA Port (active during H2D / D2H transfers, 2 words/cycle)
    input  logic                        dma_we0,
    input  logic                        dma_we1,
    input  logic [AW-1:0]               dma_addr0,
    input  logic [AW-1:0]               dma_addr1,
    input  logic [31:0]                 dma_wdata0,
    input  logic [31:0]                 dma_wdata1,
    output logic [31:0]                 dma_rdata0,
    output logic [31:0]                 dma_rdata1
);
    logic [31:0] mem [0:SPM_WORDS-1];

    always_comb begin
        for (int l = 0; l < N_LANES; l++) begin
            lsu_rdata[l] = mem[lsu_addr[l]];
        end
        dma_rdata0 = mem[dma_addr0];
        dma_rdata1 = mem[dma_addr1];
    end

    always_ff @(posedge clk) begin
        if (dma_we0) mem[dma_addr0] <= dma_wdata0;
        if (dma_we1) mem[dma_addr1] <= dma_wdata1;
        for (int l = 0; l < N_LANES; l++) begin
            if (lsu_we[l])
                mem[lsu_addr[l]] <= lsu_wdata[l];
        end
    end
endmodule
