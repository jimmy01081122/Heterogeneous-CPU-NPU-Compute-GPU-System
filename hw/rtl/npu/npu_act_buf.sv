// SPDX-License-Identifier: MIT
// On-chip Activation Buffer for the NPU.
//
// Stores:
//   * `x_q` : up to MAX_DIM int8 quantized activations (packed in 32-bit words,
//             even/odd banked so any 4-byte-aligned index `col_idx` can read up
//             to 16 contiguous int8 bytes in a single cycle).
//   * `x_s` : up to MAX_GROUPS IEEE-754 FP32 activation group scales.
//
// Loaded once via NPU_CMD_LOAD_X / NPU_CMD_LOAD_XS and reused across multiple
// back-to-back GEMV calls (e.g. Wq, Wk, Wv or W1, W3).

`timescale 1ns / 1ps

module npu_act_buf #(
    parameter int N_LANES    = 8,
    parameter int MAX_DIM    = 1024,
    parameter int MAX_GROUPS = 256
) (
    input  logic                           clk,
    /* verilator lint_off UNUSEDSIGNAL */
    // Write port for x_q (64-bit = 8 int8 bytes at byte offset wr_x_byte_addr, 8B-aligned)
    input  logic                           wr_x_en,
    input  logic [15:0]                    wr_x_byte_addr,
    input  logic [63:0]                    wr_x_data,
    // Write port for x_s (64-bit = 2 FP32 scales at group index wr_xs_idx, 2-aligned)
    input  logic                           wr_xs_en,
    input  logic [15:0]                    wr_xs_idx,
    input  logic [63:0]                    wr_xs_data,
    // Combinational/synchronous read port for x_q (4-byte aligned col_idx)
    input  logic [15:0]                    rd_col_idx,
    output logic signed [N_LANES-1:0][7:0] rd_act_vec,
    // Read port for x_s
    input  logic [15:0]                    rd_group_idx,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [31:0]                    rd_xs_val
);
    localparam int WORDS32 = MAX_DIM / 4; // 256 32-bit words for MAX_DIM=1024
    localparam int W_AW    = $clog2(WORDS32);
    localparam int G_AW    = $clog2(MAX_GROUPS);

    logic [31:0] x_mem  [0:WORDS32-1];
    logic [31:0] xs_mem [0:MAX_GROUPS-1];

    wire [W_AW-1:0] w0_idx = wr_x_byte_addr[$clog2(MAX_DIM)-1:2];
    wire [W_AW-1:0] w1_idx = w0_idx + W_AW'(1);

    wire [G_AW-1:0] xs0_idx = wr_xs_idx[G_AW-1:0];
    wire [G_AW-1:0] xs1_idx = xs0_idx + G_AW'(1);

    always_ff @(posedge clk) begin
        if (wr_x_en) begin
            x_mem[w0_idx] <= wr_x_data[31:0];
            x_mem[w1_idx] <= wr_x_data[63:32];
        end
        if (wr_xs_en) begin
            xs_mem[xs0_idx] <= wr_xs_data[31:0];
            xs_mem[xs1_idx] <= wr_xs_data[63:32];
        end
    end

    // Read N_LANES contiguous bytes starting at 4-byte-aligned rd_col_idx
    wire [W_AW-1:0] r_word_base = rd_col_idx[$clog2(MAX_DIM)-1:2];
    logic [127:0] rd_window;

    always_comb begin
        rd_window[31:0]   = x_mem[r_word_base];
        rd_window[63:32]  = x_mem[r_word_base + W_AW'(1)];
        rd_window[95:64]  = x_mem[r_word_base + W_AW'(2)];
        rd_window[127:96] = x_mem[r_word_base + W_AW'(3)];
        for (int c = 0; c < N_LANES; c++) begin
            rd_act_vec[c] = $signed(rd_window[c*8 +: 8]);
        end
        rd_xs_val = xs_mem[rd_group_idx[G_AW-1:0]];
    end
endmodule
