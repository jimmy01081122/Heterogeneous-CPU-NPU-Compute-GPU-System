// SPDX-License-Identifier: MIT
// NPU package: register offsets, command codes, output modes, and constants.

`timescale 1ns / 1ps

package npu_pkg;
    localparam logic [31:0] NPU_MAGIC_ID = 32'h4E50_5531; // "NPU1"

    // AXI4-Lite register byte offsets (7-bit address space 0x00 .. 0x7F)
    localparam logic [6:0] REG_CTRL            = 7'h00;
    localparam logic [6:0] REG_STATUS          = 7'h04;
    localparam logic [6:0] REG_DIM_D           = 7'h08; // output rows d
    localparam logic [6:0] REG_DIM_N           = 7'h0C; // input cols n
    localparam logic [6:0] REG_GROUP_SIZE      = 7'h10; // Q8_0 group size GS
    localparam logic [6:0] REG_TILE_ROWS       = 7'h14; // rows per tile (1 .. N_ROWS)
    localparam logic [6:0] REG_PERF_CYCLES     = 7'h20; // busy cycles
    localparam logic [6:0] REG_PERF_COMPUTE    = 7'h24; // MAC array active cycles
    localparam logic [6:0] REG_PERF_STALL_IN   = 7'h28; // input stream starvation cycles
    localparam logic [6:0] REG_PERF_STALL_OUT  = 7'h2C; // output stream backpressure cycles
    localparam logic [6:0] REG_PERF_BYTES_IN   = 7'h30; // s_axis bytes accepted
    localparam logic [6:0] REG_PERF_BYTES_OUT  = 7'h34; // m_axis bytes emitted
    localparam logic [6:0] REG_ID              = 7'h38; // NPU_MAGIC_ID

    // Commands in REG_CTRL[5:2]
    typedef enum logic [3:0] {
        NPU_CMD_NOP        = 4'd0,
        NPU_CMD_LOAD_X     = 4'd1, // load int8 activation vector x[0..n-1] from s_axis
        NPU_CMD_LOAD_XS    = 4'd2, // load fp32 activation scales xs[0..n/GS-1] from s_axis
        NPU_CMD_RUN_GEMV   = 4'd3, // stream weights from s_axis, emit outputs on m_axis
        NPU_CMD_CLEAR_PERF = 4'd4  // reset all performance counters
    } npu_cmd_e;

    // Output modes in REG_CTRL[7:6]
    typedef enum logic [1:0] {
        NPU_OUT_GROUP_PSUM = 2'd0, // emit INT32 partial sum per (row, group): d*(n/GS) words
        NPU_OUT_ROW_ACC    = 2'd1, // emit INT32 unscaled row sum: d words
        NPU_OUT_ROW_FP32   = 2'd2  // consume per-group w_scale, emit IEEE-754 FP32 row sum: d words
    } npu_out_mode_e;
endpackage
