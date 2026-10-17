// SPDX-License-Identifier: MIT
// NPU Top-Level (npu_top) for Heterogeneous CPU-NPU-Compute GPU System on PYNQ-Z2.
//
// Features:
//   * Parameterizable N_ROWS x N_LANES signed INT8 MAC array (default 8x8 = 64 MACs;
//     supports 4x16, 4x8, 2x8 for Hardware-Software Co-design studies).
//   * On-chip Activation Buffer (x_q int8 + x_s fp32) with multi-GEMV reuse
//     (e.g. load x once, run Wq / Wk / Wv or W1 / W3 without re-reading x).
//   * Flexible group size GS (4, 8, 16, 32, 64) and arbitrary output row count d
//     (handles tail tiles when d % N_ROWS != 0, e.g. d=172 in stories260K).
//   * Supports both tiled weights (cfg_tile_rows = N_ROWS) and untiled row-major
//     weights straight from llama2.c checkpoints (cfg_tile_rows = 1).
//   * Three selectable output modes:
//       0: NPU_OUT_GROUP_PSUM - exact INT32 partial sum per (row, group)
//       1: NPU_OUT_ROW_ACC    - exact INT32 row dot-product accumulated across groups
//       2: NPU_OUT_ROW_FP32   - on-chip Q8_0 dequant + IEEE-754 FP32 row accumulation
//   * Built-in cycle-accurate hardware performance counters for co-design profiling.

`timescale 1ns / 1ps

module npu_top
    import npu_pkg::*;
    import fp32_pkg::*;
#(
    parameter int N_ROWS         = 8,
    parameter int N_LANES        = 8,
    parameter int MAX_DIM        = 1024,
    parameter int MAX_GROUPS     = 256
) (
    input  logic        clk,
    input  logic        rst_n,

    // AXI4-Lite Slave Control Interface (byte addresses 0x00 .. 0x7F)
    input  logic [6:0]  s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [6:0]  s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    // AXI4-Stream Slave (MM2S from AXI DMA: activations, scales, or weights)
    input  logic [63:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,

    // AXI4-Stream Master (S2MM to AXI DMA or direct stream to Compute GPU)
    output logic [63:0] m_axis_tdata,
    output logic [7:0]  m_axis_tkeep,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,

    // Interrupt output (high when status_done is set)
    output logic        irq
);

    // -------------------------------------------------------------------------
    // Configuration & Status Registers
    // -------------------------------------------------------------------------
    logic [15:0]   cfg_d;
    logic [15:0]   cfg_n;
    logic [15:0]   cfg_gs;
    logic [7:0]    cfg_tile_rows;
    npu_out_mode_e cfg_out_mode;
    npu_cmd_e      cfg_cmd;
    logic          cmd_start_pulse;
    logic          soft_rst_pulse;

    logic          status_busy;
    logic          status_done;
    logic          status_x_loaded;

    // Hardware Performance Counters
    logic [31:0]   perf_cycles;
    logic [31:0]   perf_compute;
    logic [31:0]   perf_stall_in;
    logic [31:0]   perf_stall_out;
    logic [31:0]   perf_bytes_in;
    logic [31:0]   perf_bytes_out;

    assign irq = status_done;

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave Logic
    // -------------------------------------------------------------------------
    logic        aw_captured, w_captured;
    logic [6:0]  awaddr_reg;
    logic [31:0] wdata_reg;
    /* verilator lint_off UNUSEDSIGNAL */
    logic [3:0]  wstrb_reg;
    logic        unused_tlast;
    assign unused_tlast = s_axis_tlast;
    /* verilator lint_on UNUSEDSIGNAL */

    assign s_axi_awready = !aw_captured && !s_axi_bvalid;
    assign s_axi_wready  = !w_captured  && !s_axi_bvalid;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;

    wire wr_fire = (aw_captured || s_axi_awvalid) && (w_captured || s_axi_wvalid) && !s_axi_bvalid;
    wire [6:0]  wr_addr = aw_captured ? awaddr_reg : s_axi_awaddr;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] wr_data = w_captured  ? wdata_reg  : s_axi_wdata;
    /* verilator lint_on UNUSEDSIGNAL */

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            aw_captured     <= 1'b0;
            w_captured      <= 1'b0;
            awaddr_reg      <= 7'd0;
            wdata_reg       <= 32'd0;
            wstrb_reg       <= 4'd0;
            s_axi_bvalid    <= 1'b0;
            s_axi_rvalid    <= 1'b0;
            s_axi_rdata     <= 32'd0;
            cfg_d           <= 16'd64;
            cfg_n           <= 16'd64;
            cfg_gs          <= 16'd32;
            cfg_tile_rows   <= 8'(N_ROWS);
            cfg_out_mode    <= NPU_OUT_GROUP_PSUM;
            cfg_cmd         <= NPU_CMD_NOP;
            cmd_start_pulse <= 1'b0;
            soft_rst_pulse  <= 1'b0;
        end else begin
            cmd_start_pulse <= 1'b0;
            soft_rst_pulse  <= 1'b0;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_captured <= 1'b1;
                awaddr_reg  <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_captured <= 1'b1;
                wdata_reg  <= s_axi_wdata;
                wstrb_reg  <= s_axi_wstrb;
            end

            if (wr_fire) begin
                aw_captured  <= 1'b0;
                w_captured   <= 1'b0;
                s_axi_bvalid <= 1'b1;
                unique case (wr_addr & 7'h7C)
                    REG_CTRL: begin
                        cmd_start_pulse <= wr_data[0];
                        soft_rst_pulse  <= wr_data[1];
                        cfg_cmd         <= npu_cmd_e'(wr_data[5:2]);
                        cfg_out_mode    <= npu_out_mode_e'(wr_data[7:6]);
                    end
                    REG_DIM_D:      cfg_d         <= wr_data[15:0];
                    REG_DIM_N:      cfg_n         <= wr_data[15:0];
                    REG_GROUP_SIZE: cfg_gs        <= wr_data[15:0];
                    REG_TILE_ROWS:  cfg_tile_rows <= wr_data[7:0];
                    default: ;
                endcase
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                unique case (s_axi_araddr & 7'h7C)
                    REG_CTRL:           s_axi_rdata <= {24'd0, cfg_out_mode, cfg_cmd, 2'b00};
                    REG_STATUS:         s_axi_rdata <= {8'd0, 8'(N_LANES), 8'(N_ROWS), 5'd0,
                                                        status_x_loaded, status_done, status_busy};
                    REG_DIM_D:          s_axi_rdata <= {16'd0, cfg_d};
                    REG_DIM_N:          s_axi_rdata <= {16'd0, cfg_n};
                    REG_GROUP_SIZE:     s_axi_rdata <= {16'd0, cfg_gs};
                    REG_TILE_ROWS:      s_axi_rdata <= {24'd0, cfg_tile_rows};
                    REG_PERF_CYCLES:    s_axi_rdata <= perf_cycles;
                    REG_PERF_COMPUTE:   s_axi_rdata <= perf_compute;
                    REG_PERF_STALL_IN:  s_axi_rdata <= perf_stall_in;
                    REG_PERF_STALL_OUT: s_axi_rdata <= perf_stall_out;
                    REG_PERF_BYTES_IN:  s_axi_rdata <= perf_bytes_in;
                    REG_PERF_BYTES_OUT: s_axi_rdata <= perf_bytes_out;
                    REG_ID:             s_axi_rdata <= NPU_MAGIC_ID;
                    default:            s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Input Stream 32-bit Word FIFO (depth = 32 words = 128 bytes)
    // Accepts 64-bit beats (2 words) from s_axis, pops 1, 2, or 4 words/cycle.
    // -------------------------------------------------------------------------
    localparam int IN_FIFO_WORDS = 32;
    localparam int IN_PTR_W      = $clog2(IN_FIFO_WORDS) + 1;

    logic [31:0]         in_fifo [0:IN_FIFO_WORDS-1];
    logic [IN_PTR_W-1:0] in_wr_ptr, in_rd_ptr;
    wire  [IN_PTR_W-1:0] in_count = in_wr_ptr - in_rd_ptr;
    logic                in_fifo_clear;
    logic [2:0]          in_pop_words; // 0, 1, 2, or 4

    // Accept s_axis when active and room for at least 2 words
    typedef enum logic [2:0] {
        ST_IDLE            = 3'd0,
        ST_LOAD_X          = 3'd1,
        ST_LOAD_XS         = 3'd2,
        ST_GEMV_WGT        = 3'd3,
        ST_GEMV_POST       = 3'd4,
        ST_GEMV_SCALE_FP32 = 3'd5,
        ST_GEMV_EMIT       = 3'd6,
        ST_GEMV_DRAIN      = 3'd7
    } state_e;

    state_e state;

    logic [31:0] in_beats_needed, in_beats_rcvd;
    wire need_more_s_axis = (state != ST_IDLE) && (state != ST_GEMV_DRAIN) &&
                            (in_beats_rcvd < in_beats_needed);

    assign s_axis_tready = need_more_s_axis && (in_count <= IN_PTR_W'(IN_FIFO_WORDS - 2));
    wire s_axis_fire = s_axis_tvalid && s_axis_tready;

    wire [$clog2(IN_FIFO_WORDS)-1:0] in_w0 = in_wr_ptr[$clog2(IN_FIFO_WORDS)-1:0];
    wire [$clog2(IN_FIFO_WORDS)-1:0] in_w1 = in_w0 + 1'b1;

    always_ff @(posedge clk) begin
        if (!rst_n || soft_rst_pulse || in_fifo_clear) begin
            in_wr_ptr     <= '0;
            in_rd_ptr     <= '0;
            in_beats_rcvd <= 32'd0;
        end else begin
            if (s_axis_fire) begin
                in_fifo[in_w0] <= s_axis_tdata[31:0];
                in_fifo[in_w1] <= s_axis_tdata[63:32];
                in_wr_ptr      <= in_wr_ptr + IN_PTR_W'(2);
                in_beats_rcvd  <= in_beats_rcvd + 32'd1;
            end
            if (in_pop_words != 3'd0) begin
                in_rd_ptr <= in_rd_ptr + IN_PTR_W'(in_pop_words);
            end
        end
    end

    localparam int IN_IDX_W = $clog2(IN_FIFO_WORDS);
    wire [IN_IDX_W-1:0] in_r0 = in_rd_ptr[IN_IDX_W-1:0];
    wire [IN_IDX_W-1:0] in_r1 = in_r0 + IN_IDX_W'(1);
    wire [IN_IDX_W-1:0] in_r2 = in_r0 + IN_IDX_W'(2);
    wire [IN_IDX_W-1:0] in_r3 = in_r0 + IN_IDX_W'(3);

    wire [127:0] in_peek128 = {in_fifo[in_r3], in_fifo[in_r2], in_fifo[in_r1], in_fifo[in_r0]};

    // -------------------------------------------------------------------------
    // Output Stream 32-bit Word FIFO (depth = 64 words = 256 bytes)
    // Pushes 1 word/cycle during EMIT, drains 2 words/cycle onto 64-bit m_axis.
    // -------------------------------------------------------------------------
    localparam int OUT_FIFO_WORDS = 64;
    localparam int OUT_PTR_W      = $clog2(OUT_FIFO_WORDS) + 1;

    logic [31:0]          out_fifo [0:OUT_FIFO_WORDS-1];
    logic [OUT_PTR_W-1:0] out_wr_ptr, out_rd_ptr;
    wire  [OUT_PTR_W-1:0] out_count = out_wr_ptr - out_rd_ptr;
    logic                 out_fifo_clear;
    logic                 out_push_en;
    logic [31:0]          out_push_data;

    logic [31:0]          total_out_words;
    logic [31:0]          out_words_sent;

    wire [31:0] out_words_rem = total_out_words - out_words_sent;
    wire        out_last_odd  = (out_words_rem == 32'd1) && (out_count >= OUT_PTR_W'(1));
    wire        out_pair_rdy  = (out_words_rem >= 32'd2) && (out_count >= OUT_PTR_W'(2));

    wire [$clog2(OUT_FIFO_WORDS)-1:0] out_r0 = out_rd_ptr[$clog2(OUT_FIFO_WORDS)-1:0];
    wire [$clog2(OUT_FIFO_WORDS)-1:0] out_r1 = out_r0 + 1'b1;

    assign m_axis_tvalid = (state != ST_IDLE) && (out_pair_rdy || out_last_odd);
    assign m_axis_tdata  = out_last_odd ? {32'd0, out_fifo[out_r0]} : {out_fifo[out_r1], out_fifo[out_r0]};
    assign m_axis_tkeep  = out_last_odd ? 8'h0F : 8'hFF;
    assign m_axis_tlast  = m_axis_tvalid && (out_words_rem <= 32'd2);

    wire m_axis_fire = m_axis_tvalid && m_axis_tready;
    wire [OUT_PTR_W-1:0] out_pop_cnt = m_axis_fire ? (out_last_odd ? OUT_PTR_W'(1) : OUT_PTR_W'(2)) : '0;

    always_ff @(posedge clk) begin
        if (!rst_n || soft_rst_pulse || out_fifo_clear) begin
            out_wr_ptr     <= '0;
            out_rd_ptr     <= '0;
            out_words_sent <= 32'd0;
        end else begin
            if (out_push_en) begin
                out_fifo[out_wr_ptr[$clog2(OUT_FIFO_WORDS)-1:0]] <= out_push_data;
                out_wr_ptr <= out_wr_ptr + OUT_PTR_W'(1);
            end
            if (m_axis_fire) begin
                out_rd_ptr     <= out_rd_ptr + out_pop_cnt;
                out_words_sent <= out_words_sent + (out_last_odd ? 32'd1 : 32'd2);
            end
        end
    end

    // -------------------------------------------------------------------------
    // Activation Buffer Instance
    // -------------------------------------------------------------------------
    logic                           act_wr_x_en, act_wr_xs_en;
    logic [15:0]                    act_wr_addr;
    logic [63:0]                    act_wr_data;
    logic [15:0]                    act_rd_col_idx;
    logic [15:0]                    act_rd_group_idx;
    logic signed [N_LANES-1:0][7:0] act_rd_vec;
    logic [31:0]                    act_rd_xs_val;

    npu_act_buf #(
        .N_LANES    (N_LANES),
        .MAX_DIM    (MAX_DIM),
        .MAX_GROUPS (MAX_GROUPS)
    ) u_act_buf (
        .clk            (clk),
        .wr_x_en        (act_wr_x_en),
        .wr_x_byte_addr (act_wr_addr),
        .wr_x_data      (act_wr_data),
        .wr_xs_en       (act_wr_xs_en),
        .wr_xs_idx      (act_wr_addr),
        .wr_xs_data     (act_wr_data),
        .rd_col_idx     (act_rd_col_idx),
        .rd_act_vec     (act_rd_vec),
        .rd_group_idx   (act_rd_group_idx),
        .rd_xs_val      (act_rd_xs_val)
    );

    // -------------------------------------------------------------------------
    // MAC Array Instance
    // -------------------------------------------------------------------------
    logic                                    mac_valid_in;
    logic                                    mac_first_step;
    logic                                    mac_last_step;
    logic [N_LANES-1:0]                      mac_lane_mask;
    logic signed [N_ROWS-1:0][N_LANES-1:0][7:0] wgt_staging;
    logic signed [N_ROWS-1:0][N_LANES-1:0][7:0] mac_wgt_in;
    logic                                    mac_psum_valid;
    logic signed [N_ROWS-1:0][31:0]          mac_psum_out;

    npu_mac_array #(
        .N_ROWS  (N_ROWS),
        .N_LANES (N_LANES)
    ) u_mac_array (
        .clk        (clk),
        .rst_n      (rst_n && !soft_rst_pulse),
        .valid_in   (mac_valid_in),
        .first_step (mac_first_step),
        .last_step  (mac_last_step),
        .lane_mask  (mac_lane_mask),
        .act_in     (act_rd_vec),
        .wgt_in     (mac_wgt_in),
        .psum_valid (mac_psum_valid),
        .psum_out   (mac_psum_out)
    );

    // -------------------------------------------------------------------------
    // Control FSM & Post-Processing Pipeline
    // -------------------------------------------------------------------------
    logic [15:0] eff_lanes;
    logic [2:0]  words_per_step;
    logic [15:0] steps_per_group;
    logic [15:0] num_groups;

    logic [15:0] row_base;
    logic [7:0]  r_active;
    logic [7:0]  row_in_tile;
    logic [15:0] group_idx;
    logic [15:0] step_idx;
    logic [7:0]  emit_idx;
    logic [15:0] load_idx;

    logic signed [N_ROWS-1:0][31:0] group_psum_latched;
    logic signed [N_ROWS-1:0][31:0] row_i32_acc;
    logic        [N_ROWS-1:0][31:0] row_fp32_acc;

    assign act_rd_col_idx   = group_idx * cfg_gs + step_idx * eff_lanes;
    assign act_rd_group_idx = group_idx;

    // Unpack current step's weight bytes from in_peek128
    logic signed [N_LANES-1:0][7:0] step_wgt_vec;
    always_comb begin
        for (int c = 0; c < N_LANES; c++) begin
            step_wgt_vec[c] = $signed(in_peek128[c*8 +: 8]);
        end
        mac_wgt_in = wgt_staging;
        if (row_in_tile < 8'(N_ROWS)) begin
            mac_wgt_in[row_in_tile[$clog2(N_ROWS)-1:0]] = step_wgt_vec;
        end
    end

    wire wgt_step_rdy = (state == ST_GEMV_WGT) &&
                        (in_count >= IN_PTR_W'(words_per_step)) &&
                        (out_count <= OUT_PTR_W'(OUT_FIFO_WORDS - N_ROWS * 2));

    wire scale_step_rdy = (state == ST_GEMV_SCALE_FP32) && (in_count >= IN_PTR_W'(1));
    wire emit_step_rdy  = (state == ST_GEMV_EMIT) && (out_count < OUT_PTR_W'(OUT_FIFO_WORDS - 2));

    assign mac_valid_in   = wgt_step_rdy && (row_in_tile + 8'd1 == r_active);
    assign mac_first_step = (step_idx == 16'd0);
    assign mac_last_step  = (step_idx + 16'd1 == steps_per_group);

    // Combinational helpers for next-tile calculation
    wire [15:0] next_row_base = row_base + {8'd0, r_active};
    wire [15:0] rem_rows_next = cfg_d - next_row_base;
    wire [7:0]  tile_cap      = (cfg_tile_rows == 8'd0 || cfg_tile_rows > 8'(N_ROWS)) ? 8'(N_ROWS) : cfg_tile_rows;

    // Combinational FP32 group scale multiply-accumulate for ST_GEMV_SCALE_FP32
    logic [31:0] fp32_psum_cur, fp32_scale_prod, fp32_term, fp32_acc_next;
    always_comb begin
        fp32_psum_cur   = i32_to_fp32(group_psum_latched[emit_idx[$clog2(N_ROWS)-1:0]]);
        fp32_scale_prod = fp32_mul(act_rd_xs_val, in_peek128[31:0]);
        fp32_term       = fp32_mul(fp32_psum_cur, fp32_scale_prod);
        if (group_idx == 16'd0)
            fp32_acc_next = fp32_term;
        else
            fp32_acc_next = fp32_add(row_fp32_acc[emit_idx[$clog2(N_ROWS)-1:0]], fp32_term);
    end

    // Drive FIFO pop / push signals
    always_comb begin
        in_fifo_clear  = 1'b0;
        out_fifo_clear = 1'b0;
        in_pop_words   = 3'd0;
        out_push_en    = 1'b0;
        out_push_data  = 32'd0;
        act_wr_x_en    = 1'b0;
        act_wr_xs_en   = 1'b0;
        act_wr_addr    = load_idx;
        act_wr_data    = in_peek128[63:0];

        unique case (state)
            ST_IDLE: begin
                if (cmd_start_pulse && (cfg_cmd == NPU_CMD_LOAD_X ||
                                        cfg_cmd == NPU_CMD_LOAD_XS ||
                                        cfg_cmd == NPU_CMD_RUN_GEMV)) begin
                    in_fifo_clear  = 1'b1;
                    out_fifo_clear = (cfg_cmd == NPU_CMD_RUN_GEMV);
                end
            end
            ST_LOAD_X: begin
                if (in_count >= IN_PTR_W'(2)) begin
                    in_pop_words = 3'd2;
                    act_wr_x_en  = 1'b1;
                end
            end
            ST_LOAD_XS: begin
                if (in_count >= IN_PTR_W'(2)) begin
                    in_pop_words = 3'd2;
                    act_wr_xs_en = 1'b1;
                end else if (in_count == IN_PTR_W'(1) && (load_idx + 16'd1 >= num_groups)) begin
                    in_pop_words = 3'd1;
                    act_wr_xs_en = 1'b1;
                end
            end
            ST_GEMV_WGT: begin
                if (wgt_step_rdy) begin
                    in_pop_words = words_per_step;
                end
            end
            ST_GEMV_SCALE_FP32: begin
                if (scale_step_rdy) begin
                    in_pop_words = 3'd1;
                end
            end
            ST_GEMV_EMIT: begin
                if (emit_step_rdy) begin
                    out_push_en = 1'b1;
                    unique case (cfg_out_mode)
                        NPU_OUT_GROUP_PSUM: out_push_data = group_psum_latched[emit_idx[$clog2(N_ROWS)-1:0]];
                        NPU_OUT_ROW_ACC:    out_push_data = row_i32_acc[emit_idx[$clog2(N_ROWS)-1:0]];
                        NPU_OUT_ROW_FP32:   out_push_data = row_fp32_acc[emit_idx[$clog2(N_ROWS)-1:0]];
                        default:            out_push_data = 32'd0;
                    endcase
                end
            end
            default: ;
        endcase
    end

    // Main state machine
    always_ff @(posedge clk) begin
        if (!rst_n || soft_rst_pulse) begin
            state           <= ST_IDLE;
            status_busy     <= 1'b0;
            status_done     <= 1'b0;
            status_x_loaded <= 1'b0;
            eff_lanes       <= 16'd8;
            words_per_step  <= 3'd2;
            steps_per_group <= 16'd4;
            num_groups      <= 16'd2;
            mac_lane_mask   <= '1;
            row_base        <= 16'd0;
            r_active        <= 8'd1;
            row_in_tile     <= 8'd0;
            group_idx       <= 16'd0;
            step_idx        <= 16'd0;
            emit_idx        <= 8'd0;
            load_idx        <= 16'd0;
            in_beats_needed <= 32'd0;
            total_out_words <= 32'd0;
            wgt_staging     <= '0;
            group_psum_latched <= '0;
            row_i32_acc     <= '0;
            row_fp32_acc    <= '0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (cmd_start_pulse) begin
                        status_done <= 1'b0;
                        num_groups  <= (cfg_gs != 16'd0) ? (cfg_n / cfg_gs) : 16'd1;
                        unique case (cfg_cmd)
                            NPU_CMD_CLEAR_PERF: begin
                                status_done <= 1'b1;
                            end
                            NPU_CMD_LOAD_X: begin
                                status_busy     <= 1'b1;
                                load_idx        <= 16'd0;
                                in_beats_needed <= ({16'd0, cfg_n} + 32'd7) >> 3;
                                state           <= ST_LOAD_X;
                            end
                            NPU_CMD_LOAD_XS: begin
                                status_busy     <= 1'b1;
                                load_idx        <= 16'd0;
                                in_beats_needed <= (({16'd0, cfg_n / cfg_gs}) + 32'd1) >> 1;
                                state           <= ST_LOAD_XS;
                            end
                            NPU_CMD_RUN_GEMV: begin
                                logic [15:0] el;
                                logic [15:0] ng;
                                logic [31:0] wgt_bytes, scale_bytes;
                                el          = (cfg_gs < 16'(N_LANES)) ? cfg_gs : 16'(N_LANES);
                                ng          = (cfg_gs != 16'd0) ? (cfg_n / cfg_gs) : 16'd1;
                                eff_lanes       <= el;
                                words_per_step  <= 3'(el >> 2);
                                steps_per_group <= cfg_gs / el;
                                num_groups      <= ng;
                                for (int c = 0; c < N_LANES; c++) begin
                                    mac_lane_mask[c] <= (c < int'(el));
                                end
                                row_base    <= 16'd0;
                                r_active    <= (cfg_d < {8'd0, tile_cap}) ? cfg_d[7:0] : tile_cap;
                                row_in_tile <= 8'd0;
                                group_idx   <= 16'd0;
                                step_idx    <= 16'd0;
                                wgt_staging <= '0;
                                row_i32_acc <= '0;
                                row_fp32_acc <= '0;
                                wgt_bytes   = {16'd0, cfg_d} * {16'd0, cfg_n};
                                scale_bytes = (cfg_out_mode == NPU_OUT_ROW_FP32) ?
                                              ({16'd0, cfg_d} * {16'd0, ng} * 32'd4) : 32'd0;
                                in_beats_needed <= (wgt_bytes + scale_bytes + 32'd7) >> 3;
                                total_out_words <= (cfg_out_mode == NPU_OUT_GROUP_PSUM) ?
                                                   ({16'd0, cfg_d} * {16'd0, ng}) : {16'd0, cfg_d};
                                status_busy <= 1'b1;
                                state       <= ST_GEMV_WGT;
                            end
                            default: begin
                                status_done <= 1'b1;
                            end
                        endcase
                    end
                end

                ST_LOAD_X: begin
                    if (in_count >= IN_PTR_W'(2)) begin
                        if (load_idx + 16'd8 >= cfg_n) begin
                            status_x_loaded <= 1'b1;
                            status_busy     <= 1'b0;
                            status_done     <= 1'b1;
                            state           <= ST_IDLE;
                        end else begin
                            load_idx <= load_idx + 16'd8;
                        end
                    end
                end

                ST_LOAD_XS: begin
                    if (act_wr_xs_en) begin
                        if (load_idx + 16'd2 >= num_groups) begin
                            status_busy <= 1'b0;
                            status_done <= 1'b1;
                            state       <= ST_IDLE;
                        end else begin
                            load_idx <= load_idx + 16'd2;
                        end
                    end
                end

                ST_GEMV_WGT: begin
                    if (wgt_step_rdy) begin
                        wgt_staging[row_in_tile[$clog2(N_ROWS)-1:0]] <= step_wgt_vec;
                        if (row_in_tile + 8'd1 == r_active) begin
                            row_in_tile <= 8'd0;
                            if (step_idx + 16'd1 == steps_per_group) begin
                                state <= ST_GEMV_POST;
                            end else begin
                                step_idx <= step_idx + 16'd1;
                            end
                        end else begin
                            row_in_tile <= row_in_tile + 8'd1;
                        end
                    end
                end

                ST_GEMV_POST: begin
                    if (mac_psum_valid) begin
                        group_psum_latched <= mac_psum_out;
                        emit_idx           <= 8'd0;
                        unique case (cfg_out_mode)
                            NPU_OUT_GROUP_PSUM: begin
                                state <= ST_GEMV_EMIT;
                            end
                            NPU_OUT_ROW_ACC: begin
                                for (int r = 0; r < N_ROWS; r++) begin
                                    if (group_idx == 16'd0)
                                        row_i32_acc[r] <= mac_psum_out[r];
                                    else
                                        row_i32_acc[r] <= row_i32_acc[r] + mac_psum_out[r];
                                end
                                if (group_idx + 16'd1 == num_groups) begin
                                    state <= ST_GEMV_EMIT;
                                end else begin
                                    group_idx <= group_idx + 16'd1;
                                    step_idx  <= 16'd0;
                                    state     <= ST_GEMV_WGT;
                                end
                            end
                            NPU_OUT_ROW_FP32: begin
                                state <= ST_GEMV_SCALE_FP32;
                            end
                            default: state <= ST_GEMV_EMIT;
                        endcase
                    end
                end

                ST_GEMV_SCALE_FP32: begin
                    if (scale_step_rdy) begin
                        row_fp32_acc[emit_idx[$clog2(N_ROWS)-1:0]] <= fp32_acc_next;
                        if (emit_idx + 8'd1 == r_active) begin
                            emit_idx <= 8'd0;
                            if (group_idx + 16'd1 == num_groups) begin
                                state <= ST_GEMV_EMIT;
                            end else begin
                                group_idx <= group_idx + 16'd1;
                                step_idx  <= 16'd0;
                                state     <= ST_GEMV_WGT;
                            end
                        end else begin
                            emit_idx <= emit_idx + 8'd1;
                        end
                    end
                end

                ST_GEMV_EMIT: begin
                    if (emit_step_rdy) begin
                        if (emit_idx + 8'd1 == r_active) begin
                            emit_idx <= 8'd0;
                            if (cfg_out_mode == NPU_OUT_GROUP_PSUM && (group_idx + 16'd1 < num_groups)) begin
                                group_idx <= group_idx + 16'd1;
                                step_idx  <= 16'd0;
                                state     <= ST_GEMV_WGT;
                            end else if (next_row_base < cfg_d) begin
                                row_base    <= next_row_base;
                                r_active    <= (rem_rows_next < {8'd0, tile_cap}) ? rem_rows_next[7:0] : tile_cap;
                                group_idx   <= 16'd0;
                                step_idx    <= 16'd0;
                                wgt_staging <= '0;
                                state       <= ST_GEMV_WGT;
                            end else begin
                                state <= ST_GEMV_DRAIN;
                            end
                        end else begin
                            emit_idx <= emit_idx + 8'd1;
                        end
                    end
                end

                ST_GEMV_DRAIN: begin
                    if (out_words_sent >= total_out_words) begin
                        status_busy <= 1'b0;
                        status_done <= 1'b1;
                        state       <= ST_IDLE;
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Cycle-Accurate Performance Counters
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n || soft_rst_pulse || (cmd_start_pulse && cfg_cmd == NPU_CMD_CLEAR_PERF)) begin
            perf_cycles    <= 32'd0;
            perf_compute   <= 32'd0;
            perf_stall_in  <= 32'd0;
            perf_stall_out <= 32'd0;
            perf_bytes_in  <= 32'd0;
            perf_bytes_out <= 32'd0;
        end else begin
            if (status_busy)
                perf_cycles <= perf_cycles + 32'd1;
            if (mac_valid_in)
                perf_compute <= perf_compute + 32'd1;
            if (status_busy && need_more_s_axis && !s_axis_tvalid && (in_count < IN_PTR_W'(words_per_step)))
                perf_stall_in <= perf_stall_in + 32'd1;
            if (m_axis_tvalid && !m_axis_tready)
                perf_stall_out <= perf_stall_out + 32'd1;
            if (s_axis_fire)
                perf_bytes_in <= perf_bytes_in + 32'd8;
            if (m_axis_fire)
                perf_bytes_out <= perf_bytes_out + (out_last_odd ? 32'd4 : 32'd8);
        end
    end
endmodule
