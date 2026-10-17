// SPDX-License-Identifier: MIT
// Compute GPU Top-Level (gpu_top) for Heterogeneous CPU-NPU-Compute GPU System.
//
// Integrates the 6 core submodules specified in method.md Section 4.3:
//   1. Instruction Decoder           (gpu_decoder)
//   2. Parallel Execution Controller (gpu_controller)
//   3. Vector Register File          (gpu_regfile)
//   4. Arithmetic Processing Lanes   (gpu_lane x N_LANES, default N_LANES = 4)
//   5. Scratchpad Memory             (gpu_scratchpad, 4096 x 32-bit = 16 KB)
//   6. Load/Store Unit               (gpu_lsu)
//
// Host/DMA interfaces match npu_top (AXI4-Lite control + 64-bit AXI4-Stream
// slave/master) so the runtime HAL can drive both accelerators uniformly.

`timescale 1ns / 1ps

module gpu_top
    import gpu_pkg::*;
    import fp32_pkg::*;
#(
    parameter int N_LANES    = 4,
    parameter int SPM_WORDS  = 4096,
    parameter int IMEM_WORDS = 256
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

    // AXI4-Stream Slave (H2D DMA into Scratchpad Memory)
    input  logic [63:0] s_axis_tdata,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,

    // AXI4-Stream Master (D2H DMA out of Scratchpad Memory)
    output logic [63:0] m_axis_tdata,
    output logic [7:0]  m_axis_tkeep,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,

    // Interrupt output
    output logic        irq
);
    localparam int SPM_AW  = $clog2(SPM_WORDS);
    localparam int IMEM_AW = $clog2(IMEM_WORDS);

    // -------------------------------------------------------------------------
    // Instruction Memory (IMEM) & Control Registers
    // -------------------------------------------------------------------------
    logic [31:0]      imem [0:IMEM_WORDS-1];
    logic [IMEM_AW-1:0] cfg_imem_addr;

    logic [31:0]      cfg_vec_len;
    logic [SPM_AW-1:0] cfg_dma_addr;
    logic [31:0]      cfg_dma_len;
    logic [31:0]      cfg_param0;
    logic [31:0]      cfg_param1;
    gpu_cmd_e         cfg_cmd;
    logic             cmd_start_pulse;
    logic             soft_rst_pulse;

    logic             status_busy;
    logic             status_done;
    assign irq = status_done;

    // Performance counters
    logic [31:0]      perf_cycles;
    logic [31:0]      perf_insts;
    logic [31:0]      perf_stall;
    logic [31:0]      perf_bytes_in;
    logic [31:0]      perf_bytes_out;

    // -------------------------------------------------------------------------
    // AXI4-Lite Slave
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
    wire [31:0] wr_data = w_captured  ? wdata_reg  : s_axi_wdata;

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
            cfg_vec_len     <= 32'd64;
            cfg_dma_addr    <= '0;
            cfg_dma_len     <= 32'd0;
            cfg_imem_addr   <= '0;
            cfg_param0      <= 32'd0;
            cfg_param1      <= 32'd0;
            cfg_cmd         <= GPU_CMD_NOP;
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
                    GPU_REG_CTRL: begin
                        cmd_start_pulse <= wr_data[0];
                        soft_rst_pulse  <= wr_data[1];
                        cfg_cmd         <= gpu_cmd_e'(wr_data[5:2]);
                    end
                    GPU_REG_VEC_LEN:   cfg_vec_len   <= wr_data;
                    GPU_REG_DMA_ADDR:  cfg_dma_addr  <= wr_data[SPM_AW-1:0];
                    GPU_REG_DMA_LEN:   cfg_dma_len   <= wr_data;
                    GPU_REG_IMEM_ADDR: cfg_imem_addr <= wr_data[IMEM_AW-1:0];
                    GPU_REG_IMEM_DATA: begin
                        imem[cfg_imem_addr] <= wr_data;
                        cfg_imem_addr       <= cfg_imem_addr + IMEM_AW'(1);
                    end
                    GPU_REG_PARAM0:    cfg_param0    <= wr_data;
                    GPU_REG_PARAM1:    cfg_param1    <= wr_data;
                    default: ;
                endcase
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                unique case (s_axi_araddr & 7'h7C)
                    GPU_REG_CTRL:           s_axi_rdata <= {26'd0, cfg_cmd, 2'b00};
                    GPU_REG_STATUS:         s_axi_rdata <= {8'd0, 8'(SPM_WORDS >> 8), 8'(N_LANES),
                                                            6'd0, status_done, status_busy};
                    GPU_REG_VEC_LEN:        s_axi_rdata <= cfg_vec_len;
                    GPU_REG_DMA_ADDR:       s_axi_rdata <= {{(32-SPM_AW){1'b0}}, cfg_dma_addr};
                    GPU_REG_DMA_LEN:        s_axi_rdata <= cfg_dma_len;
                    GPU_REG_IMEM_ADDR:      s_axi_rdata <= {{(32-IMEM_AW){1'b0}}, cfg_imem_addr};
                    GPU_REG_IMEM_DATA:      s_axi_rdata <= imem[cfg_imem_addr];
                    GPU_REG_PARAM0:         s_axi_rdata <= cfg_param0;
                    GPU_REG_PARAM1:         s_axi_rdata <= cfg_param1;
                    GPU_REG_PERF_CYCLES:    s_axi_rdata <= perf_cycles;
                    GPU_REG_PERF_INSTS:     s_axi_rdata <= perf_insts;
                    GPU_REG_PERF_STALL:     s_axi_rdata <= perf_stall;
                    GPU_REG_PERF_BYTES_IN:  s_axi_rdata <= perf_bytes_in;
                    GPU_REG_PERF_BYTES_OUT: s_axi_rdata <= perf_bytes_out;
                    GPU_REG_ID:             s_axi_rdata <= GPU_MAGIC_ID;
                    default:                s_axi_rdata <= 32'd0;
                endcase
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Top-level Command State Machine (DMA H2D, DMA D2H, or Kernel Execution)
    // -------------------------------------------------------------------------
    typedef enum logic [1:0] {
        TOP_IDLE   = 2'd0,
        TOP_H2D    = 2'd1,
        TOP_D2H    = 2'd2,
        TOP_KERNEL = 2'd3
    } top_state_e;

    top_state_e        top_state;
    logic [SPM_AW-1:0] dma_cur_addr;
    logic [31:0]       dma_words_rem;
    logic              start_kernel_pulse;
    /* verilator lint_off UNUSEDSIGNAL */
    logic              kernel_busy;
    /* verilator lint_on UNUSEDSIGNAL */
    logic              kernel_done_pulse;
    logic              inst_retired;

    assign s_axis_tready = (top_state == TOP_H2D) && (dma_words_rem != 32'd0);
    wire s_axis_fire     = s_axis_tvalid && s_axis_tready;

    wire d2h_last_odd    = (top_state == TOP_D2H) && (dma_words_rem == 32'd1);
    assign m_axis_tvalid = (top_state == TOP_D2H) && (dma_words_rem != 32'd0);
    assign m_axis_tkeep  = d2h_last_odd ? 8'h0F : 8'hFF;
    assign m_axis_tlast  = m_axis_tvalid && (dma_words_rem <= 32'd2);
    wire m_axis_fire     = m_axis_tvalid && m_axis_tready;

    wire dma_we0 = s_axis_fire;
    wire dma_we1 = s_axis_fire && (dma_words_rem >= 32'd2);
    wire [SPM_AW-1:0] dma_addr0 = dma_cur_addr;
    wire [SPM_AW-1:0] dma_addr1 = dma_cur_addr + SPM_AW'(1);
    logic [31:0] dma_rdata0, dma_rdata1;

    assign m_axis_tdata = d2h_last_odd ? {32'd0, dma_rdata0} : {dma_rdata1, dma_rdata0};

    always_ff @(posedge clk) begin
        if (!rst_n || soft_rst_pulse) begin
            top_state          <= TOP_IDLE;
            status_busy        <= 1'b0;
            status_done        <= 1'b0;
            dma_cur_addr       <= '0;
            dma_words_rem      <= 32'd0;
            start_kernel_pulse <= 1'b0;
        end else begin
            start_kernel_pulse <= 1'b0;
            unique case (top_state)
                TOP_IDLE: begin
                    if (cmd_start_pulse) begin
                        status_done <= 1'b0;
                        unique case (cfg_cmd)
                            GPU_CMD_CLEAR_PERF: begin
                                status_done <= 1'b1;
                            end
                            GPU_CMD_DMA_H2D: begin
                                dma_cur_addr  <= cfg_dma_addr;
                                dma_words_rem <= cfg_dma_len;
                                status_busy   <= (cfg_dma_len != 32'd0);
                                status_done   <= (cfg_dma_len == 32'd0);
                                top_state     <= (cfg_dma_len != 32'd0) ? TOP_H2D : TOP_IDLE;
                            end
                            GPU_CMD_DMA_D2H: begin
                                dma_cur_addr  <= cfg_dma_addr;
                                dma_words_rem <= cfg_dma_len;
                                status_busy   <= (cfg_dma_len != 32'd0);
                                status_done   <= (cfg_dma_len == 32'd0);
                                top_state     <= (cfg_dma_len != 32'd0) ? TOP_D2H : TOP_IDLE;
                            end
                            GPU_CMD_RUN_KERNEL: begin
                                start_kernel_pulse <= 1'b1;
                                status_busy        <= 1'b1;
                                top_state          <= TOP_KERNEL;
                            end
                            default: status_done <= 1'b1;
                        endcase
                    end
                end

                TOP_H2D: begin
                    if (s_axis_fire) begin
                        if (dma_words_rem <= 32'd2) begin
                            dma_words_rem <= 32'd0;
                            status_busy   <= 1'b0;
                            status_done   <= 1'b1;
                            top_state     <= TOP_IDLE;
                        end else begin
                            dma_cur_addr  <= dma_cur_addr + SPM_AW'(2);
                            dma_words_rem <= dma_words_rem - 32'd2;
                        end
                    end
                end

                TOP_D2H: begin
                    if (m_axis_fire) begin
                        if (dma_words_rem <= 32'd2) begin
                            dma_words_rem <= 32'd0;
                            status_busy   <= 1'b0;
                            status_done   <= 1'b1;
                            top_state     <= TOP_IDLE;
                        end else begin
                            dma_cur_addr  <= dma_cur_addr + SPM_AW'(2);
                            dma_words_rem <= dma_words_rem - 32'd2;
                        end
                    end
                end

                TOP_KERNEL: begin
                    if (kernel_done_pulse) begin
                        status_busy <= 1'b0;
                        status_done <= 1'b1;
                        top_state   <= TOP_IDLE;
                    end
                end
            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Core Submodules: Decoder, Controller, Regfile, Lanes, LSU, Scratchpad
    // -------------------------------------------------------------------------
    logic [IMEM_AW-1:0]             pc;
    wire  [31:0]                    inst_word = imem[pc];

    gpu_opcode_e                    opcode;
    logic [3:0]                     rd, rs1, rs2;
    logic [13:0]                    imm14;
    logic [31:0]                    imm_sext;
    alu_op_e                        alu_op;
    logic                           is_load, is_alu, is_reduce, dec_rf_we;
    /* verilator lint_off UNUSEDSIGNAL */
    logic                           is_store, is_branch;
    /* verilator lint_on UNUSEDSIGNAL */

    gpu_decoder u_decoder (
        .inst_word (inst_word),
        .opcode    (opcode),
        .rd        (rd),
        .rs1       (rs1),
        .rs2       (rs2),
        .imm14     (imm14),
        .imm_sext  (imm_sext),
        .alu_op    (alu_op),
        .is_load   (is_load),
        .is_store  (is_store),
        .is_alu    (is_alu),
        .is_reduce (is_reduce),
        .is_branch (is_branch),
        .rf_we     (dec_rf_we)
    );

    logic                           exec_valid;
    logic [31:0]                    vloop_idx;
    logic [N_LANES-1:0]             lane_active;
    logic                           rf_we;
    logic [N_LANES-1:0]             rf_lane_mask;
    logic [N_LANES-1:0][31:0]       rf_wb_data;

    logic [N_LANES-1:0][31:0]       rs1_data, rs2_data, rd_cur_data;
    logic [N_LANES-1:0][31:0]       alu_out, lsu_rdata;

    gpu_controller #(
        .N_LANES    (N_LANES),
        .IMEM_WORDS (IMEM_WORDS)
    ) u_controller (
        .clk               (clk),
        .rst_n             (rst_n && !soft_rst_pulse),
        .start_kernel      (start_kernel_pulse),
        .cfg_vec_len       (cfg_vec_len),
        .kernel_busy       (kernel_busy),
        .kernel_done_pulse (kernel_done_pulse),
        .inst_retired      (inst_retired),
        .pc                (pc),
        .opcode            (opcode),
        .imm14             (imm14),
        .is_load           (is_load),
        .is_alu            (is_alu),
        .is_reduce         (is_reduce),
        .dec_rf_we         (dec_rf_we),
        .exec_valid        (exec_valid),
        .vloop_idx         (vloop_idx),
        .lane_active       (lane_active),
        .rf_we             (rf_we),
        .rf_lane_mask      (rf_lane_mask),
        .rf_wb_data        (rf_wb_data),
        .alu_out           (alu_out),
        .lsu_rdata         (lsu_rdata),
        .rs1_data          (rs1_data),
        .rd_cur_data       (rd_cur_data)
    );

    gpu_regfile #(
        .N_LANES (N_LANES)
    ) u_regfile (
        .clk          (clk),
        .rst_n        (rst_n && !soft_rst_pulse),
        .rs1_addr     (rs1),
        .rs2_addr     (rs2),
        .rd_addr      (rd),
        .rs1_data     (rs1_data),
        .rs2_data     (rs2_data),
        .rd_cur_data  (rd_cur_data),
        .we           (rf_we),
        .lane_we_mask (rf_lane_mask),
        .wb_data      (rf_wb_data)
    );

    generate
        for (genvar l = 0; l < N_LANES; l++) begin : gen_lanes
            gpu_lane #(
                .LANE_ID (l)
            ) u_lane (
                .opcode    (opcode),
                .alu_op    (alu_op),
                .rs1_val   (rs1_data[l]),
                .rs2_val   (rs2_data[l]),
                .imm_sext  (imm_sext),
                .param0    (cfg_param0),
                .param1    (cfg_param1),
                .vloop_idx (vloop_idx),
                .lane_out  (alu_out[l])
            );
        end
    endgenerate

    logic [N_LANES-1:0]             lsu_we;
    logic [N_LANES-1:0][SPM_AW-1:0] lsu_addr;
    logic [N_LANES-1:0][31:0]       lsu_wdata;

    gpu_lsu #(
        .N_LANES   (N_LANES),
        .SPM_WORDS (SPM_WORDS)
    ) u_lsu (
        .exec_valid  (exec_valid),
        .opcode      (opcode),
        .imm14       (imm14),
        .vloop_idx   (vloop_idx),
        .lane_active (lane_active),
        .rs1_data    (rs1_data),
        .rs2_data    (rs2_data),
        .spm_we      (lsu_we),
        .spm_addr    (lsu_addr),
        .spm_wdata   (lsu_wdata)
    );

    gpu_scratchpad #(
        .N_LANES   (N_LANES),
        .SPM_WORDS (SPM_WORDS)
    ) u_scratchpad (
        .clk        (clk),
        .lsu_we     (lsu_we),
        .lsu_addr   (lsu_addr),
        .lsu_wdata  (lsu_wdata),
        .lsu_rdata  (lsu_rdata),
        .dma_we0    (dma_we0),
        .dma_we1    (dma_we1),
        .dma_addr0  (dma_addr0),
        .dma_addr1  (dma_addr1),
        .dma_wdata0 (s_axis_tdata[31:0]),
        .dma_wdata1 (s_axis_tdata[63:32]),
        .dma_rdata0 (dma_rdata0),
        .dma_rdata1 (dma_rdata1)
    );

    // -------------------------------------------------------------------------
    // Performance Counters
    // -------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n || soft_rst_pulse || (cmd_start_pulse && cfg_cmd == GPU_CMD_CLEAR_PERF)) begin
            perf_cycles    <= 32'd0;
            perf_insts     <= 32'd0;
            perf_stall     <= 32'd0;
            perf_bytes_in  <= 32'd0;
            perf_bytes_out <= 32'd0;
        end else begin
            if (status_busy)
                perf_cycles <= perf_cycles + 32'd1;
            if (inst_retired)
                perf_insts <= perf_insts + 32'd1;
            if ((top_state == TOP_H2D && !s_axis_tvalid) || (top_state == TOP_D2H && !m_axis_tready))
                perf_stall <= perf_stall + 32'd1;
            if (s_axis_fire)
                perf_bytes_in <= perf_bytes_in + (dma_we1 ? 32'd8 : 32'd4);
            if (m_axis_fire)
                perf_bytes_out <= perf_bytes_out + (d2h_last_odd ? 32'd4 : 32'd8);
        end
    end
endmodule
