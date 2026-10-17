// SPDX-License-Identifier: MIT
// Compute GPU Parallel Execution Controller (gpu_controller).
//
// Coordinates:
//   * Program Counter (PC) fetch & branch control (`OP_JMP`, `OP_VRESET`,
//     `OP_VSTEP`, `OP_HALT`).
//   * Hardware SIMT strip-mining loop index (`vloop_idx`) and per-lane
//     predicate mask (`lane_active[l] = (vloop_idx + l < cfg_vec_len)`).
//   * Cross-lane FP32 reduction tree (`OP_RADD`, `OP_RMAX`) across active lanes,
//     broadcasting the reduced scalar back to all lanes of `rd`.
//   * Writeback selection (ALU lanes vs LSU loads vs cross-lane reductions).

`timescale 1ns / 1ps

module gpu_controller
    import gpu_pkg::*;
    import fp32_pkg::*;
#(
    parameter int N_LANES    = 4,
    parameter int IMEM_WORDS = 256,
    parameter int PC_W       = $clog2(IMEM_WORDS)
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start_kernel,
    input  logic [31:0]                 cfg_vec_len,
    output logic                        kernel_busy,
    output logic                        kernel_done_pulse,
    output logic                        inst_retired,

    // Instruction memory interface
    output logic [PC_W-1:0]             pc,
    input  gpu_opcode_e                 opcode,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [13:0]                 imm14,
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic                        is_load,
    input  logic                        is_alu,
    input  logic                        is_reduce,
    input  logic                        dec_rf_we,

    // SIMT state to Lanes / LSU / Regfile
    output logic                        exec_valid,
    output logic [31:0]                 vloop_idx,
    output logic [N_LANES-1:0]          lane_active,
    output logic                        rf_we,
    output logic [N_LANES-1:0]          rf_lane_mask,
    output logic [N_LANES-1:0][31:0]    rf_wb_data,

    // Data inputs for writeback & cross-lane reductions
    input  logic [N_LANES-1:0][31:0]    alu_out,
    input  logic [N_LANES-1:0][31:0]    lsu_rdata,
    input  logic [N_LANES-1:0][31:0]    rs1_data,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [N_LANES-1:0][31:0]    rd_cur_data
    /* verilator lint_on UNUSEDSIGNAL */
);
    // Compute per-lane SIMT active mask
    always_comb begin
        for (int l = 0; l < N_LANES; l++) begin
            lane_active[l] = ((vloop_idx + 32'(l)) < cfg_vec_len);
        end
    end

    assign exec_valid   = kernel_busy && (opcode != OP_HALT);
    assign inst_retired = kernel_busy;

    // Cross-lane FP32 reduction (OP_RADD accumulates into rd_cur_data[0];
    // OP_RMAX takes max with rd_cur_data[0]; broadcasts result to all lanes)
    logic [31:0] reduce_acc;
    always_comb begin
        reduce_acc = rd_cur_data[0];
        for (int l = 0; l < N_LANES; l++) begin
            if (lane_active[l]) begin
                if (opcode == OP_RADD)
                    reduce_acc = fp32_add(reduce_acc, rs1_data[l]);
                else if (opcode == OP_RMAX)
                    reduce_acc = fp32_gt(rs1_data[l], reduce_acc) ? rs1_data[l] : reduce_acc;
            end
        end
    end

    // Writeback mux & lane mask
    // Note: scalar/broadcast ops (LDI, LD_PARAM, SLD, RADD, RMAX) write all lanes
    // even if vloop_idx is at the tail; vector element ops respect lane_active.
    wire is_broadcast_wb = (opcode == OP_LDI) || (opcode == OP_LD_PARAM) ||
                           (opcode == OP_SLD) || is_reduce;

    assign rf_we        = exec_valid && dec_rf_we;
    assign rf_lane_mask = is_broadcast_wb ? '1 : lane_active;

    always_comb begin
        for (int l = 0; l < N_LANES; l++) begin
            if (is_load)
                rf_wb_data[l] = lsu_rdata[l];
            else if (is_reduce)
                rf_wb_data[l] = reduce_acc;
            else if (is_alu)
                rf_wb_data[l] = alu_out[l];
            else
                rf_wb_data[l] = 32'd0;
        end
    end

    wire [31:0] next_vloop = vloop_idx + 32'(N_LANES);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            kernel_busy       <= 1'b0;
            kernel_done_pulse <= 1'b0;
            pc                <= '0;
            vloop_idx         <= 32'd0;
        end else begin
            kernel_done_pulse <= 1'b0;
            if (start_kernel) begin
                kernel_busy <= 1'b1;
                pc          <= '0;
                vloop_idx   <= 32'd0;
            end else if (kernel_busy) begin
                unique case (opcode)
                    OP_HALT: begin
                        kernel_busy       <= 1'b0;
                        kernel_done_pulse <= 1'b1;
                    end
                    OP_JMP: begin
                        pc <= imm14[PC_W-1:0];
                    end
                    OP_VRESET: begin
                        vloop_idx <= 32'd0;
                        pc        <= pc + PC_W'(1);
                    end
                    OP_VSTEP: begin
                        if (next_vloop < cfg_vec_len) begin
                            vloop_idx <= next_vloop;
                            pc        <= imm14[PC_W-1:0];
                        end else begin
                            vloop_idx <= 32'd0;
                            pc        <= pc + PC_W'(1);
                        end
                    end
                    default: begin
                        pc <= pc + PC_W'(1);
                    end
                endcase
            end
        end
    end
endmodule
