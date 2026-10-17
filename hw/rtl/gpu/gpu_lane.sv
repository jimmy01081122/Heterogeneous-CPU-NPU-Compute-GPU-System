// SPDX-License-Identifier: MIT
// Compute GPU Arithmetic Processing Lane (gpu_lane).
//
// Each lane contains an instance of the shared IEEE-754 FP32 / INT32 / Q16.16
// ALU plus source operand muxing for immediates, scalar parameters, and SIMT
// element indices (`vloop_idx + LANE_ID`).

`timescale 1ns / 1ps

module gpu_lane
    import gpu_pkg::*;
    import fp32_pkg::*;
#(
    parameter int LANE_ID = 0
) (
    input  gpu_opcode_e opcode,
    input  alu_op_e     alu_op,
    input  logic [31:0] rs1_val,
    input  logic [31:0] rs2_val,
    input  logic [31:0] imm_sext,
    input  logic [31:0] param0,
    input  logic [31:0] param1,
    input  logic [31:0] vloop_idx,
    output logic [31:0] lane_out
);
    logic [31:0] op_a, op_b;

    always_comb begin
        op_a = rs1_val;
        unique case (opcode)
            OP_LDI:      op_b = imm_sext;
            OP_LD_PARAM: op_b = imm_sext[0] ? param1 : param0;
            OP_LD_LANE:  op_b = vloop_idx + 32'(LANE_ID);
            default:     op_b = rs2_val;
        endcase
    end

    fp32_alu u_alu (
        .op (alu_op),
        .a  (op_a),
        .b  (op_b),
        .y  (lane_out)
    );
endmodule
