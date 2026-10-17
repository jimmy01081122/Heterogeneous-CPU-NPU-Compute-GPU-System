// SPDX-License-Identifier: MIT
// Compute GPU Instruction Decoder (gpu_decoder).
//
// Decodes a 32-bit instruction word into control signals for the Parallel
// Execution Controller, Vector Register File, Arithmetic Lanes, and LSU.

`timescale 1ns / 1ps

module gpu_decoder
    import gpu_pkg::*;
    import fp32_pkg::*;
(
    input  logic [31:0] inst_word,
    output gpu_opcode_e opcode,
    output logic [3:0]  rd,
    output logic [3:0]  rs1,
    output logic [3:0]  rs2,
    output logic [13:0] imm14,
    output logic [31:0] imm_sext,
    output alu_op_e     alu_op,
    output logic        is_load,
    output logic        is_store,
    output logic        is_alu,
    output logic        is_reduce,
    output logic        is_branch,
    output logic        rf_we
);
    assign opcode   = gpu_opcode_e'(inst_word[31:26]);
    assign rd       = inst_word[25:22];
    assign rs1      = inst_word[21:18];
    assign rs2      = inst_word[17:14];
    assign imm14    = inst_word[13:0];
    assign imm_sext = {{18{imm14[13]}}, imm14};

    always_comb begin
        alu_op    = ALU_NOP;
        is_load   = 1'b0;
        is_store  = 1'b0;
        is_alu    = 1'b0;
        is_reduce = 1'b0;
        is_branch = 1'b0;
        rf_we     = 1'b0;

        unique case (opcode)
            OP_VLD, OP_SLD, OP_VLD_ABS: begin
                is_load = 1'b1;
                rf_we   = 1'b1;
            end
            OP_VST, OP_SST, OP_VST_ABS: begin
                is_store = 1'b1;
            end
            OP_LDI, OP_LD_PARAM, OP_LD_LANE: begin
                is_alu = 1'b1;
                rf_we  = 1'b1;
                alu_op = ALU_MOV_B;
            end
            OP_MOV: begin
                is_alu = 1'b1;
                rf_we  = 1'b1;
                alu_op = ALU_MOV_A;
            end
            OP_IADD:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_IADD;   end
            OP_ISUB:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_ISUB;   end
            OP_IMUL:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_IMUL;   end
            OP_Q16MUL: begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_Q16MUL; end
            OP_AND:    begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_AND;    end
            OP_OR:     begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_OR;     end
            OP_XOR:    begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_XOR;    end
            OP_SHL:    begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_SHL;    end
            OP_SRA:    begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_SRA;    end
            OP_FADD:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FADD;   end
            OP_FSUB:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FSUB;   end
            OP_FMUL:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FMUL;   end
            OP_FMAX:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FMAX;   end
            OP_FMIN:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FMIN;   end
            OP_I2F:    begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_I2F;    end
            OP_F2I:    begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_F2I;    end
            OP_FRECIP: begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FRECIP; end
            OP_FRSQRT: begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FRSQRT; end
            OP_FEXP:   begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FEXP;   end
            OP_FSILU:  begin is_alu = 1'b1; rf_we = 1'b1; alu_op = ALU_FSILU;  end
            OP_RADD, OP_RMAX: begin
                is_reduce = 1'b1;
                rf_we     = 1'b1;
            end
            OP_VRESET, OP_VSTEP, OP_JMP, OP_HALT: begin
                is_branch = 1'b1;
            end
            default: ;
        endcase
    end
endmodule
