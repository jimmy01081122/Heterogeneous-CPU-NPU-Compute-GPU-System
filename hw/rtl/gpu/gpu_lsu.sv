// SPDX-License-Identifier: MIT
// Compute GPU Load/Store Unit (gpu_lsu).
//
// Computes per-lane Scratchpad Memory addresses and SIMT-predicated write
// enables for vector, scalar-broadcast, and indexed gather/scatter memory ops:
//   * OP_VLD     : vd[l] <= SPM[imm14 + vloop_idx + l]
//   * OP_VST     : SPM[imm14 + vloop_idx + l] <= vs2[l]        (if lane_active[l])
//   * OP_SLD     : vd[l] <= SPM[imm14]                         (broadcast scalar)
//   * OP_SST     : SPM[imm14] <= vs2[0]                        (lane 0 only)
//   * OP_VLD_ABS : vd[l] <= SPM[imm14 + vs1[l]]                (indexed gather)
//   * OP_VST_ABS : SPM[imm14 + vs1[l]] <= vs2[l]               (indexed scatter, if lane_active[l])

`timescale 1ns / 1ps

module gpu_lsu
    import gpu_pkg::*;
#(
    parameter int N_LANES   = 4,
    parameter int SPM_WORDS = 4096,
    parameter int AW        = $clog2(SPM_WORDS)
) (
    input  logic                        exec_valid,
    input  gpu_opcode_e                 opcode,
    input  logic [13:0]                 imm14,
    input  logic [31:0]                 vloop_idx,
    input  logic [N_LANES-1:0]          lane_active,
    input  logic [N_LANES-1:0][31:0]    rs1_data,
    input  logic [N_LANES-1:0][31:0]    rs2_data,
    output logic [N_LANES-1:0]          spm_we,
    output logic [N_LANES-1:0][AW-1:0]  spm_addr,
    output logic [N_LANES-1:0][31:0]    spm_wdata
);
    /* verilator lint_off UNUSEDSIGNAL */
    logic [N_LANES-1:0][31:0] full_addr;
    /* verilator lint_on UNUSEDSIGNAL */

    always_comb begin
        for (int l = 0; l < N_LANES; l++) begin
            full_addr[l] = 32'd0;
            spm_we[l]    = 1'b0;
            spm_wdata[l] = rs2_data[l];

            unique case (opcode)
                OP_VLD: begin
                    full_addr[l] = {18'd0, imm14} + vloop_idx + 32'(l);
                end
                OP_VST: begin
                    full_addr[l] = {18'd0, imm14} + vloop_idx + 32'(l);
                    spm_we[l]    = exec_valid && lane_active[l];
                end
                OP_SLD: begin
                    full_addr[l] = {18'd0, imm14};
                end
                OP_SST: begin
                    full_addr[l] = {18'd0, imm14};
                    spm_we[l]    = exec_valid && (l == 0);
                end
                OP_VLD_ABS: begin
                    full_addr[l] = {18'd0, imm14} + rs1_data[l];
                end
                OP_VST_ABS: begin
                    full_addr[l] = {18'd0, imm14} + rs1_data[l];
                    spm_we[l]    = exec_valid && lane_active[l];
                end
                default: ;
            endcase

            spm_addr[l] = full_addr[l][AW-1:0];
        end
    end
endmodule
