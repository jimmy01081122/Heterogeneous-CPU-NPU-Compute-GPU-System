// SPDX-License-Identifier: MIT
// Compute GPU Vector Register File (gpu_regfile).
//
// 16 vector registers (v0 .. v15), each holding N_LANES x 32-bit words.
// Provides 3 combinational read ports (rs1, rs2, and rd for accumulative
// reductions) and 1 per-lane masked synchronous write port.

`timescale 1ns / 1ps

module gpu_regfile #(
    parameter int N_LANES = 4,
    parameter int N_REGS  = 16
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic [3:0]                  rs1_addr,
    input  logic [3:0]                  rs2_addr,
    input  logic [3:0]                  rd_addr,
    output logic [N_LANES-1:0][31:0]    rs1_data,
    output logic [N_LANES-1:0][31:0]    rs2_data,
    output logic [N_LANES-1:0][31:0]    rd_cur_data,
    input  logic                        we,
    input  logic [N_LANES-1:0]          lane_we_mask,
    input  logic [N_LANES-1:0][31:0]    wb_data
);
    logic [N_LANES-1:0][31:0] vrf [0:N_REGS-1];

    always_comb begin
        rs1_data    = vrf[rs1_addr];
        rs2_data    = vrf[rs2_addr];
        rd_cur_data = vrf[rd_addr];
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int r = 0; r < N_REGS; r++) begin
                vrf[r] <= '0;
            end
        end else if (we) begin
            for (int l = 0; l < N_LANES; l++) begin
                if (lane_we_mask[l])
                    vrf[rd_addr][l] <= wb_data[l];
            end
        end
    end
endmodule
