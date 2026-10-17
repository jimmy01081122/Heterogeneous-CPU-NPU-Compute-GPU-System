// SPDX-License-Identifier: MIT
// Parameterizable N_ROWS x N_LANES signed INT8 MAC array with per-row adder
// tree and signed INT32 group accumulator.
//
// In each valid cycle (`valid_in`), row `r` (0..N_ROWS-1) computes:
//     step_sum[r] = sum_{c=0..N_LANES-1} (lane_mask[c] ? wgt_in[r][c] * act_in[c] : 0)
// and accumulates `step_sum[r]` into `g_acc[r]` (or initializes from 0 when
// `first_step` is asserted). When `last_step` is asserted, `psum_valid` pulses
// on the next cycle with the completed group partial sums `psum_out[0..N_ROWS-1]`.

`timescale 1ns / 1ps

module npu_mac_array #(
    parameter int N_ROWS  = 8,
    parameter int N_LANES = 8
) (
    input  logic                                    clk,
    input  logic                                    rst_n,
    input  logic                                    valid_in,
    input  logic                                    first_step,
    input  logic                                    last_step,
    input  logic [N_LANES-1:0]                      lane_mask,
    input  logic signed [N_LANES-1:0][7:0]          act_in,
    input  logic signed [N_ROWS-1:0][N_LANES-1:0][7:0] wgt_in,
    output logic                                    psum_valid,
    output logic signed [N_ROWS-1:0][31:0]          psum_out
);
    logic signed [N_ROWS-1:0][N_LANES-1:0][15:0] prod;
    logic signed [N_ROWS-1:0][31:0]              step_sum;
    logic signed [N_ROWS-1:0][31:0]              next_acc;
    logic signed [N_ROWS-1:0][31:0]              g_acc;

    always_comb begin
        for (int r = 0; r < N_ROWS; r++) begin
            step_sum[r] = 32'sd0;
            for (int c = 0; c < N_LANES; c++) begin
                if (lane_mask[c])
                    prod[r][c] = 16'($signed(wgt_in[r][c]) * $signed(act_in[c]));
                else
                    prod[r][c] = 16'sd0;
                step_sum[r] = step_sum[r] + 32'($signed(prod[r][c]));
            end
            if (first_step)
                next_acc[r] = step_sum[r];
            else
                next_acc[r] = g_acc[r] + step_sum[r];
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            psum_valid <= 1'b0;
            for (int r = 0; r < N_ROWS; r++) begin
                g_acc[r]    <= 32'sd0;
                psum_out[r] <= 32'sd0;
            end
        end else begin
            psum_valid <= valid_in & last_step;
            if (valid_in) begin
                for (int r = 0; r < N_ROWS; r++) begin
                    g_acc[r] <= next_acc[r];
                    if (last_step)
                        psum_out[r] <= next_acc[r];
                end
            end
        end
    end
endmodule
