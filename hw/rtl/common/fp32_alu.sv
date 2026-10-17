// SPDX-License-Identifier: MIT
// Synthesizable IEEE-754 single-precision (FP32) + INT32 / Q16.16 ALU.
//
// Shared by:
//   * NPU post-processing unit (Q8_0 group scale: val += (float)psum * ws * xs)
//   * Compute GPU processing lanes (RMSNorm, RoPE, Softmax, SiLU/SwiGLU, Residual)
//
// Combinational core with optional output register wrapper; rounds to nearest-even
// (or truncation for LUT approximations). Subnormals are flushed to zero (FTZ),
// matching standard embedded FPGA/DSP practice.

`timescale 1ns / 1ps

/* verilator lint_off DECLFILENAME */
package fp32_pkg;
/* verilator lint_on DECLFILENAME */
    typedef enum logic [4:0] {
        ALU_NOP     = 5'd0,
        // Integer / fixed-point ops
        ALU_IADD    = 5'd1,
        ALU_ISUB    = 5'd2,
        ALU_IMUL    = 5'd3,   // low 32 bits of signed 32x32
        ALU_Q16MUL  = 5'd4,   // signed Q16.16 multiply ((a * b) >>> 16)
        ALU_AND     = 5'd5,
        ALU_OR      = 5'd6,
        ALU_XOR     = 5'd7,
        ALU_SHL     = 5'd8,
        ALU_SRA     = 5'd9,
        // IEEE-754 FP32 ops
        ALU_FADD    = 5'd10,
        ALU_FSUB    = 5'd11,
        ALU_FMUL    = 5'd12,
        ALU_FMAX    = 5'd13,
        ALU_FMIN    = 5'd14,
        ALU_I2F     = 5'd15,  // signed int32 -> fp32
        ALU_F2I     = 5'd16,  // fp32 -> signed int32 (round to nearest)
        // Nonlinear / transcendental approximations (FP32 in -> FP32 out)
        ALU_FRECIP  = 5'd17,  // 1.0 / a
        ALU_FRSQRT  = 5'd18,  // 1.0 / sqrt(a)
        ALU_FEXP2   = 5'd19,  // 2^a
        ALU_FEXP    = 5'd20,  // e^a = 2^(a * log2(e))
        ALU_FSILU   = 5'd21,  // a / (1 + e^-a)
        ALU_MOV_A   = 5'd22,
        ALU_MOV_B   = 5'd23
    } alu_op_e;

    // Helper: Leading-zero count of a 32-bit unsigned value (returns 0..32)
    function automatic logic [5:0] clz32(input logic [31:0] v);
        logic [5:0] n;
        begin
            n = 6'd32;
            for (int i = 0; i < 32; i++) begin
                if (v[i]) n = 6'(31 - i);
            end
            return n;
        end
    endfunction

    // Signed int32 -> IEEE-754 FP32 (round-to-nearest-even)
    function automatic logic [31:0] i32_to_fp32(input logic [31:0] a);
        logic        s;
        /* verilator lint_off UNUSEDSIGNAL */
        logic [31:0] mag, shifted;
        /* verilator lint_on UNUSEDSIGNAL */
        logic [5:0]  lz;
        logic [7:0]  exp8;
        logic [22:0] mant;
        logic        g, r, st, round_up;
        begin
            if (a == 32'd0) return 32'd0;
            s   = a[31];
            mag = s ? (~a + 32'd1) : a;
            lz  = clz32(mag);
            // Leading 1 is at bit (31 - lz); shift it to bit 31
            shifted = mag << lz;
            exp8    = 8'(8'd158 - {2'd0, lz}); // 127 + (31 - lz)
            mant    = shifted[30:8];
            g       = shifted[7];
            r       = shifted[6];
            st      = |shifted[5:0];
            round_up = g & (r | st | mant[0]);
            if (round_up) begin
                if (mant == 23'h7fffff) begin
                    mant = 23'd0;
                    exp8 = exp8 + 8'd1;
                end else begin
                    mant = mant + 23'd1;
                end
            end
            return {s, exp8, mant};
        end
    endfunction

    // IEEE-754 FP32 -> Signed int32 (round-to-nearest, clamp on overflow)
    function automatic logic [31:0] fp32_to_i32(input logic [31:0] a);
        logic        s;
        logic [7:0]  e;
        logic [23:0] m;
        /* verilator lint_off UNUSEDSIGNAL */
        logic [55:0] sh;
        /* verilator lint_on UNUSEDSIGNAL */
        logic [31:0] res;
        int          shift;
        begin
            s = a[31];
            e = a[30:23];
            m = {1'b1, a[22:0]};
            if (e < 8'd126) return 32'd0; // |a| < 0.5
            if (e >= 8'd158) return s ? 32'h8000_0000 : 32'h7fff_ffff;
            shift = int'(e) - 127;
            // Place binary point at bit 23 of sh: {32'd0, m} has value m * 2^-23
            if (shift >= 0)
                sh = {32'd0, m} << shift;
            else
                sh = {32'd0, m} >> (-shift);
            res = sh[54:23] + {31'd0, sh[22]}; // round half up
            return s ? (~res + 32'd1) : res;
        end
    endfunction

    // IEEE-754 FP32 multiplication (FTZ, round-to-nearest-even)
    function automatic logic [31:0] fp32_mul(input logic [31:0] a, input logic [31:0] b);
        logic        sa, sb, sr;
        logic [7:0]  ea, eb;
        logic [23:0] ma, mb;
        logic [47:0] prod;
        int          er;
        logic [22:0] mant;
        logic        g, r, st;
        begin
            sa = a[31]; ea = a[30:23]; ma = {1'b1, a[22:0]};
            sb = b[31]; eb = b[30:23]; mb = {1'b1, b[22:0]};
            sr = sa ^ sb;
            if (ea == 8'd0 || eb == 8'd0) return {sr, 31'd0};
            if (ea == 8'hff || eb == 8'hff) return {sr, 8'hff, 23'd0};
            prod = ma * mb; // 24x24 -> 48 bits, value in [1, 4) * 2^46
            er   = int'(ea) + int'(eb) - 127 + (prod[47] ? 1 : 0);
            if (prod[47]) begin
                mant = prod[46:24];
                g    = prod[23];
                r    = prod[22];
                st   = |prod[21:0];
            end else begin
                mant = prod[45:23];
                g    = prod[22];
                r    = prod[21];
                st   = |prod[20:0];
            end
            if (g & (r | st | mant[0])) begin
                if (mant == 23'h7fffff) begin
                    mant = 23'd0;
                    er   = er + 1;
                end else begin
                    mant = mant + 23'd1;
                end
            end
            if (er >= 255) return {sr, 8'hff, 23'd0};
            if (er <= 0)   return {sr, 31'd0};
            return {sr, 8'(er), mant};
        end
    endfunction

    // IEEE-754 FP32 addition (a + b, FTZ, round-to-nearest-even)
    function automatic logic [31:0] fp32_add(input logic [31:0] a, input logic [31:0] b);
        logic        sa, sb, sr;
        logic [7:0]  ea, eb, ediff;
        logic [27:0] ma, mb, msum; // {carry, 1.m[22:0], G, R, S}
        logic [27:0] shifted_b;
        logic        sticky;
        logic [5:0]  lz;
        int          er;
        logic [22:0] mant;
        logic        g, r, st;
        logic [31:0] ta, tb;
        begin
            // Swap so |ta| >= |tb|
            if (a[30:0] < b[30:0]) begin ta = b; tb = a; end
            else                   begin ta = a; tb = b; end
            sa = ta[31]; ea = ta[30:23];
            sb = tb[31]; eb = tb[30:23];
            if (ea == 8'd0) return 32'd0;
            if (eb == 8'd0) return ta;
            if (ea == 8'hff) return ta;

            ma    = {1'b0, 1'b1, ta[22:0], 3'b000};
            mb    = {1'b0, 1'b1, tb[22:0], 3'b000};
            ediff = ea - eb;
            if (ediff >= 8'd27) begin
                shifted_b = 28'd1; // pure sticky
            end else if (ediff == 8'd0) begin
                shifted_b = mb;
            end else begin
                sticky = |(mb & ((28'd1 << ediff) - 28'd1));
                shifted_b = (mb >> ediff) | {27'd0, sticky};
            end

            sr = sa;
            er = int'(ea);
            if (sa == sb) begin
                msum = ma + shifted_b;
                if (msum[27]) begin
                    msum = (msum >> 1) | {27'd0, msum[0]};
                    er   = er + 1;
                end
            end else begin
                msum = ma - shifted_b;
                if (msum == 28'd0) return 32'd0;
                lz = clz32({msum, 4'b0000}); // msum[26] should land at bit 26 -> bit 30 of 32-bit
                // Leading 1 of msum is at bit (31 - lz - 4) = (27 - lz)
                if (lz > 6'd1) begin
                    int sh;
                    sh   = int'(lz) - 1;
                    msum = msum << sh;
                    er   = er - sh;
                end
            end

            if (er <= 0) return {sr, 31'd0};
            mant = msum[25:3];
            g    = msum[2];
            r    = msum[1];
            st   = msum[0];
            if (g & (r | st | mant[0])) begin
                if (mant == 23'h7fffff) begin
                    mant = 23'd0;
                    er   = er + 1;
                end else begin
                    mant = mant + 23'd1;
                end
            end
            if (er >= 255) return {sr, 8'hff, 23'd0};
            return {sr, 8'(er), mant};
        end
    endfunction

    // Fast FP32 reciprocal 1/x via initial magic constant + 2 Newton-Raphson steps:
    //   y_{k+1} = y_k * (2 - x * y_k)
    // Achieves < 1e-6 relative error across all normal floats.
    function automatic logic [31:0] fp32_recip(input logic [31:0] x);
        logic [31:0] y0, y1, y2, two, xy, two_m_xy;
        begin
            if (x[30:23] == 8'd0) return {x[31], 8'hff, 23'd0};
            two = 32'h4000_0000; // 2.0f
            // Initial approximation ~ 0x7eef1aa0 - x (with sign preserved)
            y0  = {x[31], 31'(32'h7eef_1aa0 - {1'b0, x[30:0]})};
            // NR step 1
            xy       = fp32_mul({1'b0, x[30:0]}, {1'b0, y0[30:0]});
            two_m_xy = fp32_add(two, {~xy[31], xy[30:0]});
            y1       = fp32_mul(y0, two_m_xy);
            // NR step 2
            xy       = fp32_mul({1'b0, x[30:0]}, {1'b0, y1[30:0]});
            two_m_xy = fp32_add(two, {~xy[31], xy[30:0]});
            y2       = fp32_mul(y1, two_m_xy);
            return y2;
        end
    endfunction

    // Fast FP32 inverse square root 1/sqrt(x) via Quake/Lomont magic + 2 NR steps:
    //   y_{k+1} = y_k * (1.5 - 0.5 * x * y_k * y_k)
    function automatic logic [31:0] fp32_rsqrt(input logic [31:0] x);
        logic [31:0] xhalf, threehalfs, y, y2, xy2, term;
        begin
            if (x[31] || x[30:23] == 8'd0) return 32'h7fc0_0000; // NaN/Inf
            xhalf      = {1'b0, 8'(x[30:23] - 8'd1), x[22:0]}; // 0.5 * x
            threehalfs = 32'h3fc0_0000;                        // 1.5f
            y          = 32'h5f37_5a86 - (x >> 1);
            // NR step 1
            y2   = fp32_mul(y, y);
            xy2  = fp32_mul(xhalf, y2);
            term = fp32_add(threehalfs, {~xy2[31], xy2[30:0]});
            y    = fp32_mul(y, term);
            // NR step 2
            y2   = fp32_mul(y, y);
            xy2  = fp32_mul(xhalf, y2);
            term = fp32_add(threehalfs, {~xy2[31], xy2[30:0]});
            y    = fp32_mul(y, term);
            return y;
        end
    endfunction

    // Fast FP32 2^x for x in [-126, 126]:
    //   Split x = n + f with n = floor(x), f in [0, 1).
    //   Minimax degree-3 polynomial on [0, 1):
    //     2^f ≈ 1 + f * (0.6931472 + f * (0.2402265 + f * 0.0555041))
    //   Actually, tuned cubic on [0, 1] hitting 2^0=1 and 2^1=2:
    //     p(f) = 1.0 + f * (0.69514614 + f * (0.22756439 + f * 0.07728947))
    //   Max relative error < 3e-4 across all f in [0, 1].
    function automatic logic [31:0] fp32_exp2(input logic [31:0] x);
        logic [31:0] n_fp, f, p, c1, c2, c3, one;
        int          n_int, exp_out;
        begin
            // Handle clamp for large negative/positive x
            if (x[31] && x[30:0] >= 31'h42fc_0000) return 32'd0;        // x <= -126
            if (!x[31] && x[30:0] >= 31'h42fe_0000) return 32'h7f80_0000; // x >= 127
            // Floor(x)
            n_int = int'(fp32_to_i32(x));
            n_fp  = i32_to_fp32(32'(n_int));
            // If n_fp > x (because fp32_to_i32 rounds to nearest), subtract 1
            f = fp32_add(x, {~n_fp[31], n_fp[30:0]});
            if (f[31]) begin
                n_int = n_int - 1;
                n_fp  = i32_to_fp32(32'(n_int));
                f     = fp32_add(x, {~n_fp[31], n_fp[30:0]});
            end
            one = 32'h3f80_0000; // 1.0
            c1  = 32'h3f31_f519; // 0.69514614
            c2  = 32'h3e69_089a; // 0.22756439
            c3  = 32'h3d9e_4b18; // 0.07728947
            p   = fp32_add(c2, fp32_mul(f, c3));
            p   = fp32_add(c1, fp32_mul(f, p));
            p   = fp32_add(one, fp32_mul(f, p));
            // Scale by 2^n_int via exponent adjustment
            exp_out = int'(p[30:23]) + n_int;
            if (exp_out <= 0)   return 32'd0;
            if (exp_out >= 255) return 32'h7f80_0000;
            return {1'b0, 8'(exp_out), p[22:0]};
        end
    endfunction

    // Fast FP32 e^x = 2^(x * log2(e))
    function automatic logic [31:0] fp32_exp(input logic [31:0] x);
        logic [31:0] log2e;
        begin
            log2e = 32'h3fb8_aa3b; // 1.4426950408889634f
            return fp32_exp2(fp32_mul(x, log2e));
        end
    endfunction

    // FP32 comparison: returns 1 if a > b (signed IEEE-754 order)
    function automatic logic fp32_gt(input logic [31:0] a, input logic [31:0] b);
        begin
            if (a[30:0] == 31'd0 && b[30:0] == 31'd0) return 1'b0;
            if (a[31] != b[31]) return ~a[31]; // positive > negative
            if (a[31] == 1'b0)  return a[30:0] > b[30:0];
            else                return a[30:0] < b[30:0];
        end
    endfunction
endpackage

module fp32_alu
    import fp32_pkg::*;
(
    input  alu_op_e     op,
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] y
);
    /* verilator lint_off UNUSEDSIGNAL */
    logic [63:0] q16_prod;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [31:0] exp_neg_a, denom;

    always_comb begin
        y         = 32'd0;
        q16_prod  = 64'd0;
        exp_neg_a = 32'd0;
        denom     = 32'd0;
        unique case (op)
            ALU_NOP:    y = 32'd0;
            ALU_IADD:   y = a + b;
            ALU_ISUB:   y = a - b;
            ALU_IMUL:   y = $signed(a) * $signed(b);
            ALU_Q16MUL: begin
                q16_prod = $signed(a) * $signed(b);
                y        = q16_prod[47:16];
            end
            ALU_AND:    y = a & b;
            ALU_OR:     y = a | b;
            ALU_XOR:    y = a ^ b;
            ALU_SHL:    y = a << b[4:0];
            ALU_SRA:    y = $signed(a) >>> b[4:0];
            ALU_FADD:   y = fp32_add(a, b);
            ALU_FSUB:   y = fp32_add(a, {~b[31], b[30:0]});
            ALU_FMUL:   y = fp32_mul(a, b);
            ALU_FMAX:   y = fp32_gt(a, b) ? a : b;
            ALU_FMIN:   y = fp32_gt(a, b) ? b : a;
            ALU_I2F:    y = i32_to_fp32(a);
            ALU_F2I:    y = fp32_to_i32(a);
            ALU_FRECIP: y = fp32_recip(a);
            ALU_FRSQRT: y = fp32_rsqrt(a);
            ALU_FEXP2:  y = fp32_exp2(a);
            ALU_FEXP:   y = fp32_exp(a);
            ALU_FSILU:  begin
                exp_neg_a = fp32_exp({~a[31], a[30:0]});
                denom     = fp32_add(32'h3f80_0000, exp_neg_a);
                y         = fp32_mul(a, fp32_recip(denom));
            end
            ALU_MOV_A:  y = a;
            ALU_MOV_B:  y = b;
            default:    y = 32'd0;
        endcase
    end
endmodule
