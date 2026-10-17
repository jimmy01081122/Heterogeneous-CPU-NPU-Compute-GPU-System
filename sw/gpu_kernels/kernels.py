#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Assembler and LLM Vector Kernel Library for the Programmable Compute GPU.

Encodes the 32-bit SIMD/SIMT ISA defined in `hw/rtl/gpu/gpu_pkg.sv`:
    [31:26] opcode (6b) | [25:22] rd (4b) | [21:18] rs1 (4b) | [17:14] rs2 (4b) | [13:0] imm14 (14b)

Provides ready-to-run kernels for:
    * RMSNorm       (sum-of-squares reduction + rsqrt + elementwise scale)
    * Softmax       (max reduction + exp & sum reduction + reciprocal normalize)
    * SiLU / SwiGLU (silu(hb[i]) * hb2[i])
    * RoPE          (in-place rotary positional embedding via gather/scatter)
    * Residual Add  (x[i] += xb[i])
    * Q8_0 Epilogue (dequantize INT32 partial sums: float(psum[i]) * ws[i] * xs)

Run directly to emit `sw/gpu_kernels/gpu_kernels_bin.h` for the C runtime/driver:
    python sw/gpu_kernels/kernels.py
"""
from __future__ import annotations

import os

OPCODES: dict[str, int] = {
    "NOP":      0x00,
    "VLD":      0x01,
    "VST":      0x02,
    "SLD":      0x03,
    "SST":      0x04,
    "VLD_ABS":  0x05,
    "VST_ABS":  0x06,
    "LDI":      0x08,
    "LD_PARAM": 0x09,
    "LD_LANE":  0x0A,
    "MOV":      0x0B,
    "IADD":     0x10,
    "ISUB":     0x11,
    "IMUL":     0x12,
    "Q16MUL":   0x13,
    "AND":      0x14,
    "OR":       0x15,
    "XOR":      0x16,
    "SHL":      0x17,
    "SRA":      0x18,
    "FADD":     0x20,
    "FSUB":     0x21,
    "FMUL":     0x22,
    "FMAX":     0x23,
    "FMIN":     0x24,
    "I2F":      0x25,
    "F2I":      0x26,
    "FRECIP":   0x28,
    "FRSQRT":   0x29,
    "FEXP":     0x2A,
    "FSILU":    0x2B,
    "RADD":     0x30,
    "RMAX":     0x31,
    "VRESET":   0x38,
    "VSTEP":    0x39,
    "JMP":      0x3E,
    "HALT":     0x3F,
}


def enc(op: str, rd: int = 0, rs1: int = 0, rs2: int = 0, imm: int = 0) -> int:
    """Encode a single 32-bit GPU instruction."""
    opc = OPCODES[op] & 0x3F
    return (
        (opc << 26)
        | ((rd & 0xF) << 22)
        | ((rs1 & 0xF) << 18)
        | ((rs2 & 0xF) << 14)
        | (imm & 0x3FFF)
    )


def build_rmsnorm_kernel(x_base: int = 0, w_base: int = 1024, o_base: int = 2048) -> list[int]:
    """
    RMSNorm kernel:
        PARAM0 = float32(1.0 / size)
        PARAM1 = float32(1e-5)
        SPM[x_base .. x_base+N-1] : input x[0..N-1]
        SPM[w_base .. w_base+N-1] : weight[0..N-1]
        SPM[o_base .. o_base+N-1] : output o[0..N-1]
    """
    prog: list[int] = [
        enc("LDI", rd=0, imm=0),                # 0: v0 = 0.0f (sum-of-squares accumulator)
        enc("VRESET"),                          # 1: vloop_idx = 0
        # L_SUMSQ (PC = 2):
        enc("VLD", rd=1, imm=x_base),           # 2: v1[l] = x[vloop_idx + l]
        enc("FMUL", rd=2, rs1=1, rs2=1),        # 3: v2[l] = x[i]^2
        enc("RADD", rd=0, rs1=2),               # 4: v0[all] += sum_{active l} v2[l]
        enc("VSTEP", imm=2),                    # 5: loop to PC=2 while vloop_idx < N
        # Scalar inv-rms:
        enc("LD_PARAM", rd=3, imm=0),           # 6: v3 = 1.0f / N
        enc("LD_PARAM", rd=4, imm=1),           # 7: v4 = 1e-5f
        enc("FMUL", rd=0, rs1=0, rs2=3),        # 8: v0 = mean(x^2)
        enc("FADD", rd=0, rs1=0, rs2=4),        # 9: v0 = mean(x^2) + 1e-5f
        enc("FRSQRT", rd=0, rs1=0),             # 10: v0 = 1.0f / sqrtf(mean(x^2) + 1e-5f)
        # L_SCALE (PC = 12):
        enc("VRESET"),                          # 11: vloop_idx = 0
        enc("VLD", rd=1, imm=x_base),           # 12: v1[l] = x[vloop_idx + l]
        enc("VLD", rd=2, imm=w_base),           # 13: v2[l] = w[vloop_idx + l]
        enc("FMUL", rd=1, rs1=1, rs2=0),        # 14: v1[l] = ss * x[i]
        enc("FMUL", rd=1, rs1=1, rs2=2),        # 15: v1[l] = w[i] * (ss * x[i])
        enc("VST", rs2=1, imm=o_base),          # 16: o[vloop_idx + l] = v1[l]
        enc("VSTEP", imm=12),                   # 17: loop to PC=12
        enc("HALT"),                            # 18
    ]
    return prog


def build_softmax_kernel(x_base: int = 0) -> list[int]:
    """
    In-place numerically stable Softmax over SPM[x_base .. x_base+N-1]:
        Pass 1: max_val = max_i x[i]
        Pass 2: x[i] = expf(x[i] - max_val), sum += x[i]
        Pass 3: x[i] *= (1.0f / sum)
    """
    return [
        enc("SLD", rd=0, imm=x_base),           # 0: v0[all] = x[0]
        enc("VRESET"),                          # 1: vloop_idx = 0
        # L_MAX (PC = 2):
        enc("VLD", rd=1, imm=x_base),           # 2: v1[l] = x[vloop_idx + l]
        enc("RMAX", rd=0, rs1=1),               # 3: v0[all] = max(v0[0], max_{active} v1[l])
        enc("VSTEP", imm=2),                    # 4: loop to PC=2
        # Pass 2: exp and sum
        enc("LDI", rd=2, imm=0),                # 5: v2[all] = 0.0f (sum accumulator)
        enc("VRESET"),                          # 6: vloop_idx = 0
        # L_EXP (PC = 7):
        enc("VLD", rd=1, imm=x_base),           # 7: v1[l] = x[vloop_idx + l]
        enc("FSUB", rd=1, rs1=1, rs2=0),        # 8: v1[l] = x[i] - max_val
        enc("FEXP", rd=1, rs1=1),               # 9: v1[l] = expf(x[i] - max_val)
        enc("VST", rs2=1, imm=x_base),          # 10: x[vloop_idx + l] = v1[l]
        enc("RADD", rd=2, rs1=1),               # 11: v2[all] += sum_{active} v1[l]
        enc("VSTEP", imm=7),                    # 12: loop to PC=7
        # Pass 3: reciprocal & normalize
        enc("FRECIP", rd=2, rs1=2),             # 13: v2[all] = 1.0f / sum
        enc("VRESET"),                          # 14: vloop_idx = 0
        # L_NORM (PC = 15):
        enc("VLD", rd=1, imm=x_base),           # 15: v1[l] = x[vloop_idx + l]
        enc("FMUL", rd=1, rs1=1, rs2=2),        # 16: v1[l] *= inv_sum
        enc("VST", rs2=1, imm=x_base),          # 17: x[vloop_idx + l] = v1[l]
        enc("VSTEP", imm=15),                   # 18: loop to PC=15
        enc("HALT"),                            # 19
    ]


def build_swiglu_kernel(hb_base: int = 0, hb2_base: int = 1024) -> list[int]:
    """
    In-place SwiGLU over SPM[hb_base .. hb_base+N-1] and SPM[hb2_base .. hb2_base+N-1]:
        hb[i] = silu(hb[i]) * hb2[i]
    """
    return [
        enc("VRESET"),                          # 0: vloop_idx = 0
        # L_LOOP (PC = 1):
        enc("VLD", rd=1, imm=hb_base),          # 1: v1[l] = hb[vloop_idx + l]
        enc("VLD", rd=2, imm=hb2_base),         # 2: v2[l] = hb2[vloop_idx + l]
        enc("FSILU", rd=1, rs1=1),              # 3: v1[l] = v1[l] / (1 + exp(-v1[l]))
        enc("FMUL", rd=1, rs1=1, rs2=2),        # 4: v1[l] = silu(hb[i]) * hb2[i]
        enc("VST", rs2=1, imm=hb_base),         # 5: hb[vloop_idx + l] = v1[l]
        enc("VSTEP", imm=1),                    # 6: loop to PC=1
        enc("HALT"),                            # 7
    ]


def build_rope_kernel(vec_base: int = 0, fcr_base: int = 1024, fci_base: int = 1536) -> list[int]:
    """
    In-place Rotary Positional Embedding (RoPE) over N = dim/2 complex pairs:
        vec[2*k]     = v0 * fcr[k] - v1 * fci[k]
        vec[2*k + 1] = v0 * fci[k] + v1 * fcr[k]
    Uses indexed gather/scatter (`VLD_ABS` / `VST_ABS`) so interleaved `q`/`k`
    vectors are rotated in place without host-side de-interleaving.
    """
    return [
        enc("LDI", rd=2, imm=1),                # 0: v2 = 1
        enc("VRESET"),                          # 1: vloop_idx = 0 (pair index k)
        # L_ROPE (PC = 2):
        enc("LD_LANE", rd=1),                   # 2: v1[l] = k = vloop_idx + l
        enc("SHL", rd=3, rs1=1, rs2=2),         # 3: v3[l] = 2*k (even index)
        enc("IADD", rd=4, rs1=3, rs2=2),        # 4: v4[l] = 2*k + 1 (odd index)
        enc("VLD_ABS", rd=5, rs1=3, imm=vec_base),  # 5: v5[l] = vec[2*k]     (v0)
        enc("VLD_ABS", rd=6, rs1=4, imm=vec_base),  # 6: v6[l] = vec[2*k + 1] (v1)
        enc("VLD", rd=7, imm=fcr_base),         # 7: v7[l] = fcr[k]
        enc("VLD", rd=8, imm=fci_base),         # 8: v8[l] = fci[k]
        enc("FMUL", rd=9, rs1=5, rs2=7),        # 9:  v9  = v0 * fcr
        enc("FMUL", rd=10, rs1=6, rs2=8),       # 10: v10 = v1 * fci
        enc("FSUB", rd=11, rs1=9, rs2=10),      # 11: v11 = v0*fcr - v1*fci
        enc("FMUL", rd=12, rs1=5, rs2=8),       # 12: v12 = v0 * fci
        enc("FMUL", rd=13, rs1=6, rs2=7),       # 13: v13 = v1 * fcr
        enc("FADD", rd=14, rs1=12, rs2=13),     # 14: v14 = v0*fci + v1*fcr
        enc("VST_ABS", rs1=3, rs2=11, imm=vec_base),  # 15: vec[2*k]     = v11
        enc("VST_ABS", rs1=4, rs2=14, imm=vec_base),  # 16: vec[2*k + 1] = v14
        enc("VSTEP", imm=2),                    # 17: loop to PC=2
        enc("HALT"),                            # 18
    ]


def build_residual_kernel(x_base: int = 0, xb_base: int = 1024) -> list[int]:
    """In-place Residual Add: x[i] += xb[i] over N elements."""
    return [
        enc("VRESET"),                          # 0
        enc("VLD", rd=1, imm=x_base),           # 1: v1 = x[i]
        enc("VLD", rd=2, imm=xb_base),          # 2: v2 = xb[i]
        enc("FADD", rd=1, rs1=1, rs2=2),        # 3: v1 = x[i] + xb[i]
        enc("VST", rs2=1, imm=x_base),          # 4: x[i] = v1
        enc("VSTEP", imm=1),                    # 5
        enc("HALT"),                            # 6
    ]


def build_q80_epilogue_kernel(
    psum_base: int = 0,
    ws_base: int = 1024,
    acc_base: int = 2048,
) -> list[int]:
    """
    Fused NPU->GPU group dequantization epilogue:
        acc[r] += float32(psum[r]) * ws[r] * xs
    where PARAM0 holds `xs` (the activation scale for the current group).
    """
    return [
        enc("LD_PARAM", rd=0, imm=0),           # 0: v0 = xs (FP32)
        enc("VRESET"),                          # 1
        enc("VLD", rd=1, imm=psum_base),        # 2: v1 = psum[r] (INT32)
        enc("I2F", rd=1, rs1=1),                # 3: v1 = float(psum[r])
        enc("VLD", rd=2, imm=ws_base),          # 4: v2 = ws[r] (FP32)
        enc("VLD", rd=3, imm=acc_base),         # 5: v3 = acc[r] (FP32)
        enc("FMUL", rd=2, rs1=2, rs2=0),        # 6: v2 = ws[r] * xs
        enc("FMUL", rd=1, rs1=1, rs2=2),        # 7: v1 = float(psum[r]) * ws[r] * xs
        enc("FADD", rd=3, rs1=3, rs2=1),        # 8: v3 = acc[r] + term
        enc("VST", rs2=3, imm=acc_base),        # 9: acc[r] = v3
        enc("VSTEP", imm=2),                    # 10
        enc("HALT"),                            # 11
    ]


def emit_c_header(path: str) -> None:
    kernels = {
        "gpu_prog_rmsnorm": build_rmsnorm_kernel(),
        "gpu_prog_softmax": build_softmax_kernel(),
        "gpu_prog_swiglu": build_swiglu_kernel(),
        "gpu_prog_rope": build_rope_kernel(),
        "gpu_prog_residual": build_residual_kernel(),
        "gpu_prog_q80_epilogue": build_q80_epilogue_kernel(),
    }
    lines = [
        "/* Auto-generated by sw/gpu_kernels/kernels.py - do not edit manually. */",
        "#ifndef HETERO_GPU_KERNELS_BIN_H",
        "#define HETERO_GPU_KERNELS_BIN_H",
        "#include <stdint.h>",
        "",
    ]
    for name, words in kernels.items():
        hex_words = ", ".join(f"0x{w:08x}u" for w in words)
        lines.append(f"static const uint32_t {name}[{len(words)}] = {{ {hex_words} }};")
    lines.append("\n#endif /* HETERO_GPU_KERNELS_BIN_H */\n")
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    out_h = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_kernels_bin.h")
    emit_c_header(out_h)
    print(f"wrote {out_h}")
