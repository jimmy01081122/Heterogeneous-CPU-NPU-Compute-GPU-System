// SPDX-License-Identifier: MIT
// Compute GPU package: 32-bit SIMD/SIMT ISA encoding, opcodes, and AXI register map.
//
// 32-bit Instruction Format:
//   [31:26] opcode  (6 bits)
//   [25:22] rd      (4 bits, v0..v15)
//   [21:18] rs1     (4 bits, v0..v15)
//   [17:14] rs2     (4 bits, v0..v15)
//   [13:0]  imm14   (14-bit signed/unsigned immediate: SPM word offset, branch target, or const)

`timescale 1ns / 1ps

package gpu_pkg;
    localparam logic [31:0] GPU_MAGIC_ID = 32'h4750_5531; // "GPU1"

    // AXI4-Lite register byte offsets (0x00 .. 0x7F)
    localparam logic [6:0] GPU_REG_CTRL           = 7'h00;
    localparam logic [6:0] GPU_REG_STATUS         = 7'h04;
    localparam logic [6:0] GPU_REG_VEC_LEN        = 7'h08; // total vector elements N
    localparam logic [6:0] GPU_REG_DMA_ADDR       = 7'h0C; // SPM word base address for DMA
    localparam logic [6:0] GPU_REG_DMA_LEN        = 7'h10; // number of 32-bit words for DMA
    localparam logic [6:0] GPU_REG_IMEM_ADDR      = 7'h14; // IMEM word index (auto-increments on IMEM_DATA write)
    localparam logic [6:0] GPU_REG_IMEM_DATA      = 7'h18; // write instruction word to IMEM
    localparam logic [6:0] GPU_REG_PARAM0         = 7'h1C; // scalar parameter 0 (e.g. eps, 1/N, pos, head_size)
    localparam logic [6:0] GPU_REG_PARAM1         = 7'h20; // scalar parameter 1
    localparam logic [6:0] GPU_REG_PERF_CYCLES    = 7'h24; // total busy cycles
    localparam logic [6:0] GPU_REG_PERF_INSTS     = 7'h28; // dynamic instructions executed
    localparam logic [6:0] GPU_REG_PERF_STALL     = 7'h2C; // stream stall cycles
    localparam logic [6:0] GPU_REG_PERF_BYTES_IN  = 7'h30;
    localparam logic [6:0] GPU_REG_PERF_BYTES_OUT = 7'h34;
    localparam logic [6:0] GPU_REG_ID             = 7'h38; // GPU_MAGIC_ID

    // Commands in GPU_REG_CTRL[5:2]
    typedef enum logic [3:0] {
        GPU_CMD_NOP        = 4'd0,
        GPU_CMD_DMA_H2D    = 4'd1, // load REG_DMA_LEN words from s_axis into SPM[REG_DMA_ADDR..]
        GPU_CMD_DMA_D2H    = 4'd2, // stream REG_DMA_LEN words from SPM[REG_DMA_ADDR..] to m_axis
        GPU_CMD_RUN_KERNEL = 4'd3, // execute IMEM from PC=0 until OP_HALT
        GPU_CMD_CLEAR_PERF = 4'd4
    } gpu_cmd_e;

    // 6-bit Instruction Opcodes
    typedef enum logic [5:0] {
        OP_NOP      = 6'h00,
        // Memory Load / Store (via Load/Store Unit + Scratchpad Memory)
        OP_VLD      = 6'h01, // vd[l] <= SPM[imm14 + vloop_idx + (rs1 ? v[rs1][l] : 0) + l]
        OP_VST      = 6'h02, // SPM[imm14 + vloop_idx + (rs1 ? v[rs1][l] : 0) + l] <= vs2[l]
        OP_SLD      = 6'h03, // broadcast scalar: vd[l] <= SPM[imm14 + v[rs1][0]]
        OP_SST      = 6'h04, // scalar store:     SPM[imm14 + v[rs1][0]] <= vs2[0]
        OP_VLD_ABS  = 6'h05, // absolute indexed load:  vd[l] <= SPM[imm14 + v[rs1][l]] (no vloop_idx)
        OP_VST_ABS  = 6'h06, // absolute indexed store: SPM[imm14 + v[rs1][l]] <= vs2[l]
        // Immediate / Special Register Moves
        OP_LDI      = 6'h08, // vd[l] <= sign_extend(imm14) (int32)
        OP_LD_PARAM = 6'h09, // vd[l] <= (imm14[0] ? REG_PARAM1 : REG_PARAM0)
        OP_LD_LANE  = 6'h0A, // vd[l] <= vloop_idx + l (int32 element index)
        OP_MOV      = 6'h0B, // vd[l] <= vs1[l]
        // Integer / Bitwise Vector Arithmetic
        OP_IADD     = 6'h10,
        OP_ISUB     = 6'h11,
        OP_IMUL     = 6'h12,
        OP_Q16MUL   = 6'h13,
        OP_AND      = 6'h14,
        OP_OR       = 6'h15,
        OP_XOR      = 6'h16,
        OP_SHL      = 6'h17,
        OP_SRA      = 6'h18,
        // IEEE-754 FP32 Vector Arithmetic
        OP_FADD     = 6'h20,
        OP_FSUB     = 6'h21,
        OP_FMUL     = 6'h22,
        OP_FMAX     = 6'h23,
        OP_FMIN     = 6'h24,
        OP_I2F      = 6'h25,
        OP_F2I      = 6'h26,
        // Nonlinear / Transcendental Vector Ops (FP32)
        OP_FRECIP   = 6'h28, // vd[l] <= 1.0f / vs1[l]
        OP_FRSQRT   = 6'h29, // vd[l] <= 1.0f / sqrtf(vs1[l])
        OP_FEXP     = 6'h2A, // vd[l] <= expf(vs1[l])
        OP_FSILU    = 6'h2B, // vd[l] <= vs1[l] / (1.0f + expf(-vs1[l]))
        // Cross-Lane Reductions (broadcasts reduced scalar to all lanes of vd)
        OP_RADD     = 6'h30, // vd[all] <= vd[0] + sum_{active l} vs1[l] (FP32 accumulative tree sum)
        OP_RMAX     = 6'h31, // vd[all] <= max(vd[0], max_{active l} vs1[l]) (FP32 tree max)
        // Control Flow / SIMT Strip-Mining Loop
        OP_VRESET   = 6'h38, // vloop_idx <= 0
        OP_VSTEP    = 6'h39, // vloop_idx += N_LANES; if (vloop_idx < REG_VEC_LEN) PC <= imm14
        OP_JMP      = 6'h3E, // unconditional jump PC <= imm14
        OP_HALT     = 6'h3F  // finish kernel, assert status_done
    } gpu_opcode_e;

    typedef struct packed {
        gpu_opcode_e opcode;
        logic [3:0]  rd;
        logic [3:0]  rs1;
        logic [3:0]  rs2;
        logic [13:0] imm14;
    } gpu_inst_t;
endpackage
