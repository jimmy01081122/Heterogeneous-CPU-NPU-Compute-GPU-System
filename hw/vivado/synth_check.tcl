# SPDX-License-Identifier: MIT
# Out-of-Context (OOC) Synthesis & Resource/Timing Characterization for PYNQ-Z2.
#
# Run manually on your Vivado machine from the repo root:
#   vivado -mode batch -source hw/vivado/synth_check.tcl
#
# Generates LUT / FF / DSP48E1 / BRAM36 utilization and timing summaries for:
#   1. npu_top (8x8 INT8 MAC array)
#   2. gpu_top (4-lane SIMD/SIMT Compute GPU)
# under hw/vivado/build/reports/.

set part "xc7z020clg400-1"
set root_dir [file normalize [file dirname [info script]]/../..]
set rpt_dir  "$root_dir/hw/vivado/build/reports"
file mkdir $rpt_dir

# Create an in-memory project targeting PYNQ-Z2 (Zynq XC7Z020-1CLG400C)
create_project -in_memory -part $part

read_verilog -sv [list \
    "$root_dir/hw/rtl/common/fp32_alu.sv" \
    "$root_dir/hw/rtl/npu/npu_pkg.sv" \
    "$root_dir/hw/rtl/npu/npu_mac_array.sv" \
    "$root_dir/hw/rtl/npu/npu_act_buf.sv" \
    "$root_dir/hw/rtl/npu/npu_top.sv" \
    "$root_dir/hw/rtl/gpu/gpu_pkg.sv" \
    "$root_dir/hw/rtl/gpu/gpu_decoder.sv" \
    "$root_dir/hw/rtl/gpu/gpu_regfile.sv" \
    "$root_dir/hw/rtl/gpu/gpu_lane.sv" \
    "$root_dir/hw/rtl/gpu/gpu_scratchpad.sv" \
    "$root_dir/hw/rtl/gpu/gpu_lsu.sv" \
    "$root_dir/hw/rtl/gpu/gpu_controller.sv" \
    "$root_dir/hw/rtl/gpu/gpu_top.sv" \
]

foreach top {npu_top gpu_top} {
    puts "=== Synthesizing $top for $part (OOC, 100 MHz target clock) ==="
    synth_design -top $top -part $part -mode out_of_context
    create_clock -period 10.000 -name clk [get_ports clk]
    report_utilization -file "$rpt_dir/${top}_utilization.rpt"
    report_timing_summary -file "$rpt_dir/${top}_timing.rpt"
    write_checkpoint -force "$rpt_dir/${top}_synth.dcp"
}

puts "Done. Reports written to $rpt_dir"
