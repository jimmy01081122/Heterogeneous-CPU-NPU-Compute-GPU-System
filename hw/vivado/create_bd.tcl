# SPDX-License-Identifier: MIT
# Vivado Block Design & Bitstream Generation Script for PYNQ-Z2 (xc7z020clg400-1)
#
# Usage:
#   # Create Vivado project + Block Design (fast, ~30s):
#   vivado -mode batch -source hw/vivado/create_bd.tcl
#
#   # Create project + Block Design + run synthesis, implementation & bitstream:
#   vivado -mode batch -source hw/vivado/create_bd.tcl -tclargs --bitstream
#
# Physical Address Map (matches sw/driver/accel_hw.c):
#   0x43C0_0000 (64 KB) : npu_0 (npu_top AXI4-Lite slave)
#   0x43C1_0000 (64 KB) : gpu_0 (gpu_top AXI4-Lite slave)
#   0x4040_0000 (64 KB) : axi_dma_npu (Direct Register Mode, MM2S/S2MM <-> S_AXI_HP0)
#   0x4041_0000 (64 KB) : axi_dma_gpu (Direct Register Mode, MM2S/S2MM <-> S_AXI_HP1)

set script_dir [file dirname [file normalize [info script]]]
set root_dir   [file normalize "$script_dir/../.."]
set proj_dir   "$script_dir/proj"
set proj_name  "hetero_llm_pynq_z2"
set part_name  "xc7z020clg400-1"

set run_bitstream 0
foreach arg $argv {
    if {$arg eq "--bitstream"} {
        set run_bitstream 1
    }
}

puts "=================================================================="
puts " Creating Vivado Project: $proj_name ($part_name)"
puts "=================================================================="

create_project -force $proj_name $proj_dir -part $part_name

# 1. Add SystemVerilog RTL, Verilog-2001 BD wrappers, and XDC constraints
add_files -norecurse [list \
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
    "$root_dir/hw/vivado/rtl_wrappers.v" \
]

add_files -fileset constrs_1 -norecurse "$root_dir/hw/vivado/pynq_z2.xdc"
update_compile_order -fileset sources_1

# 2. Create Block Design "system"
create_bd_design "system"

# 2a. Zynq-7000 Processing System (PS7) configured for PYNQ-Z2
set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
set_property -dict [list \
    CONFIG.PCW_APU_PERIPHERAL_FREQMHZ {650} \
    CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
    CONFIG.PCW_UIPARAM_DDR_PARTNO {MT41K256M16 RE-125} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_USE_S_AXI_HP1 {1} \
    CONFIG.PCW_S_AXI_HP1_DATA_WIDTH {64} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
] $ps7

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} $ps7

# 2b. Processor System Reset (100 MHz domain)
set rst_ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_100M]

# 2c. Instantiate NPU and Compute GPU RTL modules
set npu_0 [create_bd_cell -type module -reference npu_bd_wrapper npu_0]
set gpu_0 [create_bd_cell -type module -reference gpu_bd_wrapper gpu_0]

# 2d. Instantiate AXI DMAs for NPU (HP0) and GPU (HP1)
set dma_npu [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_npu]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_s2mm_burst_size {64} \
] $dma_npu

set dma_gpu [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_gpu]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_s2mm_burst_size {64} \
] $dma_gpu

# 2e. AXI Interconnects (GP0 control + HP0 NPU data + HP1 GPU data)
set axi_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_ctrl]
set_property CONFIG.NUM_MI {4} $axi_ctrl

set axi_hp0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_hp0]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $axi_hp0

set axi_hp1 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_mem_hp1]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $axi_hp1

# 2f. Interrupt Concatenation (6 sources -> IRQ_F2P)
set xlconcat_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_0]
set_property CONFIG.NUM_PORTS {6} $xlconcat_0

# 3. Connect Clocks & Resets
set clk_net [get_bd_pins processing_system7_0/FCLK_CLK0]
connect_bd_net $clk_net [get_bd_pins rst_ps7_100M/slowest_sync_clk]
connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] [get_bd_pins rst_ps7_100M/ext_reset_in]

set aresetn_net [get_bd_pins rst_ps7_100M/peripheral_aresetn]
set ic_aresetn  [get_bd_pins rst_ps7_100M/interconnect_aresetn]

foreach pin [list \
    processing_system7_0/M_AXI_GP0_ACLK \
    processing_system7_0/S_AXI_HP0_ACLK \
    processing_system7_0/S_AXI_HP1_ACLK \
    npu_0/clk \
    gpu_0/clk \
    axi_dma_npu/s_axi_lite_aclk \
    axi_dma_npu/m_axi_mm2s_aclk \
    axi_dma_npu/m_axi_s2mm_aclk \
    axi_dma_gpu/s_axi_lite_aclk \
    axi_dma_gpu/m_axi_mm2s_aclk \
    axi_dma_gpu/m_axi_s2mm_aclk \
    axi_mem_ctrl/ACLK \
    axi_mem_ctrl/S00_ACLK \
    axi_mem_ctrl/M00_ACLK \
    axi_mem_ctrl/M01_ACLK \
    axi_mem_ctrl/M02_ACLK \
    axi_mem_ctrl/M03_ACLK \
    axi_mem_hp0/ACLK \
    axi_mem_hp0/S00_ACLK \
    axi_mem_hp0/S01_ACLK \
    axi_mem_hp0/M00_ACLK \
    axi_mem_hp1/ACLK \
    axi_mem_hp1/S00_ACLK \
    axi_mem_hp1/S01_ACLK \
    axi_mem_hp1/M00_ACLK \
] {
    connect_bd_net $clk_net [get_bd_pins $pin]
}

foreach pin [list \
    npu_0/rst_n \
    gpu_0/rst_n \
    axi_dma_npu/axi_resetn \
    axi_dma_gpu/axi_resetn \
    axi_mem_ctrl/S00_ARESETN \
    axi_mem_ctrl/M00_ARESETN \
    axi_mem_ctrl/M01_ARESETN \
    axi_mem_ctrl/M02_ARESETN \
    axi_mem_ctrl/M03_ARESETN \
    axi_mem_hp0/S00_ARESETN \
    axi_mem_hp0/S01_ARESETN \
    axi_mem_hp0/M00_ARESETN \
    axi_mem_hp1/S00_ARESETN \
    axi_mem_hp1/S01_ARESETN \
    axi_mem_hp1/M00_ARESETN \
] {
    connect_bd_net $aresetn_net [get_bd_pins $pin]
}

foreach pin [list axi_mem_ctrl/ARESETN axi_mem_hp0/ARESETN axi_mem_hp1/ARESETN] {
    connect_bd_net $ic_aresetn [get_bd_pins $pin]
}

# 4. Connect AXI4-Lite Control Interfaces (GP0)
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] [get_bd_intf_pins axi_mem_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_ctrl/M00_AXI] [get_bd_intf_pins npu_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_mem_ctrl/M01_AXI] [get_bd_intf_pins gpu_0/s_axi]
connect_bd_intf_net [get_bd_intf_pins axi_mem_ctrl/M02_AXI] [get_bd_intf_pins axi_dma_npu/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins axi_mem_ctrl/M03_AXI] [get_bd_intf_pins axi_dma_gpu/S_AXI_LITE]

# 5. Connect 64-bit AXI4-Stream Data Paths
connect_bd_intf_net [get_bd_intf_pins axi_dma_npu/M_AXIS_MM2S] [get_bd_intf_pins npu_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins npu_0/m_axis]            [get_bd_intf_pins axi_dma_npu/S_AXIS_S2MM]

connect_bd_intf_net [get_bd_intf_pins axi_dma_gpu/M_AXIS_MM2S] [get_bd_intf_pins gpu_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins gpu_0/m_axis]            [get_bd_intf_pins axi_dma_gpu/S_AXIS_S2MM]

# 6. Connect High-Performance DDR Ports (HP0 for NPU, HP1 for GPU)
connect_bd_intf_net [get_bd_intf_pins axi_dma_npu/M_AXI_MM2S] [get_bd_intf_pins axi_mem_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_npu/M_AXI_S2MM] [get_bd_intf_pins axi_mem_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_hp0/M00_AXI]    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

connect_bd_intf_net [get_bd_intf_pins axi_dma_gpu/M_AXI_MM2S] [get_bd_intf_pins axi_mem_hp1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_gpu/M_AXI_S2MM] [get_bd_intf_pins axi_mem_hp1/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_mem_hp1/M00_AXI]    [get_bd_intf_pins processing_system7_0/S_AXI_HP1]

# 7. Connect Interrupts
connect_bd_net [get_bd_pins npu_0/irq]                [get_bd_pins xlconcat_0/In0]
connect_bd_net [get_bd_pins gpu_0/irq]                [get_bd_pins xlconcat_0/In1]
connect_bd_net [get_bd_pins axi_dma_npu/mm2s_introut] [get_bd_pins xlconcat_0/In2]
connect_bd_net [get_bd_pins axi_dma_npu/s2mm_introut] [get_bd_pins xlconcat_0/In3]
connect_bd_net [get_bd_pins axi_dma_gpu/mm2s_introut] [get_bd_pins xlconcat_0/In4]
connect_bd_net [get_bd_pins axi_dma_gpu/s2mm_introut] [get_bd_pins xlconcat_0/In5]
connect_bd_net [get_bd_pins xlconcat_0/dout]          [get_bd_pins processing_system7_0/IRQ_F2P]

# 8. Assign Physical Address Map (matching sw/driver/accel_hw.c)
assign_bd_address -offset 0x43C00000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    [get_bd_addr_segs npu_0/s_axi/reg0]

assign_bd_address -offset 0x43C10000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    [get_bd_addr_segs gpu_0/s_axi/reg0]

assign_bd_address -offset 0x40400000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    [get_bd_addr_segs axi_dma_npu/S_AXI_LITE/Reg]

assign_bd_address -offset 0x40410000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces processing_system7_0/Data] \
    [get_bd_addr_segs axi_dma_gpu/S_AXI_LITE/Reg]

assign_bd_address -offset 0x00000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_npu/Data_MM2S] \
    [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]

assign_bd_address -offset 0x00000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_npu/Data_S2MM] \
    [get_bd_addr_segs processing_system7_0/S_AXI_HP0/HP0_DDR_LOWOCM]

assign_bd_address -offset 0x00000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_gpu/Data_MM2S] \
    [get_bd_addr_segs processing_system7_0/S_AXI_HP1/HP1_DDR_LOWOCM]

assign_bd_address -offset 0x00000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces axi_dma_gpu/Data_S2MM] \
    [get_bd_addr_segs processing_system7_0/S_AXI_HP1/HP1_DDR_LOWOCM]

# 9. Validate, Save Block Design, and Generate Top Wrapper
regenerate_bd_layout
validate_bd_design
save_bd_design

set bd_file [get_files system.bd]
make_wrapper -files $bd_file -top
set wrapper_file [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_file
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

puts "=================================================================="
puts " Block Design 'system' validated and saved successfully."
puts " Project path: $proj_dir/$proj_name.xpr"
puts "=================================================================="

if {$run_bitstream} {
    puts " Launching Synthesis, Implementation, and Bitstream generation..."
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    launch_runs impl_1 -to_step write_bitstream -jobs 4
    wait_on_run impl_1

    file mkdir "$script_dir/overlay"
    file copy -force "$proj_dir/$proj_name.runs/impl_1/system_wrapper.bit" "$script_dir/overlay/hetero_llm.bit"
    set hwh_files [glob -nocomplain "$proj_dir/$proj_name.gen/sources_1/bd/system/hw_handoff/system.hwh" \
                                    "$proj_dir/$proj_name.srcs/sources_1/bd/system/hw_handoff/system.hwh"]
    if {[llength $hwh_files] > 0} {
        file copy -force [lindex $hwh_files 0] "$script_dir/overlay/hetero_llm.hwh"
    }
    puts " Overlay exported to: $script_dir/overlay/hetero_llm.{bit,hwh}"
}
