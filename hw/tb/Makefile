# Cocotb simulation Makefile (Verilator 5).
#
#   make -C hw/tb TARGET=npu
#   make -C hw/tb TARGET=gpu
#   make -C hw/tb all

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

TARGET ?= npu
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)

ifeq ($(TARGET),npu)
  TOPLEVEL := npu_top
  MODULE   := test_npu
  VERILOG_SOURCES := \
    $(ROOT)/hw/rtl/common/fp32_alu.sv \
    $(ROOT)/hw/rtl/npu/npu_pkg.sv \
    $(ROOT)/hw/rtl/npu/npu_mac_array.sv \
    $(ROOT)/hw/rtl/npu/npu_act_buf.sv \
    $(ROOT)/hw/rtl/npu/npu_top.sv
  SIM_BUILD := sim_build/npu
else ifeq ($(TARGET),gpu)
  TOPLEVEL := gpu_top
  MODULE   := test_gpu
  VERILOG_SOURCES := \
    $(ROOT)/hw/rtl/common/fp32_alu.sv \
    $(ROOT)/hw/rtl/gpu/gpu_pkg.sv \
    $(ROOT)/hw/rtl/gpu/gpu_decoder.sv \
    $(ROOT)/hw/rtl/gpu/gpu_regfile.sv \
    $(ROOT)/hw/rtl/gpu/gpu_lane.sv \
    $(ROOT)/hw/rtl/gpu/gpu_scratchpad.sv \
    $(ROOT)/hw/rtl/gpu/gpu_lsu.sv \
    $(ROOT)/hw/rtl/gpu/gpu_controller.sv \
    $(ROOT)/hw/rtl/gpu/gpu_top.sv
  SIM_BUILD := sim_build/gpu
endif

EXTRA_ARGS += --sv -Wno-fatal

include $(shell cocotb-config --makefiles)/Makefile.sim
