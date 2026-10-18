# Runtime build: host (x86, golden reference + Verilated RTL co-sim) and arm (PYNQ-Z2 Cortex-A9).
#
#   make                  host + arm, plain + profiled + heterogeneous builds
#   make host | arm       one target only
#   make FAST=1           -Ofast (enables NEON float vectorisation on A9, changes numerics)
#   make OMP=1            OpenMP (A9 is dual-core)
#
# Outputs:
#   build/<target>/run_{fp32,int8,hetero}[_prof]

ROOT     := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
SRC_DIR  := src
DRV_DIR  := ../driver
GPU_DIR  := ../gpu_kernels
BUILD    := build
SOURCES  := run_fp32 run_int8

OPT      := -O3
ifeq ($(FAST),1)
  OPT    := -Ofast
endif
ifeq ($(OMP),1)
  OPT    += -fopenmp
endif

CFLAGS_COMMON := $(OPT) -std=gnu11 -Wall -Wno-unused-result -Wno-unused-function -Wno-unknown-pragmas
CXXFLAGS_COMMON := $(OPT) -std=c++17 -Wall -Wno-unused-result -Wno-unused-function
LDLIBS        := -lm

HOST_CC     := gcc
HOST_CXX    := g++
HOST_CFLAGS := $(CFLAGS_COMMON) -march=x86-64
HOST_CXXFLAGS := $(CXXFLAGS_COMMON) -march=x86-64

ARM_CC      := arm-linux-gnueabihf-gcc
ARM_CFLAGS  := $(CFLAGS_COMMON) -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard

VERILATOR      ?= verilator
VERILATOR_ROOT ?= $(shell $(VERILATOR) -getenv VERILATOR_ROOT 2>/dev/null || echo /usr/local/share/verilator)
VERILATOR_INC  := $(VERILATOR_ROOT)/include

NPU_RTL_SRCS := \
  $(ROOT)/hw/rtl/common/fp32_alu.sv \
  $(ROOT)/hw/rtl/npu/npu_pkg.sv \
  $(ROOT)/hw/rtl/npu/npu_mac_array.sv \
  $(ROOT)/hw/rtl/npu/npu_act_buf.sv \
  $(ROOT)/hw/rtl/npu/npu_top.sv

GPU_RTL_SRCS := \
  $(ROOT)/hw/rtl/common/fp32_alu.sv \
  $(ROOT)/hw/rtl/gpu/gpu_pkg.sv \
  $(ROOT)/hw/rtl/gpu/gpu_decoder.sv \
  $(ROOT)/hw/rtl/gpu/gpu_regfile.sv \
  $(ROOT)/hw/rtl/gpu/gpu_lane.sv \
  $(ROOT)/hw/rtl/gpu/gpu_scratchpad.sv \
  $(ROOT)/hw/rtl/gpu/gpu_lsu.sv \
  $(ROOT)/hw/rtl/gpu/gpu_controller.sv \
  $(ROOT)/hw/rtl/gpu/gpu_top.sv

DRV_C_SRCS := $(DRV_DIR)/accel.c $(DRV_DIR)/accel_sim.c $(DRV_DIR)/accel_hw.c
DRV_HDRS   := $(DRV_DIR)/accel.h $(DRV_DIR)/accel_backend.h $(DRV_DIR)/accel_hal.h $(GPU_DIR)/gpu_kernels_bin.h

VNPU_DIR := $(BUILD)/verilated_npu
VGPU_DIR := $(BUILD)/verilated_gpu
VNPU_LIB := $(VNPU_DIR)/libVnpu_top.a
VGPU_LIB := $(VGPU_DIR)/libVgpu_top.a

HOST_BINS := $(foreach s,$(SOURCES),$(BUILD)/host/$(s) $(BUILD)/host/$(s)_prof) \
             $(BUILD)/host/run_hetero $(BUILD)/host/run_hetero_prof
ARM_BINS  := $(foreach s,$(SOURCES),$(BUILD)/arm/$(s) $(BUILD)/arm/$(s)_prof) \
             $(BUILD)/arm/run_hetero $(BUILD)/arm/run_hetero_prof

.PHONY: all host arm clean
all: host arm
host: $(HOST_BINS)
arm:  $(ARM_BINS)

$(VNPU_LIB): $(NPU_RTL_SRCS)
	$(VERILATOR) --cc --build -j 4 --Mdir $(VNPU_DIR) --prefix Vnpu_top --top-module npu_top -Wno-fatal $(NPU_RTL_SRCS)

$(VGPU_LIB): $(GPU_RTL_SRCS)
	$(VERILATOR) --cc --build -j 4 --Mdir $(VGPU_DIR) --prefix Vgpu_top --top-module gpu_top -Wno-fatal $(GPU_RTL_SRCS)

$(BUILD)/host/accel_rtl.o: $(DRV_DIR)/accel_rtl.cpp $(DRV_HDRS) $(VNPU_LIB) $(VGPU_LIB) | $(BUILD)/host
	$(HOST_CXX) $(HOST_CXXFLAGS) -I$(VERILATOR_INC) -I$(VERILATOR_INC)/vltstd -I$(VNPU_DIR) -I$(VGPU_DIR) -c -o $@ $<

$(BUILD)/host/run_hetero: $(SRC_DIR)/run_hetero.c $(DRV_C_SRCS) $(DRV_HDRS) $(BUILD)/host/accel_rtl.o $(VNPU_LIB) $(VGPU_LIB) | $(BUILD)/host
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -c -o $(BUILD)/host/run_hetero.o $(SRC_DIR)/run_hetero.c
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -c -o $(BUILD)/host/accel.o $(DRV_DIR)/accel.c
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -c -o $(BUILD)/host/accel_sim.o $(DRV_DIR)/accel_sim.c
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -c -o $(BUILD)/host/accel_hw.o $(DRV_DIR)/accel_hw.c
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ $(BUILD)/host/run_hetero.o $(BUILD)/host/accel.o $(BUILD)/host/accel_sim.o $(BUILD)/host/accel_hw.o $(BUILD)/host/accel_rtl.o $(VNPU_LIB) $(VGPU_LIB) $(VNPU_DIR)/libverilated.a -pthread $(LDLIBS)

$(BUILD)/host/run_hetero_prof: $(SRC_DIR)/run_hetero.c $(DRV_C_SRCS) $(DRV_HDRS) $(BUILD)/host/accel_rtl.o $(VNPU_LIB) $(VGPU_LIB) | $(BUILD)/host
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -DPROFILE -c -o $(BUILD)/host/run_hetero_prof.o $(SRC_DIR)/run_hetero.c
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -DPROFILE -c -o $(BUILD)/host/accel.o $(DRV_DIR)/accel.c
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -DPROFILE -c -o $(BUILD)/host/accel_sim.o $(DRV_DIR)/accel_sim.c
	$(HOST_CC) $(HOST_CFLAGS) -DHAVE_VERILATOR_RTL -DPROFILE -c -o $(BUILD)/host/accel_hw.o $(DRV_DIR)/accel_hw.c
	$(HOST_CXX) $(HOST_CXXFLAGS) -o $@ $(BUILD)/host/run_hetero_prof.o $(BUILD)/host/accel.o $(BUILD)/host/accel_sim.o $(BUILD)/host/accel_hw.o $(BUILD)/host/accel_rtl.o $(VNPU_LIB) $(VGPU_LIB) $(VNPU_DIR)/libverilated.a -pthread $(LDLIBS)

$(BUILD)/arm/run_hetero: $(SRC_DIR)/run_hetero.c $(DRV_C_SRCS) $(DRV_HDRS) | $(BUILD)/arm
	$(ARM_CC) $(ARM_CFLAGS) -o $@ $(SRC_DIR)/run_hetero.c $(DRV_C_SRCS) $(LDLIBS)

$(BUILD)/arm/run_hetero_prof: $(SRC_DIR)/run_hetero.c $(DRV_C_SRCS) $(DRV_HDRS) | $(BUILD)/arm
	$(ARM_CC) $(ARM_CFLAGS) -DPROFILE -o $@ $(SRC_DIR)/run_hetero.c $(DRV_C_SRCS) $(LDLIBS)

$(BUILD)/host/%_prof: $(SRC_DIR)/%.c $(SRC_DIR)/prof.h | $(BUILD)/host
	$(HOST_CC) $(HOST_CFLAGS) -DPROFILE -o $@ $< $(LDLIBS)
$(BUILD)/host/%: $(SRC_DIR)/%.c $(SRC_DIR)/prof.h | $(BUILD)/host
	$(HOST_CC) $(HOST_CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD)/arm/%_prof: $(SRC_DIR)/%.c $(SRC_DIR)/prof.h | $(BUILD)/arm
	$(ARM_CC) $(ARM_CFLAGS) -DPROFILE -o $@ $< $(LDLIBS)
$(BUILD)/arm/%: $(SRC_DIR)/%.c $(SRC_DIR)/prof.h | $(BUILD)/arm
	$(ARM_CC) $(ARM_CFLAGS) -o $@ $< $(LDLIBS)

$(BUILD)/host $(BUILD)/arm:
	mkdir -p $@

clean:
	rm -rf $(BUILD)
