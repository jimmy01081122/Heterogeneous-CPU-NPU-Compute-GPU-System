# Runtime build: host (x86, golden reference) and arm (PYNQ-Z2 Cortex-A9).
#
#   make                  host + arm, plain + profiled builds
#   make host | arm       one target only
#   make FAST=1           -Ofast (enables NEON float vectorisation on A9, changes numerics)
#   make OMP=1            OpenMP (A9 is dual-core)
#
# Outputs: build/<target>/run_{fp32,int8}[_prof]

SRC_DIR  := src
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
LDLIBS        := -lm

HOST_CC     := gcc
HOST_CFLAGS := $(CFLAGS_COMMON) -march=x86-64

ARM_CC      := arm-linux-gnueabihf-gcc
ARM_CFLAGS  := $(CFLAGS_COMMON) -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard

HOST_BINS := $(foreach s,$(SOURCES),$(BUILD)/host/$(s) $(BUILD)/host/$(s)_prof)
ARM_BINS  := $(foreach s,$(SOURCES),$(BUILD)/arm/$(s) $(BUILD)/arm/$(s)_prof)

.PHONY: all host arm clean
all: host arm
host: $(HOST_BINS)
arm:  $(ARM_BINS)

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
