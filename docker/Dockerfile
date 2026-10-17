# syntax=docker/dockerfile:1
# Development environment for the Heterogeneous CPU-NPU-Compute GPU System.
# Provides everything except Vivado (which is run manually on the host side):
#   - RTL simulation : Verilator 5, Icarus Verilog 12, cocotb
#   - ARM toolchain  : arm-linux-gnueabihf (matches PYNQ v3.x Ubuntu 22.04 armhf, glibc 2.35)
#   - qemu-user      : run ARM binaries locally for functional checks
#   - Python         : numpy / torch-cpu / sentencepiece for model export & analysis
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VERILATOR_VERSION=v5.030
ARG IVERILOG_VERSION=v12_0
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget ca-certificates pkg-config \
        autoconf bison flex gperf help2man ccache \
        libfl2 libfl-dev zlib1g zlib1g-dev libgoogle-perftools-dev numactl perl \
        python3 python3-pip python3-venv python3-dev \
        gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf libc6-dev-armhf-cross \
        qemu-user \
        make cmake ninja-build gdb gdb-multiarch \
        openssh-client rsync sudo less vim file bc time \
    && rm -rf /var/lib/apt/lists/*

# ---- Verilator 5 (Ubuntu 22.04 ships 4.038, too old for cocotb / SV features) ----
RUN git clone --depth 1 --branch ${VERILATOR_VERSION} https://github.com/verilator/verilator /tmp/verilator \
    && cd /tmp/verilator && autoconf && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" && make install && rm -rf /tmp/verilator

# ---- Icarus Verilog 12 ----
RUN git clone --depth 1 --branch ${IVERILOG_VERSION} https://github.com/steveicarus/iverilog /tmp/iverilog \
    && cd /tmp/iverilog && sh autoconf.sh && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" && make install && rm -rf /tmp/iverilog

# ---- Python environment ----
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1 \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ---- Non-root user matching host UID/GID (bind-mounted repo stays writable) ----
RUN groupadd -g ${GID} ${USERNAME} \
    && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# qemu-arm needs the armhf sysroot to run dynamically-linked binaries
ENV QEMU_LD_PREFIX=/usr/arm-linux-gnueabihf

USER ${USERNAME}
WORKDIR /workspace
CMD ["/bin/bash"]
