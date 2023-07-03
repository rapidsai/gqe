ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04
FROM $BASE_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /

# Install common packages for development
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        vim \
        libpciaccess-dev \
        pciutils \
        ca-certificates \
        gnupg \
        numactl \
        libnuma-dev \
        file \
        pkg-config \
        binutils \
        binutils-dev \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install mamba
ADD https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh /mambaforge.sh
RUN sh /mambaforge.sh -b -p /conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate"
SHELL ["/bin/bash", "-c"]

# Compile libcudf from source
RUN git clone https://github.com/rapidsai/cudf.git /cudf \
    && cd /cudf \
    && git checkout branch-23.06 \
    && git submodule update --init --remote --recursive \
    && mamba env create -q --name gqe --file conda/environments/all_cuda-118_arch-x86_64.yaml \
    && source activate gqe \
    && PARALLEL_LEVEL=16 CUDF_CMAKE_CUDA_ARCHITECTURES="70;80" ./build.sh libcudf benchmarks --ptds --cmake-args=\" -DCUDAToolkit_ROOT=/usr/local/cuda -DCUDF_ENABLE_ARROW_S3=ON -DBUILD_BENCHMARKS=ON -DCUDA_ENABLE_LINEINFO=ON \"
