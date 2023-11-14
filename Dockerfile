ARG BASE_IMAGE=rapidsai/devcontainers:23.10-cpp-rust-cuda12.2-ubuntu22.04
FROM $BASE_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH=x86_64
WORKDIR /

SHELL ["/bin/bash", "-c"]

# Compile libcudf from source
RUN git clone --branch branch-23.10 https://github.com/rapidsai/cudf.git /cudf \
    && cd /cudf \
    && PARALLEL_LEVEL=16 CUDF_CMAKE_CUDA_ARCHITECTURES="70;80;90" ./build.sh libcudf benchmarks --ptds --cmake-args=\" -DCUDF_ENABLE_ARROW_S3=OFF -DBUILD_BENCHMARKS=ON -DCUDA_ENABLE_LINEINFO=ON \"

# Install GQE dependencies
#
# List these after cuDF to skip cuDF build when adding more packages
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        protobuf-compiler libprotobuf-dev

# Install common packages for development
RUN apt-get install -y --no-install-recommends \
        vim \
        libpciaccess-dev \
        pciutils \
        numactl \
        libnuma-dev \
        binutils-dev \
        openssh-client

# Clean up temporary apt files
RUN rm -rf /var/lib/apt/lists/*