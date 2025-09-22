ARG BASE_IMAGE=ubuntu:22.04
FROM $BASE_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /

# Set visible devices and mount NVIDIA driver binary utilities inside the
# container
#
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#dockerfiles
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install common packages for development
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    wget \
    git \
    vim \
    ccache \
    libpciaccess-dev \
    pciutils \
    ca-certificates \
    gnupg \
    file \
    pkg-config \
    binutils \
    binutils-dev \
    openssh-client \
    openmpi-bin \
    libopenmpi-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge3
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /miniforge.sh
RUN sh /miniforge.sh -b -p /conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate"
SHELL ["/bin/bash", "-c"]

# Compile libcudf from source
COPY conda/*.yml /config/conda/
RUN git clone https://github.com/rapidsai/cudf.git /cudf \
    && cd /cudf \
    && git checkout 0fbd451151a3f5226624448a27de75a3d81b6e4e \
    && git submodule update --init --remote --recursive \
    && mamba env create -q --name gqe --file /config/conda/docker-$(uname -m).yml \
    && source activate gqe \
    && PARALLEL_LEVEL=16 CUDF_CMAKE_CUDA_ARCHITECTURES="70-real;80-real;90-real;100-real;120" ./build.sh libcudf --ptds --cmake-args=\" -DCUDF_ENABLE_ARROW_S3=OFF -DBUILD_BENCHMARKS=OFF -DCUDA_ENABLE_LINEINFO=ON -Drapids-cmake-sha=c82ccbf86c53437d80a45c502e64029b7c1c7f31 \" \
    && conda clean --all

# Compile MLIR from source
RUN git clone https://github.com/llvm/llvm-project.git \
    && mkdir llvm-project/build \
    && pushd llvm-project/build \
    && git checkout -b llvmorg-20.1.2 \
    && source activate gqe \
    && cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    && cmake --build . --target install \
    && popd \
    && rm -rf llvm-project


# Activate the conda environment when launching the container
RUN echo "source activate gqe" >> ~/.bashrc
