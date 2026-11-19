ARG BASE_IMAGE=ubuntu:24.04
FROM $BASE_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
ARG PARALLEL_LEVEL
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
    sudo \
    gosu \
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
    gcc-14 \
    g++-14 \
    gdb \
    sqlite3 \
    ncat \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge3
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /miniforge.sh \
    && sh /miniforge.sh -b -p /conda \
    && rm /miniforge.sh \
    && /conda/bin/conda init bash --system

ENV PATH=${PATH}:/conda/bin
# Enables "source activate"
SHELL ["/bin/bash", "-c"]

# Compile libcudf from source.
COPY conda/*.yml /config/conda/
ARG CUDF_COMMIT=v26.10.02
RUN git clone https://github.com/rapidsai/cudf.git /cudf \
    && cd /cudf \
    && git checkout "${CUDF_COMMIT}" \
    && mamba env create -q --name gqe --file /config/conda/docker-$(uname -m).yml \
    && source activate gqe \
    # Fail the image build if the base image and Conda sysroot glibc versions differ.
    && python -c 'import glob, json, os, sys; \
        glibc = os.confstr("CS_GNU_LIBC_VERSION").split()[1]; \
        metadata = glob.glob(os.environ["CONDA_PREFIX"] + "/conda-meta/sysroot_linux-*.json"); \
        sysroot = json.load(open(metadata[0]))["version"]; \
        sys.exit(0 if glibc == sysroot else \
            f"glibc/sysroot mismatch: base image={glibc}, Conda sysroot={sysroot}")' \
    && conda install -y libnvcomp-dev=5.2.0.10 \
    && PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)} CUDF_CMAKE_CUDA_ARCHITECTURES="80-real;90-real;100-real;103-real;120" ./build.sh libcudf --ptds --cmake-args=\" -DCUDF_ENABLE_ARROW_S3=OFF -DBUILD_BENCHMARKS=OFF -DCUDA_ENABLE_LINEINFO=ON \" \
    && conda remove libnvcomp-dev libnvcomp -y \
    && conda clean --all

# Compile MLIR from source. Assertions are off to keep the image build short and to match what is
# benchmarked. To develop the query compiler with assertions, build LLVM locally with
# -DLLVM_ENABLE_ASSERTIONS=ON; see docs/compiler/README.md.
ARG LLVM_VERSION=23.1.5
ARG LLVM_ENABLE_ASSERTIONS=OFF
RUN git clone --branch llvmorg-${LLVM_VERSION} --depth 1 https://github.com/llvm/llvm-project.git \
    && mkdir llvm-project/build \
    && pushd llvm-project/build \
    && source activate gqe \
    && cmake -G Ninja ../llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX" \
    -DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS} \
    -DLLVM_USE_SPLIT_DWARF=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    && cmake --build . --target install \
    && popd \
    && rm -rf llvm-project

# Compiler cache for rust (RUSTC_WRAPPER=sccache). Installed from the gqe
# env's rust toolchain into a system path so cargo can find it regardless
# of which env is active.
ARG SCCACHE_VERSION=0.15.0
RUN source activate gqe \
    && cargo install sccache --version "${SCCACHE_VERSION}" --locked --root /usr/local --no-default-features \
    && sccache --version

# Empty config file so sccache stops logging "Couldn't open config file" on
# every invocation.
RUN mkdir -p /root/.config/sccache && touch /root/.config/sccache/config

# Activate the conda environment when launching the container
RUN echo "source activate gqe" >> ~/.bashrc
