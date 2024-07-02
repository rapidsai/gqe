ARG BASE_IMAGE=ubuntu:22.04
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
        file \
        pkg-config \
        binutils \
        binutils-dev \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install mamba
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-$(uname -m).sh -O /mambaforge.sh
RUN sh /mambaforge.sh -b -p /conda
ENV PATH=${PATH}:/conda/bin
# Enables "source activate"
SHELL ["/bin/bash", "-c"]

# Compile libcudf from source
COPY conda/*.yml /config/conda/
RUN git clone https://github.com/rapidsai/cudf.git /cudf \
    && cd /cudf \
    && git checkout branch-24.06 \
    && git submodule update --init --remote --recursive \
    && mamba env create -q --name gqe --file /config/conda/docker-$(uname -m).yml \
    && source activate gqe \
    && PARALLEL_LEVEL=16 CUDF_CMAKE_CUDA_ARCHITECTURES="70;80;90" ./build.sh libcudf --ptds --cmake-args=\" -DCUDF_ENABLE_ARROW_S3=OFF -DBUILD_BENCHMARKS=OFF -DCUDA_ENABLE_LINEINFO=ON \" \
    && conda clean --all

# Activate the conda environment when launching the container
RUN echo "source activate gqe" >> ~/.bashrc
