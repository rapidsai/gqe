#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Build one CI variant. Internal builds use build/ for tests in the same job;
# the public x86 build uses a separate directory for compile-only coverage.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/ci/gpu_detection.sh
source "${script_dir}/gpu_detection.sh"

# Keep tokened Git rewrite rules in a job-local global config and remove it on exit.
export GIT_CONFIG_GLOBAL="${GQE_CI_GIT_CONFIG_GLOBAL:-${PWD}/.ci-gitconfig}"
cleanup_git_config()
{
  rm -f "${GIT_CONFIG_GLOBAL}"
}
trap cleanup_git_config EXIT

git config --global --add safe.directory "${PWD}" || true

set +u
source /conda/bin/activate gqe
set -u

export CARGO_HOME=/cargo_home CPM_SOURCE_CACHE=/cpm_cache
export CCACHE_DIR=/ccache_files

hostname

detect_cuda_arch

cmake_args=(
  # RAPIDS does not lock the CPM bootstrap download. Keep that file job-local
  # while dependency sources continue to use the persistent CPM source cache.
  -DCPM_DOWNLOAD_LOCATION=/gqe_ci_home/CPM.cmake
  -DGQE_ENABLE_CUSTOMIZED_PARQUET=ON
  -DGQE_ENABLE_FAULT_INJECTION=ON
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
  -DCARGO_EXECUTABLE="${script_dir}/cargo_with_sccache.sh"
  -G Ninja
)

if [[ "${GQE_CI_PUBLIC_BUILD:-true}" != "true" ]]; then
  : "${GQE_NVCOMP_DEPLOY_USERNAME:?GQE_NVCOMP_DEPLOY_USERNAME is required}"
  : "${GQE_NVCOMP_DEPLOY_TOKEN:?GQE_NVCOMP_DEPLOY_TOKEN is required}"
  # Only internal builds fetch the private nvCOMP fork. The masked project
  # variables and tokened rewrite rules are unnecessary for public builds.
  git config --global \
    url."https://${GQE_NVCOMP_DEPLOY_USERNAME}:${GQE_NVCOMP_DEPLOY_TOKEN}@gitlab-master.nvidia.com/".insteadOf \
    "https://gitlab-master.nvidia.com/"
  git config --global --add \
    url."https://${GQE_NVCOMP_DEPLOY_USERNAME}:${GQE_NVCOMP_DEPLOY_TOKEN}@gitlab-master.nvidia.com/".insteadOf \
    "ssh://git@gitlab-master.nvidia.com:12051/"
  cmake_args+=(-DGQE_ENABLE_PUBLIC_BUILD=OFF)
fi

build_dir=${GQE_CI_BUILD_DIR:-build}
mkdir -p "${build_dir}"
pushd "${build_dir}"
# Shared CPM lock/hash files must remain writable by the cache's team group.
# The cache directory's setgid bit supplies that group. Include Ninja because
# it can rerun CMake; keep the caller's umask and credential setup unchanged.
(
  umask 002
  cmake "${cmake_args[@]}" ..
  ninja -j30
)
popd

ccache -s
