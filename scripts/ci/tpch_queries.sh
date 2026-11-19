#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Test the local internal build on the GPU allocated to this job.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=scripts/ci/gpu_detection.sh
source "${script_dir}/gpu_detection.sh"

git config --global --add safe.directory "${PWD}" || true

set +u
source /conda/bin/activate gqe
set -u

# NVSHMEM's UID bootstrap uses cuMemCreate (VMM), which requires unlimited
# memlock. Fail here so a broken container limit does not surface later as an
# unrelated NVSHMEM runtime error.
echo "memlock before:$(ulimit -l)"
if ! ulimit -l unlimited; then
  echo "Failed to set unlimited memlock; NVSHMEM tests require it." >&2
  echo "Ensure Docker starts the container with --ulimit memlock=-1." >&2
  exit 1
fi
echo "memlock after:$(ulimit -l)"

hostname
nvidia-smi

detect_visible_gpus
df -h /dev/shm

# Enable NVTX injection for CUPTI activity profiling. This is required to
# capture the nvtx markers. The soname major version tracks the CUDA toolkit in
# the conda env, so resolve it rather than pinning it; an injection path that
# does not exist fails silently and records no timings.
for cupti_lib in /conda/envs/gqe/lib/libcupti.so.[0-9][0-9] \
                 /conda/envs/gqe/lib/libcupti.so; do
  [[ -e ${cupti_lib} ]] && break
done
if [[ ! -e ${cupti_lib} ]]; then
  echo "No libcupti found under /conda/envs/gqe/lib; NVTX profiling would record nothing." >&2
  exit 1
fi
export NVTX_INJECTION64_PATH=${cupti_lib}

if [[ ! -d build ]]; then
  echo "build/ is missing; run the internal build before tests" >&2
  exit 1
fi

# Run a simple smoktest to ensure that GPUs can be seen and memory can be
# allocated and freed. If this fails, then there is problem with the node
# rather than GQE.
cuda_smoke_test=build/test/cuda_runtime_smoke_test
if [[ ! -x "${cuda_smoke_test}" ]]; then
  echo "${cuda_smoke_test} is missing; the internal build is incomplete" >&2
  exit 1
fi

if ! "${cuda_smoke_test}"; then
  echo "The allocated GPU failed the CUDA runtime smoke test on $(hostname)." >&2
  echo "GitLab job: ${CI_JOB_ID:-unknown}; Slurm job: ${SLURM_JOB_ID:-unknown}" >&2
  nvidia-smi -q || true
  exit 1
fi

# Set the initial query memory to 2 GiB for the tests. This is enough and allows
# greater concurrency for the tests. Set task manager memory to 1 GiB for Flight
# SQL lifecycle and integration tests (SF0.01).
pushd build
# Preserve the existing exclusion for the optional customized Parquet reader
# until that test is validated in CI.
GQE_INITIAL_QUERY_MEMORY=$((2 * 1024 * 1024 * 1024)) \
  GQE_MAX_QUERY_MEMORY=$((2 * 1024 * 1024 * 1024)) \
  GQE_MAX_TASK_MANAGER_MEMORY=$((1 * 1024 * 1024 * 1024)) \
  TPCH_DATA_PATH=/tpch_scratch/datasets/sf0.01/sf0.01_id64 \
  TPCH_QUERIES="${GQE_CI_TPCH_SQL_DIR:-/home/gqe/ci/sql}" \
  TPCH_PHYSICAL_PLANS=/tpch_scratch/physical_plans/sf0.01 \
  TPCH_REF_RESULTS=/tpch_scratch/reference_results/sf0.01 \
  TPCH_SCHEMA=ci_schema.sql \
  ctest --output-on-failure --exclude-regex '^parquet_reader$'
popd

# TPC-H coverage for SF0.01.
GQE_INITIAL_QUERY_MEMORY=$((1024 * 1024 * 1024)) \
  GQE_MAX_QUERY_MEMORY=$((2 * 1024 * 1024 * 1024)) \
  GQE_LOG_LEVEL=warn \
  bash ./test/end_to_end/run_tpc.sh h ./build \
    /tpch_scratch/datasets/sf0.01/sf0.01_id64/ \
    /tpch_scratch/reference_results/sf0.01 \
    /tpch_scratch/substrait_plans/sf0.01/df_q%d.bin
GQE_INITIAL_QUERY_MEMORY=$((1024 * 1024 * 1024)) \
  GQE_MAX_QUERY_MEMORY=$((2 * 1024 * 1024 * 1024)) \
  GQE_LOG_LEVEL=warn \
  bash ./test/end_to_end/run_tpc.sh h ./build \
    /tpch_scratch/datasets/sf0.01/sf0.01_id64/ \
    /tpch_scratch/reference_results/sf0.01 \
    /tpch_scratch/substrait_plans/sf0.01/df_q%d.bin \
    qep_task_graph
