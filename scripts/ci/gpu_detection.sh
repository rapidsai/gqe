#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared GPU discovery preserves the pre-Computelab CI policy: compile for GPUs
# with at least 8 GiB and hide smaller devices that cannot host the NVSHMEM pool.
detect_cuda_arch()
{
  if [[ -n "${GQE_CI_CUDA_ARCHITECTURES:-}" ]]; then
    CUDA_ARCH="${GQE_CI_CUDA_ARCHITECTURES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_info
    if ! gpu_info=$(nvidia-smi --query-gpu=compute_cap,memory.total --format=csv,noheader,nounits); then
      echo "Failed to query GPUs for CUDA arch detection with nvidia-smi" >&2
      return 1
    fi
    CUDA_ARCH=$(
      printf '%s\n' "${gpu_info}" |
        awk -F', ' '$2 >= 8192 {gsub(/\./, "", $1); print $1"-real"}' |
        sort -u |
        paste -sd';'
    )
    if [[ -z "${CUDA_ARCH}" ]]; then
      echo "No GPUs with >=8 GiB detected for CUDA arch detection; set GQE_CI_CUDA_ARCHITECTURES explicitly" >&2
      return 1
    fi
  else
    echo "GQE_CI_CUDA_ARCHITECTURES is required when building without nvidia-smi" >&2
    return 1
  fi

  export CUDA_ARCH
  echo "Building for CUDA architectures: ${CUDA_ARCH}"
}

detect_visible_gpus()
{
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required to detect visible GPUs" >&2
    return 1
  fi

  # Ensure CUDA device indices match nvidia-smi (PCI bus ID order). Needed here
  # because test binaries use NVSHMEM but run via ctest with gtest's main(), not
  # through node_manager/task_manager.
  export CUDA_DEVICE_ORDER=PCI_BUS_ID

  local gpu_info
  if ! gpu_info=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits); then
    echo "Failed to query visible GPUs with nvidia-smi" >&2
    return 1
  fi

  # Select GPUs with at least 8 GiB of memory.
  CUDA_VISIBLE_DEVICES=$(
    printf '%s\n' "${gpu_info}" |
      awk -F', ' '$2 >= 8192 {printf sep $1; sep=","}'
  )
  if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "No GPUs with >=8 GiB detected; refusing to run with empty CUDA_VISIBLE_DEVICES" >&2
    return 1
  fi
  export CUDA_VISIBLE_DEVICES
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
}
