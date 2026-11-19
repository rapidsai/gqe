#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# All phases run on the same host and reuse one authenticated image pull.
source "$(dirname "${BASH_SOURCE[0]}")/prepare_computelab_docker.sh"

bash scripts/ci/run_in_computelab_docker.sh bash scripts/ci/code_style.sh

if [[ "${GQE_CI_BUILD_TEST:-false}" != "true" ]]; then
  exit 0
fi

GQE_CI_PUBLIC_BUILD=false GQE_CI_USE_BUILD_CACHE=true \
  bash scripts/ci/run_in_computelab_docker.sh bash scripts/ci/build_gqe.sh

GQE_CI_ENABLE_NUMA_BIND=true GQE_CI_REQUIRE_GPU=true GQE_CI_REQUIRE_TPCH=true \
  bash scripts/ci/run_in_computelab_docker.sh bash scripts/ci/tpch_queries.sh

if [[ "${GQE_CI_BUILD_PUBLIC_VARIANT:-false}" == "true" ]]; then
  # Preserve the tested internal build; the public variant has its own tree.
  GQE_CI_PUBLIC_BUILD=true GQE_CI_USE_BUILD_CACHE=true GQE_CI_BUILD_DIR=build-public \
    bash scripts/ci/run_in_computelab_docker.sh bash scripts/ci/build_gqe.sh
fi
