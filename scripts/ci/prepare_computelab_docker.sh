#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Source this once per job to prepare the image for all container phases.
set -euo pipefail

: "${DOCKER_IMAGE:?DOCKER_IMAGE is required}"
: "${CI_PROJECT_DIR:?CI_PROJECT_DIR is required}"

# Keep registry credentials local to this job unless explicitly overridden.
docker_config_dir="${DOCKER_CONFIG:-${CI_PROJECT_DIR}/.ci-cache/docker_config}"
mkdir -p "${docker_config_dir}"
chmod 700 "${docker_config_dir}"
DOCKER_CONFIG=$(cd "${docker_config_dir}" && pwd -P)
export DOCKER_CONFIG
echo "Using GQE CI Docker config: ${DOCKER_CONFIG}"

if [[ -n "${CI_REGISTRY:-}" && -n "${CI_REGISTRY_USER:-}" && -n "${CI_REGISTRY_PASSWORD:-}" ]]; then
  printf '%s' "${CI_REGISTRY_PASSWORD}" | docker login \
    -u "${CI_REGISTRY_USER}" \
    --password-stdin \
    "${CI_REGISTRY}"
fi

# Do not hide a missing MR-specific image by testing an unrelated stable image.
docker pull "${DOCKER_IMAGE}"

# Pin every phase to the image we pulled, even if another job updates the tag.
GQE_CI_DOCKER_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "${DOCKER_IMAGE}")
export GQE_CI_DOCKER_IMAGE_ID
echo "Using GQE CI Docker image: ${GQE_CI_DOCKER_IMAGE_ID} (${DOCKER_IMAGE})"
