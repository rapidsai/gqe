#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Image jobs run directly on a native-architecture Computelab node. They are
# created when image inputs change or a rebuild is explicitly requested.
: "${CI_REGISTRY:?CI_REGISTRY is required}"
: "${CI_REGISTRY_USER:?CI_REGISTRY_USER is required}"
: "${CI_REGISTRY_PASSWORD:?CI_REGISTRY_PASSWORD is required}"
: "${DOCKER_IMAGE:?DOCKER_IMAGE is required}"

docker_config_dir="${DOCKER_CONFIG:-${CI_PROJECT_DIR:-${PWD}}/.ci-cache/docker_config}"
mkdir -p "${docker_config_dir}"
chmod 700 "${docker_config_dir}"
export DOCKER_CONFIG="${docker_config_dir}"

# GitLab's job-scoped registry credentials work non-interactively through
# --password-stdin and remain in the job-local Docker config above.
printf '%s' "${CI_REGISTRY_PASSWORD}" | docker login \
  -u "${CI_REGISTRY_USER}" \
  --password-stdin \
  "${CI_REGISTRY}"

# Reuse the published image when its build inputs have not changed.
# Compare this fingerprint with the label stored on the last published image.
# Keep the input list aligned with image_changes in .gitlab-ci.yml and the
# Dockerfile's COPY instructions.
fingerprint=$(
  {
    uname -m
    sha256sum Dockerfile .dockerignore conda/*.yml scripts/ci/build_image.sh
  } | sha256sum | cut -d ' ' -f 1
)
fingerprint_label=com.nvidia.gqe.ci-inputs-sha256
echo "Image input fingerprint: ${fingerprint}"

# Concurrent pipelines publish to the same branch image tag. An older build
# can finish last and overwrite a newer image, causing a later pipeline to
# rebuild despite unchanged inputs. Avoid overlapping image builds, or cancel
# the earlier pipeline when starting another.
if [[ "${GQE_CI_FORCE_IMAGE_REBUILD:-false}" != "true" ]]; then
  # Read only registry metadata; do not download the image's filesystem layers.
  if published_fingerprint=$(docker buildx imagetools inspect "${DOCKER_IMAGE}" \
    --format '{{ index .Image.Config.Labels "com.nvidia.gqe.ci-inputs-sha256" }}'); then
    if [[ "${published_fingerprint}" == "${fingerprint}" ]]; then
      echo "Reusing ${DOCKER_IMAGE}: image inputs match the published image."
      exit 0
    fi
    echo "Building ${DOCKER_IMAGE}: image inputs changed or the label is missing."
  else
    echo "Building ${DOCKER_IMAGE}: could not inspect the published image."
  fi
else
  echo "Building ${DOCKER_IMAGE}: GQE_CI_FORCE_IMAGE_REBUILD=true."
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

docker build \
  --pull \
  --build-arg "PARALLEL_LEVEL=${GQE_DOCKER_BUILD_PARALLEL_LEVEL:-$(nproc)}" \
  --label "${fingerprint_label}=${fingerprint}" \
  -t "${DOCKER_IMAGE}" \
  .

docker push "${DOCKER_IMAGE}"
