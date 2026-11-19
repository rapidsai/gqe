#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Computelab jobs execute on the allocated host, so this wrapper supplies the
# Docker behavior that GitLab's Docker executor previously provided. It keeps
# persistent caches under /home/gqe/ci/cache by default, maps the
# checkout to one stable in-container path, and exposes only the GPU and
# datasets assigned to the job.
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

: "${DOCKER_IMAGE:?DOCKER_IMAGE is required}"
: "${CI_PROJECT_DIR:?CI_PROJECT_DIR is required}"

# run_arch.sh prepares the image once; direct callers still prepare their own.
if [[ -z "${GQE_CI_DOCKER_IMAGE_ID:-}" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/prepare_computelab_docker.sh"
fi

local_cache_root="${CI_PROJECT_DIR}/.ci-cache"
# This path must remain identical in build and test containers because CMake
# records absolute paths.
container_project_dir=/workspace/gqe

# HOME is job-local and shared by the container phases.
mkdir -p "${local_cache_root}/container_home"
chmod 700 "${local_cache_root}/container_home"

local_cache_root=$(cd "${local_cache_root}" && pwd -P)
echo "Using GQE CI local cache root: ${local_cache_root}"
echo "Using stable container project path: ${container_project_dir}"

# Only build jobs opt in; lint and test containers need no dependency/compiler
# cache environment or mounts.
build_cache_args=()
if [[ "${GQE_CI_USE_BUILD_CACHE:-false}" == "true" ]]; then
  cache_project_slug=${CI_PROJECT_PATH_SLUG:-$(basename "${CI_PROJECT_DIR}")}
  cache_project_slug=${cache_project_slug//[^A-Za-z0-9_.-]/-}
  cache_user=${USER:-$(id -un)}
  cache_user=${cache_user//[^A-Za-z0-9_.-]/-}

  # CPM coordinates source-cache population with filesystem locks, so share
  # dependency sources across trusted users.
  # Keep ccache separate per user/project.
  # Cargo package extraction did not serialize reliably across Computelab NFS,
  # so Cargo remains job-local alongside ccache's temporary files.
  # Default to directory under /home/gqe/ci/cache; explicit overrides win.
  # The parent must permit user directory creation and Docker bind-mount traversal.
  cache_base=${GQE_CI_CACHE_BASE:-/home/gqe/ci/cache/${cache_user}}
  cpm_cache_root=${GQE_CI_CPM_CACHE_ROOT:-/home/gqe/ci/cache/cpm_cache}
  # Provision the shared cache with a common team group and mode 2775.
  # Setgid preserves group ownership; build_gqe.sh uses umask 002 to keep
  # new directories and lock/hash files group-writable.
  # Jobs must not chmod shared directories owned by another user.
  if [[ ! -d "${cpm_cache_root}" || ! -r "${cpm_cache_root}" || ! -w "${cpm_cache_root}" || ! -x "${cpm_cache_root}" ]]; then
    echo "${cpm_cache_root} must be provisioned with shared read/write/traversal and inherited team write permissions" >&2
    exit 1
  fi
  cpm_cache_root=$(cd "${cpm_cache_root}" && pwd -P)
  mkdir -p "${cache_base}"
  cache_base=$(cd "${cache_base}" && pwd -P)
  project_cache_root="${cache_base}/${cache_project_slug}"
  sccache_cache_root="${project_cache_root}/sccache_cache/$(uname -m)"
  mkdir -p \
    "${project_cache_root}/ccache_files" \
    "${sccache_cache_root}" \
    "${local_cache_root}/cargo_home" \
    "${local_cache_root}/ccache_tmp"
  chmod 700 \
    "${local_cache_root}/cargo_home" \
    "${local_cache_root}/ccache_tmp"

  # Docker resolves bind-mount sources as the daemon's root user, which is
  # root-squashed on Computelab NFS. Every NFS directory in the source path must
  # therefore be searchable. Mode 0711 preserves owner-only listing/writes while
  # allowing the daemon to traverse the path and mount the cache directory.
  chmod 711 \
    "${cache_base}" \
    "${project_cache_root}" \
    "${project_cache_root}/ccache_files" \
    "${project_cache_root}/sccache_cache" \
    "${sccache_cache_root}"

  project_cache_root=$(cd "${project_cache_root}" && pwd -P)
  touch "${sccache_cache_root}.lock"
  echo "Using GQE CI persistent cache root: ${project_cache_root}"
  echo "Using GQE CI shared CPM cache root: ${cpm_cache_root}"
  echo "Using GQE CI persistent sccache root: ${sccache_cache_root}"
  build_cache_args=(
    -e CCACHE_TEMPDIR=/ccache_tmp
    -e GQE_CI_NODE_HOSTNAME="$(hostname)"
    -v "${cpm_cache_root}:/cpm_cache"
    -v "${project_cache_root}/ccache_files:/ccache_files"
    -v "${local_cache_root}/cargo_home:/cargo_home"
    -v "${local_cache_root}/ccache_tmp:/ccache_tmp"
    -v "${sccache_cache_root}:/sccache_cache"
    -v "${sccache_cache_root}.lock:/sccache_cache.lock"
  )
fi

tpch_mount=()
tpch_data_root=/home/tpc/tpch
tpch_sql_dir=${GQE_CI_TPCH_SQL_DIR:-/home/gqe/ci/sql}

if [[ "${GQE_CI_REQUIRE_TPCH:-false}" == "true" ]]; then
  for required_dir in "${tpch_data_root}" "${tpch_sql_dir}"; do
    if [[ ! -d "${required_dir}" ]]; then
      echo "${required_dir} is required for this job but was not found on the Computelab node" >&2
      exit 1
    fi
  done
  tpch_mount+=(
    -v "${tpch_data_root}:/tpch_scratch:ro"
    -v "${tpch_sql_dir}:${tpch_sql_dir}:ro"
  )
fi

group_args=()
for group_id in $(id -G); do
  group_args+=(--group-add "${group_id}")
done

numa_args=()
if [[ "${GQE_CI_ENABLE_NUMA_BIND:-false}" == "true" ]]; then
  # GQE's NUMA tests use mbind/set_mempolicy. CAP_SYS_NICE admits those calls
  # through Docker's capability-aware default seccomp profile.
  numa_args+=(--cap-add SYS_NICE)
fi

gpu_args=()
if [[ -n "${NV_GPU:-}" ]]; then
  gpu_args=(--gpus "device=${NV_GPU}")
elif [[ "${GQE_CI_REQUIRE_GPU:-false}" == "true" ]]; then
  echo "NV_GPU is required for this job but was not set; check CI_CRUN_QUERY GPU allocation" >&2
  exit 1
fi

nvcomp_args=()
if [[ "${GQE_CI_PUBLIC_BUILD:-true}" != "true" ]]; then
  # Only internal builds fetch the private nvCOMP fork. Do not expose its
  # deploy credentials to lint, test, or public-build containers.
  nvcomp_args=(
    -e GQE_NVCOMP_DEPLOY_USERNAME
    -e GQE_NVCOMP_DEPLOY_TOKEN
  )
fi

# Give each job a private 2 GiB /dev/shm so concurrent containers cannot
# collide through the host IPC namespace.
exec docker run \
  --pull=never \
  --rm \
  --shm-size=2g \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  "${gpu_args[@]}" \
  --user "$(id -u):$(id -g)" \
  "${group_args[@]}" \
  "${numa_args[@]}" \
  -e HOME=/gqe_ci_home \
  -e GQE_CI_CUDA_ARCHITECTURES \
  -e GQE_CI_PUBLIC_BUILD \
  -e GQE_CI_BUILD_DIR \
  -e GQE_CI_TPCH_SQL_DIR \
  "${nvcomp_args[@]}" \
  -v "${CI_PROJECT_DIR}:${container_project_dir}" \
  -v "${local_cache_root}/container_home:/gqe_ci_home" \
  "${build_cache_args[@]}" \
  "${tpch_mount[@]}" \
  -w "${container_project_dir}" \
  "${GQE_CI_DOCKER_IMAGE_ID}" \
  "$@"
