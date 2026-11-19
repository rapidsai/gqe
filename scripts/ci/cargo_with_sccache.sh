#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run CI Cargo builds with the persistent sccache cache. An exclusive file lock
# on NFS serializes cache access across nodes. Each invocation uses a private
# server socket and holds the lock until its sccache server exits.

set -euo pipefail

export SCCACHE_DIR=${SCCACHE_DIR:-/sccache_cache} RUSTC_WRAPPER=sccache SCCACHE_IDLE_TIMEOUT=0
node_name=${GQE_CI_NODE_HOSTNAME:-$(hostname)}
lock_pid=
server_pid=
cargo_pid=
socket_dir=
lock_acquired=false
cancel_status=0

# Print a timestamped lock message identifying the node and cache.
# Timestamp format example: 2026-10-21T22:18:46+0000.
log()
{
  printf '%(%FT%T%z)T %s: %s (%s)\n' -1 "${node_name}" "$1" "${SCCACHE_DIR}"
}

# Stop child processes and remove the private socket before releasing the lock.
cleanup()
{
  local status=$?
  trap - EXIT
  trap '' INT TERM
  [[ "${cancel_status}" -eq 0 ]] || status=${cancel_status}
  if [[ -n "${lock_pid}" ]]; then
    kill -TERM "${lock_pid}" 2>/dev/null || true
    wait "${lock_pid}" || true
  fi
  if [[ -n "${cargo_pid}" ]]; then
    # Cargo has its own process group; include compilers and build scripts.
    kill -TERM -- "-${cargo_pid}" 2>/dev/null || kill -TERM "${cargo_pid}" 2>/dev/null || true
    wait "${cargo_pid}" || true
  fi
  if [[ -n "${server_pid}" ]]; then
    if ! sccache --stop-server {lock_fd}>&-; then
      kill -TERM -- "-${server_pid}" 2>/dev/null || kill -TERM "${server_pid}" 2>/dev/null || true
      [[ "${status}" -ne 0 ]] || status=1
    fi
    if ! wait "${server_pid}"; then
      [[ "${status}" -ne 0 ]] || status=1
    fi
  fi
  if [[ -n "${socket_dir}" ]]; then
    if ! rm -f "${socket_dir}/server.sock" || ! rmdir "${socket_dir}"; then
      [[ "${status}" -ne 0 ]] || status=1
    fi
  fi
  if [[ "${lock_acquired}" == true ]]; then
    exec {lock_fd}>&-
    log "Finished sccache shutdown; released lock"
  fi
  exit "${status}"
}
trap cleanup EXIT
# Defer cancellation until the PID of a newly launched child is recorded.
trap 'cancel_status=130' INT
trap 'cancel_status=143' TERM

# Let Bash allocate a descriptor for the shared lock file without truncating it.
# Run flock as a child so cancellation can interrupt the wait for exclusive access.
exec {lock_fd}>>"${SCCACHE_DIR}.lock"
log "Waiting for sccache lock"
flock --exclusive "${lock_fd}" &
lock_pid=$!
status=0
wait "${lock_pid}" || status=$?
[[ "${cancel_status}" -eq 0 ]] || exit "${cancel_status}"
[[ "${status}" -eq 0 ]] || exit "${status}"
lock_pid=
lock_acquired=true
log "Acquired sccache lock"

socket_dir=$(mktemp -d /tmp/gqe-sccache.XXXXXX)
export SCCACHE_SERVER_UDS="${socket_dir}/server.sock" SCCACHE_NO_DAEMON=1
# Only the supervisor and server retain the lock descriptor; Cargo must not inherit it.
SCCACHE_START_SERVER=1 setsid sccache &
server_pid=$!
# Wait up to about 10 seconds for the server socket, checking every 0.1s.
# Stop early if the server exits; Cargo starts only after the socket appears.
for ((attempt = 0; attempt < 100; attempt++)); do
  [[ "${cancel_status}" -eq 0 ]] || exit "${cancel_status}"
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "sccache server exited during startup" >&2
    exit 1
  fi
  [[ ! -S "${SCCACHE_SERVER_UDS}" ]] || break
  sleep 0.1
done
if [[ ! -S "${SCCACHE_SERVER_UDS}" ]]; then
  echo "Timed out waiting for sccache server startup" >&2
  exit 1
fi
[[ "${cancel_status}" -eq 0 ]] || exit "${cancel_status}"

setsid cargo "$@" {lock_fd}>&- &
cargo_pid=$!
[[ "${cancel_status}" -eq 0 ]] || exit "${cancel_status}"
status=0
wait "${cargo_pid}" || status=$?
[[ "${cancel_status}" -eq 0 ]] || exit "${cancel_status}"
exit "${status}"
