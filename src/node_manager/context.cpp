/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/node_manager/context.hpp>

#include <gqe/storage/segment_layout.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/linux.hpp>
#include <gqe/utility/logger.hpp>

#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda/memory_resource>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <numa.h>
#include <sys/statvfs.h>

#include <cerrno>
#include <exception>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>

namespace gqe::node_manager {

namespace {

/**
 * @brief Return the error reported for a shared memory segment that could not be created.
 *
 * @param[in] cause Error the segment's creation threw.
 */
std::runtime_error creation_failure(std::exception const& cause)
{
  return std::runtime_error(std::format(
    "Failed to create shared memory segment '{}': {}", shared_memory_name, cause.what()));
}

}  // namespace

context::context(optimization_parameters params, segment_layout_policy_type policy)
  : task_manager_context(
      std::move(params),
      cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}})
{
  // Create the IPC shared memory segment. This must happen before
  // get_table_memory_resource() which borrows the created mapping.
  constexpr int max_shm_percentage = 99;
  constexpr int max_ram_percentage = 90;
  constexpr std::size_t one_mb     = 1024 * 1024;

  auto pool_bytes = _optimization_parameters.max_task_manager_memory;
  if (!pool_bytes) {
    auto const cpu_numa_nodes = utility::cpu_bearing_numa_nodes();
    if (cpu_numa_nodes.empty()) {
      throw std::runtime_error(
        "Cannot size the shared-memory pool: no permitted CPU-bearing NUMA nodes were found");
    }
    struct statvfs stat;
    if (statvfs("/dev/shm", &stat) != 0) {
      throw std::system_error(
        errno, std::generic_category(), "Failed to query available space in /dev/shm");
    }
    // f_bavail is in units of f_frsize per POSIX statvfs(3)
    auto shm_available = static_cast<std::size_t>(stat.f_bavail) * stat.f_frsize;
    // Sum RAM over permitted CPU-bearing NUMA nodes.
    std::size_t cpu_ram = 0;
    for (int n = 0; n <= numa_max_node(); ++n) {
      if (cpu_numa_nodes.contains(n)) {
        auto const node_bytes = numa_node_size64(n, nullptr);
        if (node_bytes < 0) {
          throw std::runtime_error(std::format("Failed to get memory size for NUMA node {}", n));
        }
        cpu_ram += static_cast<std::size_t>(node_bytes);
      }
    }
    auto max_shm_bytes = shm_available * max_shm_percentage / 100;
    auto max_ram_bytes = cpu_ram * max_ram_percentage / 100;
    pool_bytes         = std::min(max_shm_bytes, max_ram_bytes);
    GQE_LOG_INFO(
      "Detected {} MB available in /dev/shm, {} MB CPU RAM, "
      "using {} MB (min of {}% shm, {}% ram)",
      shm_available / one_mb,
      cpu_ram / one_mb,
      pool_bytes.value() / one_mb,
      max_shm_percentage,
      max_ram_percentage);
  }

  try {
    // Restrict the segment to the creating user (owner read/write only): the task
    // managers we spawn share our uid and still attach, while other local
    // processes are boxed out.
    boost::interprocess::permissions shm_permissions;
    shm_permissions.set_permissions(0600);

    _shared_memory_segment = std::make_unique<shared_segment>(
      gqe::create_only, shared_memory_name, pool_bytes.value(), policy, shm_permissions);
    // The node manager owns this mapping, so shared_memory_segment() never attaches the poison
    // flag for it. Attach it here to see flags set by other processes and to share its own.
    // The base class destructor detaches it.
    attach_shared_memory_poison_flag(*_shared_memory_segment);
  } catch (std::system_error const& e) {
    if (e.code() != std::errc::file_exists) { throw creation_failure(e); }
    throw std::runtime_error(
      std::format("Failed to create shared memory segment '{}': {}. "
                  "Is another node manager running? "
                  "If not, remove the stale segment with: `rm /dev/shm/{}`",
                  shared_memory_name,
                  e.what(),
                  shared_memory_name));
  } catch (std::runtime_error const& e) {
    throw creation_failure(e);
  }
  GQE_LOG_INFO(
    "Created {} shared memory segment ({} MB)", to_string(policy), pool_bytes.value() / one_mb);

  // From this point, _shm_remover owns cleanup — it removes the segment when
  // this context is destroyed, or if the remainder of the constructor throws.
  _shm_remover.emplace(shared_memory_name);

  // The resource borrows the populated create mapping, which is not registered with CUDA.
  std::ignore = get_table_memory_resource(memory_kind::shared_segment{});
}

}  // namespace gqe::node_manager
