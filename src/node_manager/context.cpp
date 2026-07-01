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

#include <gqe/types.hpp>
#include <gqe/utility/logger.hpp>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda/memory_resource>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <sys/statvfs.h>
#include <sys/sysinfo.h>

#include <format>
#include <limits>
#include <string_view>
#include <tuple>

namespace gqe::node_manager {

context::context(optimization_parameters params)
  : task_manager_context(
      std::move(params),
      cuda::mr::any_resource<cuda::mr::device_accessible>{rmm::mr::cuda_memory_resource{}})
{
  // Create the boost IPC shared memory segment. This must happen before
  // get_table_memory_resource() which opens it with open_only.
  constexpr int max_shm_percentage = 99;
  constexpr int max_ram_percentage = 90;
  constexpr std::size_t one_mb     = 1024 * 1024;

  auto pool_bytes = _optimization_parameters.max_task_manager_memory;
  if (pool_bytes == std::numeric_limits<std::size_t>::max()) {
    struct statvfs stat;
    struct sysinfo si;
    if (statvfs("/dev/shm", &stat) == 0 && sysinfo(&si) == 0) {
      // f_bavail is in units of f_frsize per POSIX statvfs(3)
      auto shm_available = static_cast<std::size_t>(stat.f_bavail) * stat.f_frsize;
      auto total_ram     = static_cast<std::size_t>(si.totalram) * si.mem_unit;
      auto max_shm_bytes = shm_available * max_shm_percentage / 100;
      auto max_ram_bytes = total_ram * max_ram_percentage / 100;
      pool_bytes         = std::min(max_shm_bytes, max_ram_bytes);
      GQE_LOG_INFO(
        "Detected {} MB available in /dev/shm, {} MB total RAM, "
        "using {} MB (min of {}% shm, {}% ram)",
        shm_available / one_mb,
        total_ram / one_mb,
        pool_bytes / one_mb,
        max_shm_percentage,
        max_ram_percentage);
    } else {
      pool_bytes = _optimization_parameters.initial_task_manager_memory;
      GQE_LOG_WARN("Could not determine /dev/shm size or total RAM, falling back to {} MB",
                   pool_bytes / one_mb);
    }
  }

  try {
    // Restrict the segment to the creating user (owner read/write only): the task
    // managers we spawn share our uid and still attach, while other local
    // processes are boxed out.
    boost::interprocess::permissions shm_permissions;
    shm_permissions.set_permissions(0600);

    boost::interprocess::managed_shared_memory segment(
      boost::interprocess::create_only, shared_memory_name, pool_bytes, nullptr, shm_permissions);
  } catch (boost::interprocess::interprocess_exception const& e) {
    throw std::runtime_error(
      std::format("Failed to create shared memory segment '{}': {}. "
                  "Is another node manager running? "
                  "If not, remove the stale segment with: `rm /dev/shm/{}`",
                  shared_memory_name,
                  e.what(),
                  shared_memory_name));
  }
  GQE_LOG_INFO("Created boost shared memory segment ({} MB)", pool_bytes / one_mb);

  // From this point, _shm_remover owns cleanup — it removes the segment when
  // this context is destroyed, or if the remainder of the constructor throws.
  _shm_remover.emplace(shared_memory_name);

  // Eagerly initialize the boost shared memory resource so the segment is
  // mapped and pinned before task managers are spawned.
  std::ignore = get_table_memory_resource(memory_kind::boost_shared{});
}

}  // namespace gqe::node_manager
