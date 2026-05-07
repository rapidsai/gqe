/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <gqe/utility/error.hpp>

#include <mpi.h>

#include <string>
#include <type_traits>
#include <vector>

namespace gqe::utility {
namespace multi_process {

template <typename T, std::enable_if_t<std::is_pointer_v<T>>* = nullptr>
std::vector<T> all_gather_ptr(MPI_Comm comm, T send_ptr)
{
  int world_size = 0;
  GQE_MPI_TRY(MPI_Comm_size(comm, &world_size));

  // All gather PGAS base pointer allocations
  auto base_ptrs = std::vector<T>(world_size);
  GQE_MPI_TRY(
    MPI_Allgather(&send_ptr, sizeof(T), MPI_CHAR, base_ptrs.data(), sizeof(T), MPI_CHAR, comm));
  return base_ptrs;
}

std::vector<int> all_gather_int(MPI_Comm comm, int num);

std::vector<std::string> all_gather_string(MPI_Comm comm, std::string const& send_str);

// FIXME: We can query the communicator object to get rank easily, once we fix the issue:
// https://gitlab-master.nvidia.com/Devtech-Compute/gqe/-/issues/176
// We can then remove these helper functions

/**
 * @brief Check if the current process is rank 0.
 */
inline bool mpi_rank_zero(MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;
  GQE_MPI_TRY(MPI_Comm_rank(comm, &rank));
  return rank == 0;
}

/**
 * @brief Get the rank of the current process.
 */
inline int mpi_rank(MPI_Comm comm = MPI_COMM_WORLD)
{
  int rank;
  GQE_MPI_TRY(MPI_Comm_rank(comm, &rank));
  return rank;
}

}  // namespace multi_process
}  // namespace gqe::utility
