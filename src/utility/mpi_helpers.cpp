/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <mpi.h>

#include <gqe/utility/error.hpp>
#include <gqe/utility/mpi_helpers.hpp>

namespace gqe::utility {

namespace multi_process {

std::vector<int> all_gather_int(MPI_Comm comm, int num)
{
  int world_size = 0;
  GQE_MPI_TRY(MPI_Comm_size(comm, &world_size));

  // All gather ranks
  auto ranks = std::vector<int>(world_size);
  GQE_MPI_TRY(MPI_Allgather(&num, 1, MPI_INT, ranks.data(), 1, MPI_INT, comm));
  return ranks;
}

std::vector<std::string> all_gather_string(MPI_Comm comm, std::string const& send_str)
{
  int world_size = 0;
  GQE_MPI_TRY(MPI_Comm_size(comm, &world_size));

  auto string_lens      = std::vector<int32_t>(world_size);
  auto local_string_len = send_str.size();
  GQE_MPI_TRY(MPI_Allgather(&local_string_len, 1, MPI_INT, string_lens.data(), 1, MPI_INT, comm));

  // Prepare displacements and total size
  std::vector<int> displs(world_size, 0);
  int total_len = string_lens[0];
  for (int i = 1; i < world_size; ++i) {
    displs[i] = displs[i - 1] + string_lens[i - 1];
    total_len += string_lens[i];
  }

  // Allgatherv for string characters
  std::vector<char> all_chars(total_len);
  MPI_Allgatherv(send_str.data(),
                 local_string_len,
                 MPI_CHAR,
                 all_chars.data(),
                 string_lens.data(),
                 displs.data(),
                 MPI_CHAR,
                 comm);

  // Deserialize strings
  std::vector<std::string> gathered_strings(world_size);
  for (int i = 0; i < world_size; ++i) {
    gathered_strings[i] = std::string(&all_chars[displs[i]], string_lens[i]);
  }
  return gathered_strings;
}
}  // namespace multi_process

}  // namespace gqe::utility
