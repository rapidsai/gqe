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

#include <gqe/storage/in_memory.hpp>

#include <cub/thread/thread_search.cuh>

namespace gqe {

namespace storage {

template <typename offsets_type>
__global__ void adjust_offsets_kernel(offsets_type* offsets,
                                      size_t num_offsets,
                                      size_t num_partitions,
                                      const cudf::size_type* partition_row_offsets,
                                      const offsets_type* partition_char_offsets,
                                      size_t char_array_size)
{
  int ix_thread = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_size = gridDim.x * blockDim.x;

  for (int ix = ix_thread; ix < num_offsets; ix += grid_size) {
    if (ix == num_offsets - 1) {
      // Last thread will also fill in the final offset value (non-inclusive upper bound)
      offsets[num_offsets] = char_array_size;
    }

    // `partition_row_offsets` contains one row start per copied partition. `UpperBound()` returns
    // the first partition that starts after row `ix`, or `num_partitions` when `ix` belongs to the
    // last copied partition. Subtracting one therefore yields the copied partition that contains
    // this row and naturally skips duplicate row starts from empty partitions.
    auto const ix_partition =
      cub::UpperBound(partition_row_offsets, num_partitions, static_cast<cudf::size_type>(ix)) - 1;

    if (ix_partition == 0) { continue; }

    offsets[ix] += partition_char_offsets[ix_partition];
  }
}

template <typename offsets_type>
void adjust_offsets_api(offsets_type* offsets,
                        size_t num_offsets,
                        size_t num_partitions,
                        const cudf::size_type* partition_row_offsets,
                        const offsets_type* partition_char_offsets,
                        size_t char_array_size,
                        rmm::cuda_stream_view stream)
{
  const int block_dim = 128;
  const int grid_dim  = gqe::utility::divide_round_up(num_offsets, block_dim);
  adjust_offsets_kernel<<<grid_dim, block_dim, 0, stream>>>(offsets,
                                                            num_offsets,
                                                            num_partitions,
                                                            partition_row_offsets,
                                                            partition_char_offsets,
                                                            char_array_size);
}

// Explicit template instantiations
template void adjust_offsets_api<int32_t>(int32_t* offsets,
                                          size_t num_offsets,
                                          size_t num_partitions,
                                          const cudf::size_type* partition_row_offsets,
                                          const int32_t* partition_char_offsets,
                                          size_t char_array_size,
                                          rmm::cuda_stream_view stream);

template void adjust_offsets_api<int64_t>(int64_t* offsets,
                                          size_t num_offsets,
                                          size_t num_partitions,
                                          const cudf::size_type* partition_row_offsets,
                                          const int64_t* partition_char_offsets,
                                          size_t char_array_size,
                                          rmm::cuda_stream_view stream);

}  // namespace storage

}  // namespace gqe
