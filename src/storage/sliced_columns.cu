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
#include <gqe/utility/error.hpp>

#include <cub/device/device_transform.cuh>
#include <cub/thread/thread_search.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <thrust/iterator/counting_iterator.h>

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
void adjust_offsets(offsets_type* offsets,
                    size_t num_offsets,
                    size_t num_partitions,
                    const cudf::size_type* partition_row_offsets,
                    const offsets_type* partition_char_offsets,
                    size_t char_array_size,
                    cuda::stream_ref stream)
{
  const int block_dim = 128;
  const int grid_dim  = gqe::utility::divide_round_up(num_offsets, block_dim);
  adjust_offsets_kernel<<<grid_dim, block_dim, 0, stream.get()>>>(offsets,
                                                                  num_offsets,
                                                                  num_partitions,
                                                                  partition_row_offsets,
                                                                  partition_char_offsets,
                                                                  char_array_size);
}

// Explicit template instantiations
template void adjust_offsets<int32_t>(int32_t* offsets,
                                      size_t num_offsets,
                                      size_t num_partitions,
                                      const cudf::size_type* partition_row_offsets,
                                      const int32_t* partition_char_offsets,
                                      size_t char_array_size,
                                      cuda::stream_ref stream);

template void adjust_offsets<int64_t>(int64_t* offsets,
                                      size_t num_offsets,
                                      size_t num_partitions,
                                      const cudf::size_type* partition_row_offsets,
                                      const int64_t* partition_char_offsets,
                                      size_t char_array_size,
                                      cuda::stream_ref stream);

void widen_string_offsets(int64_t* destination,
                          int32_t const* source,
                          size_t num_offsets,
                          cuda::stream_ref stream)
{
  GQE_EXPECTS(num_offsets == 0 || (destination != nullptr && source != nullptr),
              "widen_string_offsets requires source and destination when there are offsets to "
              "widen");
  if (num_offsets == 0) { return; }

  GQE_CUDA_TRY(cub::DeviceTransform::Transform(
    source,
    destination,
    num_offsets,
    [] __device__(int32_t offset) { return static_cast<int64_t>(offset); },
    stream.get()));
}

template <typename offsets_type>
void rebase_string_offsets(offsets_type* offsets,
                           size_t num_offsets,
                           offsets_type first_offset,
                           cuda::stream_ref stream)
{
  GQE_EXPECTS(num_offsets == 0 || offsets != nullptr,
              "rebase_string_offsets requires offsets when there are offsets to rebase");

  // An unsliced column already starts at character zero, and an empty one has nothing to rebase.
  if (num_offsets == 0 || first_offset == 0) { return; }

  // Each offset is rebased from the offset at the same index, so reading and writing the same
  // buffer stays within one element and needs no separate destination.
  GQE_CUDA_TRY(cub::DeviceTransform::Transform(
    thrust::make_counting_iterator(size_t{0}),
    offsets,
    num_offsets,
    [offsets, first_offset] __device__(size_t index) { return offsets[index] - first_offset; },
    stream.get()));
}

template void rebase_string_offsets<int32_t>(int32_t* offsets,
                                             size_t num_offsets,
                                             int32_t first_offset,
                                             cuda::stream_ref stream);

template void rebase_string_offsets<int64_t>(int64_t* offsets,
                                             size_t num_offsets,
                                             int64_t first_offset,
                                             cuda::stream_ref stream);

namespace {

/**
 * @brief Return one word of a null mask that is rebased onto @p begin_bit.
 *
 * The algorithm is taken from `cudf::detail::get_mask_offset_word`, which `cudf::copy_bitmask`
 * uses for the same purpose. GQE does not call it, because it is declared in a cuDF detail header.
 *
 * @param[in] source Mask of the source column, indexed from its own bit zero.
 * @param[in] destination_word_index Index of the word to return, within the destination.
 * @param[in] begin_bit Index of the first row of the slice, within @p source.
 * @param[in] end_bit Index one past its last row, within @p source.
 * @return The rebased word.
 */
__device__ inline cudf::bitmask_type rebased_mask_word(cudf::bitmask_type const* source,
                                                       cudf::size_type destination_word_index,
                                                       cudf::size_type begin_bit,
                                                       cudf::size_type end_bit)
{
  auto const source_word_index = destination_word_index + cudf::word_index(begin_bit);
  auto const end_word_index    = cudf::word_index(end_bit - 1);
  auto const current_word      = source[source_word_index];
  auto const next_word =
    end_word_index > source_word_index ? source[source_word_index + 1] : cudf::bitmask_type{0};
  return __funnelshift_r(current_word, next_word, begin_bit);
}

}  // namespace

void rebase_null_mask(cudf::bitmask_type* destination,
                      cudf::bitmask_type const* source,
                      cudf::size_type begin_bit,
                      cudf::size_type end_bit,
                      cuda::stream_ref stream)
{
  auto const num_words = cudf::num_bitmask_words(end_bit - begin_bit);
  GQE_EXPECTS(num_words == 0 || (destination != nullptr && source != nullptr),
              "rebase_null_mask requires both masks when there are words to rebase");

  if (num_words == 0) { return; }

  GQE_CUDA_TRY(cub::DeviceTransform::Transform(
    thrust::make_counting_iterator(cudf::size_type{0}),
    destination,
    num_words,
    [source, begin_bit, end_bit] __device__(cudf::size_type destination_word_index) {
      return rebased_mask_word(source, destination_word_index, begin_bit, end_bit);
    },
    stream.get()));
}

}  // namespace storage

}  // namespace gqe
