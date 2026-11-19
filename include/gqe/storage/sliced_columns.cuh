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

#include <cuda/stream>
#include <cudf/types.hpp>

#include <cstdint>

namespace gqe {
namespace storage {

/**
 * @brief Adjust concatenated string offsets by the char-buffer base of their copied partition.
 */
template <typename offsets_type>
void adjust_offsets(offsets_type* offsets,
                    size_t num_offsets,
                    size_t num_partitions,
                    const cudf::size_type* partition_row_offsets,
                    const offsets_type* partition_char_offsets,
                    size_t char_array_size,
                    cuda::stream_ref stream);

// Extern template declarations for adjust_offsets
extern template void adjust_offsets<int32_t>(int32_t* offsets,
                                             size_t num_offsets,
                                             size_t num_partitions,
                                             const cudf::size_type* partition_row_offsets,
                                             const int32_t* partition_char_offsets,
                                             size_t char_array_size,
                                             cuda::stream_ref stream);

extern template void adjust_offsets<int64_t>(int64_t* offsets,
                                             size_t num_offsets,
                                             size_t num_partitions,
                                             const cudf::size_type* partition_row_offsets,
                                             const int64_t* partition_char_offsets,
                                             size_t char_array_size,
                                             cuda::stream_ref stream);

/**
 * @brief Widen 32-bit string offsets into a 64-bit destination.
 *
 * This operation is asynchronous. The source and destination must remain valid and unmodified by
 * the caller until the operation completes on @p stream.
 *
 * @param[out] destination Device-accessible destination with @p num_offsets entries.
 * @param[in] source Device-accessible 32-bit offsets with @p num_offsets entries.
 * @param[in] num_offsets Number of offsets to widen.
 * @param[in] stream CUDA stream used for the transformation.
 *
 * @pre @p destination and @p source are not null when @p num_offsets is positive, which
 * GQE_EXPECTS enforces.
 */
void widen_string_offsets(int64_t* destination,
                          int32_t const* source,
                          size_t num_offsets,
                          cuda::stream_ref stream);

/**
 * @brief Rebase string offsets onto the first character of their own column.
 *
 * The offsets of a sliced strings column address the character buffer of the source, so their
 * first entry is the character at which the slice begins rather than zero. Storing the slice
 * copies the characters from that entry onwards, hence the stored offsets have to be rebased by
 * subtracting it.
 *
 * @tparam offsets_type Type of the offsets, which cuDF defines as INT32 or INT64.
 *
 * @param[in,out] offsets Offsets to rebase, in device-accessible memory.
 * @param[in] num_offsets Number of offsets, which is one more than the number of rows.
 * @param[in] first_offset Character offset at which the slice begins.
 * @param[in] stream CUDA stream used for the kernel launch.
 *
 * @pre @p offsets is not null when @p num_offsets is positive, which GQE_EXPECTS enforces.
 */
template <typename offsets_type>
void rebase_string_offsets(offsets_type* offsets,
                           size_t num_offsets,
                           offsets_type first_offset,
                           cuda::stream_ref stream);

// Extern template declarations for rebase_string_offsets
extern template void rebase_string_offsets<int32_t>(int32_t* offsets,
                                                    size_t num_offsets,
                                                    int32_t first_offset,
                                                    cuda::stream_ref stream);

extern template void rebase_string_offsets<int64_t>(int64_t* offsets,
                                                    size_t num_offsets,
                                                    int64_t first_offset,
                                                    cuda::stream_ref stream);

/**
 * @brief Rebase a null mask onto the first row of its own column.
 *
 * The mask of a sliced column addresses the rows of the source, so the validity of row zero of the
 * slice is bit @p begin_bit rather than bit zero. Storing the slice copies the mask from that bit
 * onwards, which shifts every word.
 *
 * The bits at and above the row count of the slice are undefined in the destination, as they are
 * in any cuDF null mask. The destination holds `cudf::num_bitmask_words(end_bit - begin_bit)`
 * words, and any padding beyond them is left untouched.
 *
 * @param[out] destination Rebased mask, in device-accessible memory.
 * @param[in] source Mask of the source column, indexed from its own bit zero.
 * @param[in] begin_bit Index of the first row of the slice, within @p source.
 * @param[in] end_bit Index one past its last row, within @p source.
 * @param[in] stream CUDA stream used for the kernel launch.
 *
 * @pre Neither @p destination nor @p source is null when the slice covers at least one mask word,
 * which GQE_EXPECTS enforces.
 */
void rebase_null_mask(cudf::bitmask_type* destination,
                      cudf::bitmask_type const* source,
                      cudf::size_type begin_bit,
                      cudf::size_type end_bit,
                      cuda::stream_ref stream);

}  // namespace storage
}  // namespace gqe
