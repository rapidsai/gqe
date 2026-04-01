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

#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace gqe {
namespace storage {

/**
 * @brief Adjust concatenated string offsets by the char-buffer base of their copied partition.
 */
template <typename offsets_type>
void adjust_offsets_api(offsets_type* offsets,
                        size_t num_offsets,
                        size_t num_partitions,
                        const cudf::size_type* partition_row_offsets,
                        const offsets_type* partition_char_offsets,
                        size_t char_array_size,
                        rmm::cuda_stream_view stream);

// Extern template declarations for adjust_offsets_api
extern template void adjust_offsets_api<int32_t>(int32_t* offsets,
                                                 size_t num_offsets,
                                                 size_t num_partitions,
                                                 const cudf::size_type* partition_row_offsets,
                                                 const int32_t* partition_char_offsets,
                                                 size_t char_array_size,
                                                 rmm::cuda_stream_view stream);

extern template void adjust_offsets_api<int64_t>(int64_t* offsets,
                                                 size_t num_offsets,
                                                 size_t num_partitions,
                                                 const cudf::size_type* partition_row_offsets,
                                                 const int64_t* partition_char_offsets,
                                                 size_t char_array_size,
                                                 rmm::cuda_stream_view stream);

}  // namespace storage
}  // namespace gqe
