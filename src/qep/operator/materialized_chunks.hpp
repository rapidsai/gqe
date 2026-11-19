/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/qep/state.hpp>
#include <gqe/query_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <optional>
#include <vector>

namespace gqe::qep::detail {

/** @brief Owned concatenated columns and an optional, unnormalized relation mask. */
struct materialized_chunks_result {
  std::optional<cudf::table> table;  ///< Absent only when no chunks were supplied.

  std::optional<cudf::column> mask;  ///< Absent for unmasked input.
};

/**
 * @brief Concatenate plain or masked materialized chunks and release their owners.
 *
 * All chunks must have the same masked or unmasked shape, checked while collecting views.
 * Column schemas must agree, as required by cuDF concatenation. Source owners remain alive
 * through both concatenations and their release is ordered after reads on `stream`.
 * If concatenation throws, the stream is synchronized before source owners are released.
 * Mask values and nullability are preserved for caller-specific normalization.
 *
 * @param[in,out] chunks Retained tables in append order. No concurrent mutation is permitted.
 *   Empty on success or a concatenation failure. Shape validation failures leave the chunks intact.
 * @param[in] stream CUDA stream for concatenation and ordered release.
 * @param[in] mr Memory resource for all device allocations.
 * @return Owned columns and optional mask, or two disengaged optionals for no input chunks.
 * @throws std::logic_error If chunks are not materialized tables or mix masked/unmasked shapes.
 */
[[nodiscard]] materialized_chunks_result concatenate_materialized_chunks(
  std::vector<state_container>& chunks,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr);

}  // namespace gqe::qep::detail
