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

#include "materialized_chunks.hpp"

#include <gqe/qep/shapes/masked_table.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/concatenate.hpp>

#include <cstddef>
#include <iterator>
#include <numeric>
#include <utility>

namespace gqe::qep::detail {

materialized_chunks_result concatenate_materialized_chunks(std::vector<state_container>& chunks,
                                                           query_stream const& stream_handle,
                                                           rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  if (chunks.empty()) { return {}; }

  std::vector<cudf::table_view> tables;
  std::vector<cudf::column_view> masks;
  tables.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    if (auto const masked = masked_table_view::try_from(chunk)) {
      tables.push_back(masked->columns);
      masks.push_back(masked->mask);
    } else {
      auto const table = to_table_view(chunk);
      GQE_EXPECTS(table.has_value(),
                  "concatenate_materialized_chunks: chunk is not a materialized table");
      tables.push_back(*table);
    }
  }
  GQE_EXPECTS(masks.empty() || masks.size() == chunks.size(),
              "concatenate_materialized_chunks: mixed masked/unmasked chunks are unsupported");

  state_container release_batch;
  release_batch.reserve(std::accumulate(
    chunks.begin(), chunks.end(), std::size_t{0}, [](std::size_t size, auto const& chunk) {
      return size + chunk.size();
    }));
  for (auto& chunk : chunks) {
    release_batch.insert(release_batch.end(),
                         std::make_move_iterator(chunk.begin()),
                         std::make_move_iterator(chunk.end()));
  }
  chunks.clear();

  materialized_chunks_result result;
  try {
    result.table.emplace(std::move(*cudf::concatenate(tables, stream, mr)));
    if (!masks.empty()) { result.mask.emplace(std::move(*cudf::concatenate(masks, stream, mr))); }
  } catch (...) {
    // Queued reads must finish before the retained source owners unwind.
    auto const synchronization_error = cudaStreamSynchronize(stream.get());
    if (synchronization_error != cudaSuccess) { GQE_CUDA_TRY_NO_THROW(synchronization_error); }
    throw;
  }
  order_deallocation_after_stream_use(release_batch, stream_handle);
  return result;
}

}  // namespace gqe::qep::detail
