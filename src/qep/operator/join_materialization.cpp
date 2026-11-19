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

#include <gqe/qep/operator/join_materialization.hpp>

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe::qep {
namespace {

void validate_threshold(double threshold)
{
  GQE_EXPECTS(std::isfinite(threshold) && threshold >= 0.0 && threshold <= 1.0,
              "join materialization threshold must be finite and in [0, 1]",
              std::invalid_argument);
}

}  // namespace

state_container materialize_selective_join_output(state_container_view inputs,
                                                  double threshold,
                                                  query_stream const& stream_handle,
                                                  rmm::device_async_resource_ref mr)
{
  validate_threshold(threshold);
  if (try_row_count(inputs).has_value()) { return make_mutable_state_copy(inputs); }
  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value() && !logical->valid_mask().has_value(),
              "materialize_selective_join_output: input must be an unmasked logical relation");
  auto result = make_mutable_state_copy(inputs);
  if (threshold == 0.0) { return result; }

  std::vector<cudf::size_type> gathered_columns;
  for (cudf::size_type index = 0; index < logical->num_columns(); ++index) {
    auto const* indirect = try_get_indirect_column(logical->column_state(index));
    if (indirect == nullptr) { continue; }
    auto const& source       = indirect->gather_source();
    auto const selected_rows = source->position_list()->column().size();
    auto const source_rows   = source->relation_source()->row_count();
    if (selected_rows == 0 ||
        static_cast<double>(selected_rows) < threshold * static_cast<double>(source_rows)) {
      gathered_columns.push_back(index);
    }
  }
  if (gathered_columns.empty()) { return result; }

  // `materialize_columns` groups the requests by gather source and orders the release of source
  // storage after its reads, including when the caller releases `inputs` immediately.
  auto materialized =
    materialize_columns(*logical, gathered_columns, stream_handle, mr).release_column_states();
  for (std::size_t i = 0; i < gathered_columns.size(); ++i) {
    result[static_cast<std::size_t>(gathered_columns[i])] = std::move(materialized[i]);
  }
  return result;
}

join_materialization_task::join_materialization_task(double threshold) : _threshold{threshold}
{
  validate_threshold(threshold);
}

std::optional<state_container> join_materialization_task::next(
  state_container_view inputs,
  context_reference,
  query_stream const& stream_handle,
  rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::join_materialization_task::next"};
  return materialize_selective_join_output(inputs, _threshold, stream_handle, mr);
}

std::unique_ptr<task> join_materialization_task::clone() const
{
  return std::make_unique<join_materialization_task>(_threshold);
}

}  // namespace gqe::qep
