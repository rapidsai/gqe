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

#include <gqe/qep/indirect_gather.hpp>

#include <gqe/utility/error.hpp>

#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <utility>
#include <vector>

namespace gqe {
namespace qep {

position_list_handle compose_position_lists(position_list_handle const& positions,
                                            position_list_handle const& selecting_positions,
                                            query_stream const& stream_handle,
                                            rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(positions != nullptr,
              "compose_position_lists: positions must not be null",
              std::invalid_argument);
  GQE_EXPECTS(selecting_positions != nullptr,
              "compose_position_lists: selecting positions must not be null",
              std::invalid_argument);

  state_container dependencies{make_shared_state(state_kind::position_list{positions}),
                               make_shared_state(state_kind::position_list{selecting_positions})};

  auto gathered = cudf::gather(cudf::table_view{{positions->column().view()}},
                               selecting_positions->column().view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr)
                    ->release();
  GQE_EXPECTS(gathered.size() == 1,
              "compose_position_lists: gathered table has unexpected column count");
  auto result = make_position_list(std::move(gathered.front()), stream_handle);
  order_deallocation_after_stream_use(dependencies, stream_handle);
  return result;
}

position_list_handle compact_position_list(position_list_handle const& positions,
                                           cudf::column_view valid_mask,
                                           query_stream const& stream_handle,
                                           rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(positions != nullptr,
              "compact_position_list: positions must not be null",
              std::invalid_argument);
  GQE_EXPECTS(valid_mask.type().id() == cudf::type_id::BOOL8,
              "compact_position_list: valid_mask must be BOOL8");
  GQE_EXPECTS(valid_mask.size() == positions->column().size(),
              "compact_position_list: valid_mask row count must match positions");

  auto compacted = cudf::apply_retention_mask(
                     cudf::table_view{{positions->column().view()}}, valid_mask, stream, mr)
                     ->release();
  GQE_EXPECTS(compacted.size() == 1,
              "compact_position_list: compacted table has unexpected column count");
  return make_position_list(std::move(compacted.front()), stream_handle);
}

cudf::table materialize_indirect_columns(indirect_gather_source const& source,
                                         std::span<cudf::size_type const> source_column_indices,
                                         position_list_handle const& positions,
                                         query_stream const& stream_handle,
                                         rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(source != nullptr,
              "materialize_indirect_columns: source must not be null",
              std::invalid_argument);
  GQE_EXPECTS(positions != nullptr,
              "materialize_indirect_columns: positions must not be null",
              std::invalid_argument);
  GQE_EXPECTS(!source_column_indices.empty(),
              "materialize_indirect_columns: at least one source column is required",
              std::invalid_argument);

  auto const num_source_columns = source->relation_source()->num_columns();
  for (auto const index : source_column_indices) {
    GQE_EXPECTS(index >= 0 && index < num_source_columns,
                "materialize_indirect_columns: source column index is out of range",
                std::invalid_argument);
  }

  auto const relation = source->relation_source()->contiguous_table_view();
  auto const indices =
    std::vector<cudf::size_type>(source_column_indices.begin(), source_column_indices.end());
  return std::move(*cudf::gather(
    relation.select(indices), positions->column().view(), source->bounds_policy(), stream, mr));
}

}  // namespace qep
}  // namespace gqe
