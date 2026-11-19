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

#include <gqe/qep/operator/gather.hpp>

#include <gqe/qep/column_materialization.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <algorithm>
#include <memory>
#include <utility>

namespace gqe {
namespace qep {

std::unique_ptr<task> gather_task::clone() const { return std::make_unique<gather_task>(); }

std::optional<state_container> gather_task::next(state_container_view inputs,
                                                 context_reference,
                                                 query_stream const& stream_handle,
                                                 rmm::device_async_resource_ref mr) const
{
  utility::nvtx_scoped_range range{"qep::gather_task::next"};

  GQE_EXPECTS(std::ranges::none_of(inputs, [](auto const& slot) { return slot == nullptr; }),
              "qep::gather_task::next: input contains a null state slot");

  if (try_row_count(inputs).has_value()) { return make_mutable_state_copy(inputs); }

  auto const logical = logical_table_view::try_from(inputs);
  GQE_EXPECTS(logical.has_value(),
              "qep::gather_task::next: input is not a supported relational shape");
  column_materialization result = materialize_all_compacted_columns(*logical, stream_handle, mr);
  return std::move(result).release_column_states();
}

}  // namespace qep
}  // namespace gqe
