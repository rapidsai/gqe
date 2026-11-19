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

#include "filter_mask.hpp"

#include <gqe/utility/error.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <rmm/exec_policy.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <stdexcept>

namespace gqe::qep::detail {
std::unique_ptr<cudf::column> compose_filter_masks(cudf::column_view previous,
                                                   cudf::column_view predicate,
                                                   query_stream const& stream_handle,
                                                   rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  GQE_EXPECTS(previous.type().id() == cudf::type_id::BOOL8 &&
                predicate.type().id() == cudf::type_id::BOOL8 &&
                previous.size() == predicate.size(),
              "compose_filter_masks: inputs must be equally sized BOOL8 columns",
              std::invalid_argument);
  auto output            = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                          predicate.size(),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto const* old_values = previous.data<bool>();
  auto const* values     = predicate.data<bool>();
  // BOOL8 columns have no children, so these views need no metadata allocation.
  cudf::column_device_view const previous_device{previous, nullptr, nullptr};
  cudf::column_device_view const predicate_device{predicate, nullptr, nullptr};
  auto const old_validity = cudf::detail::make_validity_iterator<true>(previous_device);
  auto const validity     = cudf::detail::make_validity_iterator<true>(predicate_device);
  auto begin              = thrust::make_counting_iterator<cudf::size_type>(0);
  thrust::transform(rmm::exec_policy_nosync(stream, mr),
                    begin,
                    begin + predicate.size(),
                    output->mutable_view().data<bool>(),
                    [=] __device__(cudf::size_type row) {
                      return old_validity[row] && validity[row] && old_values[row] && values[row];
                    });
  return output;
}
}  // namespace gqe::qep::detail
