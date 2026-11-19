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

#include "launchers.hpp"

#include <cudf/utilities/bit.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

void launch_flag_selected_nulls(cudf::column_view column,
                                uint8_t const* row_valid,
                                control_block* control,
                                cuda::stream_ref stream)
{
  auto const* null_mask = column.null_mask();
  auto const offset     = column.offset();
  // Every flagging thread stores the same value, so the plain store needs no atomic.
  thrust::for_each_n(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<cudf::size_type>{0},
                     column.size(),
                     [=] __device__(cudf::size_type row) {
                       bool const selected = row_valid == nullptr || row_valid[row] != uint8_t{0};
                       if (selected && !cudf::bit_is_set(null_mask, offset + row)) {
                         control->not_null_violation = 1;
                       }
                     });
}

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
