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

#include <gqe/qep/shapes/keyed_accumulator.hpp>

#include <gqe/utility/error.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace gqe {
namespace qep {

namespace {

/**
 * @brief Validate the pre-allocated slot storage against the bound.
 *
 * Runs in the slot-storage member's initializer, so validation precedes the auxiliary-buffer
 * allocations. A malformed geometry is a planner or engine error, so it reports
 * `std::logic_error`.
 */
[[nodiscard]] std::unique_ptr<keyed_accumulator_storage> validated_storage(
  std::unique_ptr<keyed_accumulator_storage> slots, int64_t group_count_bound)
{
  GQE_EXPECTS(group_count_bound > 0, "keyed_accumulator_state: group_count_bound must be positive");
  GQE_EXPECTS(slots != nullptr, "keyed_accumulator_state: slot storage must not be null");
  GQE_EXPECTS(slots->capacity() >= group_count_bound,
              "keyed_accumulator_state: capacity must be at least group_count_bound");
  return slots;
}

/// Validate the control-region size before any buffer allocation.
[[nodiscard]] std::size_t validated_control_bytes(std::size_t control_bytes)
{
  GQE_EXPECTS(control_bytes > 0, "keyed_accumulator_state: control size must be positive");
  return control_bytes;
}

}  // namespace

keyed_accumulator_state::keyed_accumulator_state(int64_t group_count_bound,
                                                 std::unique_ptr<keyed_accumulator_storage> slots,
                                                 std::size_t value_validity_bytes,
                                                 std::size_t wrap_counter_bytes,
                                                 std::size_t control_bytes,
                                                 double load_factor,
                                                 cudf::data_type key_type,
                                                 cudf::data_type value_type,
                                                 cudf::data_type ordering_type,
                                                 cudf::aggregation::Kind aggregation_kind,
                                                 query_stream const& stream_handle,
                                                 rmm::device_async_resource_ref mr)
  : _group_count_bound{group_count_bound},
    _load_factor{load_factor},
    _key_type{key_type},
    _value_type{value_type},
    _ordering_type{ordering_type},
    _aggregation_kind{aggregation_kind},
    _slots{validated_storage(std::move(slots), group_count_bound)},
    _value_validity{value_validity_bytes, stream_handle.ref(), mr},
    _wrap_counters{wrap_counter_bytes, stream_handle.ref(), mr},
    _control{validated_control_bytes(control_bytes), stream_handle.ref(), mr}
{
}

}  // namespace qep
}  // namespace gqe
