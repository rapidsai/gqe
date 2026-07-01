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

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <type_traits>

namespace gqe::detail {

struct target_type_dispatcher {
  cudf::data_type source_type;

  template <typename Source, cudf::aggregation::Kind k>
  cudf::data_type operator()() const
  {
    using target_t = cudf::detail::target_type_t<Source, k>;
    if constexpr (std::is_void_v<target_t>) {
      CUDF_FAIL("Invalid type/aggregation combination.");
    } else {
      auto const id = cudf::type_to_id<target_t>();
      return cudf::is_fixed_point<target_t>() ? cudf::data_type{id, source_type.scale()}
                                              : cudf::data_type{id};
    }
  }
};

[[nodiscard]] inline cudf::data_type compute_target_type(cudf::data_type source_type,
                                                         cudf::aggregation::Kind kind)
{
  return cudf::detail::dispatch_type_and_aggregation(
    source_type, kind, target_type_dispatcher{source_type});
}

}  // namespace gqe::detail
