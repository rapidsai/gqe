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

#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/utility/error.hpp>

#include <cudf/utilities/traits.hpp>

#include <span>
#include <stdexcept>

namespace gqe::qep::detail {

// ----- Validation and logical schema mapping -----

/** @brief Return whether a selectively materialized operand is in the supported type domain. */
[[nodiscard]] inline bool is_supported_semi_anti_column(cudf::data_type type)
{
  return cudf::is_fixed_width(type) || type.id() == cudf::type_id::STRING;
}

/** @brief Reject out-of-range or unsupported consumed columns before materialization. */
inline void validate_semi_anti_columns(logical_table_view logical,
                                       std::span<cudf::size_type const> indices)
{
  for (auto index : indices) {
    GQE_EXPECTS(index >= 0 && index < logical.num_columns(),
                "semi/anti column out of range",
                std::invalid_argument);
    GQE_EXPECTS(is_supported_semi_anti_column(logical.column_type(index)),
                "semi/anti columns must be fixed-width or STRING",
                std::invalid_argument);
  }
}

}  // namespace gqe::qep::detail
