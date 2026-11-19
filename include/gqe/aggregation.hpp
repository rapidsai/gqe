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

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>

#include <memory>

namespace gqe {

namespace detail {

/**
 * @brief Return the cuDF result type for an aggregation over `source_type`.
 *
 * Fixed-point results retain the source scale when the aggregation's target type is fixed-point.
 *
 * @param[in] source_type Input column type.
 * @param[in] kind Aggregation kind.
 * @return Aggregation result type.
 * @throws cudf::logic_error If the type and aggregation combination is unsupported.
 */
[[nodiscard]] cudf::data_type compute_target_type(cudf::data_type source_type,
                                                  cudf::aggregation::Kind kind);

/**
 * @brief Return the widened result type for a SUM over a fixed-point column.
 *
 * Follows the engine's Spark-style decimal widening, `decimal(p + 10, s)`: every supported
 * decimal precision (9, 18, or 38 digits) plus 10 exceeds 18 digits, so a decimal SUM always
 * targets `DECIMAL128` at the source's scale. This differs from @ref compute_target_type,
 * whose cuDF semantics keep a fixed-point SUM in the source width.
 *
 * @param[in] source_type Fixed-point input column type.
 * @return `DECIMAL128` at the source's scale.
 * @throws std::logic_error If `source_type` is not a fixed-point type.
 */
[[nodiscard]] cudf::data_type decimal_sum_target_type(cudf::data_type source_type);

}  // namespace detail

/**
 * @brief Construct a cuDF group-by aggregation used by GQE execution paths.
 *
 * Supported kinds are `SUM`, `PRODUCT`, `MIN`, `MAX`, `COUNT_VALID`, `COUNT_ALL`,
 * `SUM_OF_SQUARES`, `MEAN`, `VARIANCE`, `STD`, and `MEDIAN`.
 *
 * @param[in] kind Aggregation kind.
 * @return Owning cuDF group-by aggregation.
 * @throws std::invalid_argument If `kind` is not supported.
 */
[[nodiscard]] std::unique_ptr<cudf::groupby_aggregation> make_groupby_aggregation(
  cudf::aggregation::Kind kind);

/**
 * @brief Return the aggregation kind that combines apply-concat-apply partials.
 *
 * The distributive combine mapping is:
 *
 * - `SUM`, `PRODUCT`, `MIN`, `MAX`, `ANY`, and `ALL` to themselves.
 * - `COUNT_VALID`, `COUNT_ALL`, and `SUM_OF_SQUARES` to `SUM`.
 *
 * @param[in] first_aggregation_kind First-pass aggregation kind.
 * @return Second-pass aggregation kind.
 * @throws std::logic_error If the aggregation has no established distributive combine mapping.
 */
[[nodiscard]] cudf::aggregation::Kind get_second_aggregation_kind(
  cudf::aggregation::Kind first_aggregation_kind);

}  // namespace gqe
