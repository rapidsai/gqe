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

#include <gqe/aggregation.hpp>

#include <gqe/utility/error.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace gqe {

namespace {

/** @brief Resolve an aggregation's dispatched target type while preserving fixed-point scale. */
struct target_type_dispatcher {
  cudf::data_type source_type;

  /**
   * @brief Return the result type for one dispatched source-type and aggregation-kind pair.
   *
   * @tparam Source C++ storage type of the aggregation input.
   * @tparam k Aggregation kind applied to the input.
   * @return Aggregation result type.
   */
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

}  // namespace

namespace detail {

cudf::data_type compute_target_type(cudf::data_type source_type, cudf::aggregation::Kind kind)
{
  return cudf::detail::dispatch_type_and_aggregation(
    source_type, kind, target_type_dispatcher{source_type});
}

cudf::data_type decimal_sum_target_type(cudf::data_type source_type)
{
  GQE_EXPECTS(cudf::is_fixed_point(source_type),
              "decimal_sum_target_type: the source type must be fixed-point");
  return cudf::data_type{cudf::type_id::DECIMAL128, source_type.scale()};
}

}  // namespace detail

std::unique_ptr<cudf::groupby_aggregation> make_groupby_aggregation(
  cudf::aggregation::Kind aggregation_kind)
{
  switch (aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::make_sum_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::PRODUCT:
      return cudf::make_product_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MIN: return cudf::make_min_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MAX: return cudf::make_max_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::COUNT_VALID:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::COUNT_ALL:
      return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::SUM_OF_SQUARES:
      return cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MEAN: return cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::VARIANCE:
      return cudf::make_variance_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::STD: return cudf::make_std_aggregation<cudf::groupby_aggregation>();
    case cudf::aggregation::MEDIAN:
      return cudf::make_median_aggregation<cudf::groupby_aggregation>();
    default:
      throw std::invalid_argument("Unknown groupby aggregation: " +
                                  std::to_string(aggregation_kind));
  }
}

cudf::aggregation::Kind get_second_aggregation_kind(cudf::aggregation::Kind first_aggregation_kind)
{
  switch (first_aggregation_kind) {
    case cudf::aggregation::SUM: return cudf::aggregation::SUM;
    case cudf::aggregation::PRODUCT: return cudf::aggregation::PRODUCT;
    case cudf::aggregation::MIN: return cudf::aggregation::MIN;
    case cudf::aggregation::MAX: return cudf::aggregation::MAX;
    case cudf::aggregation::ANY: return cudf::aggregation::ANY;
    case cudf::aggregation::ALL: return cudf::aggregation::ALL;
    case cudf::aggregation::SUM_OF_SQUARES: return cudf::aggregation::SUM;
    case cudf::aggregation::COUNT_VALID: return cudf::aggregation::SUM;
    case cudf::aggregation::COUNT_ALL: return cudf::aggregation::SUM;
    default:
      throw std::logic_error("Unknown aggregation type in get_second_aggregation_kind: " +
                             std::to_string(first_aggregation_kind));
  }
}

}  // namespace gqe
