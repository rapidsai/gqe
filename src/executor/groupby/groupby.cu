/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/executor/groupby.hpp>

#include "aggregation_target_type.hpp"
#include "hash_groupby.cuh"

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/traits.cuh>

#include <algorithm>
#include <memory>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gqe {
namespace groupby {

namespace {
/// Make an empty table with appropriate types for requested aggs
template <typename RequestType>
auto empty_results(cudf::host_span<RequestType const> requests)
{
  std::vector<cudf::groupby::aggregation_result> empty_results;

  std::transform(
    requests.begin(), requests.end(), std::back_inserter(empty_results), [](auto const& request) {
      std::vector<std::unique_ptr<cudf::column>> results;

      std::transform(request.aggregations.begin(),
                     request.aggregations.end(),
                     std::back_inserter(results),
                     [&request](auto const& agg) {
                       return cudf::make_empty_column(
                         gqe::detail::compute_target_type(request.values.type(), agg->kind));
                     });

      return cudf::groupby::aggregation_result{std::move(results)};
    });

  return empty_results;
}

/// Verifies the agg requested on the request's values is valid
template <typename RequestType>
void verify_valid_requests(cudf::host_span<RequestType const> requests)
{
  CUDF_EXPECTS(
    std::all_of(
      requests.begin(),
      requests.end(),
      [](auto const& request) {
        return std::all_of(
          request.aggregations.begin(), request.aggregations.end(), [&request](auto const& agg) {
            auto values_type = cudf::is_dictionary(request.values.type())
                                 ? cudf::dictionary_column_view(request.values).keys().type()
                                 : request.values.type();
            return cudf::is_valid_aggregation(values_type, agg->kind);
          });
      }),
    "Invalid type/aggregation combination.");
}

const auto hash_aggregations = std::unordered_set{// Single-pass
                                                  cudf::aggregation::SUM,
                                                  cudf::aggregation::SUM_WITH_OVERFLOW,
                                                  cudf::aggregation::SUM_OF_SQUARES,
                                                  cudf::aggregation::PRODUCT,
                                                  cudf::aggregation::MIN,
                                                  cudf::aggregation::MAX,
                                                  cudf::aggregation::COUNT_VALID,
                                                  cudf::aggregation::COUNT_ALL,
                                                  // Compound
                                                  cudf::aggregation::ARGMIN,
                                                  cudf::aggregation::ARGMAX,
                                                  cudf::aggregation::MEAN,
                                                  cudf::aggregation::M2,
                                                  cudf::aggregation::STD,
                                                  cudf::aggregation::VARIANCE};

bool is_hash_aggregation(cudf::aggregation::Kind kind) { return hash_aggregations.contains(kind); }

struct simple_aggregation_collector {
  template <cudf::aggregation::Kind k>
  std::vector<std::unique_ptr<cudf::aggregation>> operator()(cudf::data_type col_type,
                                                             cudf::aggregation const& agg) const
  {
    std::vector<std::unique_ptr<cudf::aggregation>> aggs;
    if constexpr (k == cudf::aggregation::MIN) {
      aggs.push_back(col_type.id() == cudf::type_id::STRING ? cudf::make_argmin_aggregation()
                                                            : cudf::make_min_aggregation());
    } else if constexpr (k == cudf::aggregation::MAX) {
      aggs.push_back(col_type.id() == cudf::type_id::STRING ? cudf::make_argmax_aggregation()
                                                            : cudf::make_max_aggregation());
    } else if constexpr (k == cudf::aggregation::MEAN) {
      CUDF_EXPECTS(cudf::is_fixed_width(col_type), "MEAN aggregation expects fixed width type");
      aggs.push_back(cudf::make_sum_aggregation());
      aggs.push_back(cudf::make_count_aggregation());
    } else if constexpr (k == cudf::aggregation::M2 || k == cudf::aggregation::VARIANCE ||
                         k == cudf::aggregation::STD) {
      aggs.push_back(cudf::make_sum_of_squares_aggregation());
      aggs.push_back(cudf::make_sum_aggregation());
      aggs.push_back(cudf::make_count_aggregation());
    } else {
      aggs.push_back(agg.clone());
    }
    return aggs;
  }
};

std::vector<cudf::aggregation::Kind> get_simple_aggregations(cudf::groupby_aggregation const& agg,
                                                             cudf::data_type values_type)
{
  auto aggs = cudf::detail::aggregation_dispatcher(
    agg.kind, simple_aggregation_collector{}, values_type, agg);
  std::vector<cudf::aggregation::Kind> kinds;
  std::transform(
    aggs.begin(), aggs.end(), std::back_inserter(kinds), [](auto const& a) { return a->kind; });
  return kinds;
}

struct can_use_hash_groupby_fn {
  template <typename T, cudf::aggregation::Kind K>
    requires(cudf::is_nested<T>())
  bool operator()() const
  {
    return false;
  }

  template <cudf::aggregation::Kind k>
  constexpr static bool uses_underlying_type()
  {
    return k == cudf::aggregation::MIN || k == cudf::aggregation::MAX ||
           k == cudf::aggregation::SUM;
  }

  template <typename T, cudf::aggregation::Kind K>
    requires(cudf::is_fixed_point<T>())
  bool operator()() const
  {
    if constexpr (std::is_same_v<T, ::numeric::decimal128> && K == cudf::aggregation::SUM) {
      return true;
    }
    using target_type        = cudf::detail::target_type_t<T, K>;
    using device_target_type = std::conditional_t<uses_underlying_type<K>(),
                                                  cudf::device_storage_type_t<target_type>,
                                                  target_type>;
    if constexpr (!std::is_void_v<device_target_type>) {
      return cudf::has_atomic_support<device_target_type>();
    }
    return false;
  }

  template <typename T, cudf::aggregation::Kind K>
    requires(!cudf::is_nested<T>() && !cudf::is_fixed_point<T>())
  bool operator()() const
  {
    using target_type = cudf::detail::target_type_t<T, K>;
    if constexpr (!std::is_void_v<target_type>) { return cudf::has_atomic_support<target_type>(); }
    return false;
  }
};

bool supports_gqe_hash_groupby(cudf::host_span<cudf::groupby::aggregation_request const> requests)
{
  return std::all_of(requests.begin(), requests.end(), [](auto const& request) {
    auto const values_type = cudf::is_dictionary(request.values.type())
                               ? cudf::dictionary_column_view(request.values).keys().type()
                               : request.values.type();
    return std::all_of(
      request.aggregations.begin(), request.aggregations.end(), [&](auto const& agg) {
        if (!is_hash_aggregation(agg->kind)) { return false; }
        auto const simple_aggs = get_simple_aggregations(*agg, values_type);
        return std::all_of(simple_aggs.begin(), simple_aggs.end(), [values_type](auto kind) {
          return cudf::detail::dispatch_type_and_aggregation(
            values_type, kind, can_use_hash_groupby_fn{});
        });
      });
  });
}

}  // namespace

groupby::groupby(cudf::table_view const& keys) : _keys{keys} {};

// Compute aggregation requests
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>>
groupby::aggregate(cudf::host_span<cudf::groupby::aggregation_request const> requests,
                   cudf::column_view const& active_mask,
                   rmm::device_async_resource_ref mr)
{
  return aggregate(requests, active_mask, cudf::get_default_stream(), mr);
}

// Compute aggregation requests
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>>
groupby::aggregate(cudf::host_span<cudf::groupby::aggregation_request const> requests,
                   cudf::column_view const& active_mask,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
{
  utility::nvtx_scoped_range groupby_range("groupby");
  CUDF_EXPECTS(
    std::all_of(requests.begin(),
                requests.end(),
                [this](auto const& request) { return request.values.size() == _keys.num_rows(); }),
    "Size mismatch between request values and groupby keys.");

  verify_valid_requests(requests);

  if (!supports_gqe_hash_groupby(requests)) {
    cudf::groupby::groupby cudf_groupby_obj(_keys, cudf::null_policy::INCLUDE);
    return cudf_groupby_obj.aggregate(requests);
  }

  if (_keys.num_rows() == 0) { return {empty_like(_keys), empty_results(requests)}; }

  return gqe::groupby::hash::groupby(_keys, requests, active_mask, stream, mr);
}

}  // namespace groupby
}  // namespace gqe
