/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hash_groupby.cuh"

#include <gqe/device_properties.hpp>
#include <gqe/executor/groupby.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/logger.hpp>

#include <memory>
#include <utility>

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
                         cudf::detail::target_type(request.values.type(), agg->kind));
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
            return cudf::detail::is_valid_aggregation(values_type, agg->kind);
          });
      }),
    "Invalid type/aggregation combination.");
}

}  // namespace

groupby::groupby(cudf::table_view const& keys) : _keys{keys} {};

// Compute aggregation requests
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>>
groupby::aggregate(cudf::host_span<cudf::groupby::aggregation_request const> requests,
                   cudf::column_view const& active_mask,
                   gqe::device_properties const& device_properties,
                   rmm::mr::device_memory_resource* mr)
{
  return aggregate(requests, active_mask, device_properties, cudf::get_default_stream(), mr);
}

// Compute aggregation requests
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>>
groupby::aggregate(cudf::host_span<cudf::groupby::aggregation_request const> requests,
                   cudf::column_view const& active_mask,
                   gqe::device_properties const& device_properties,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr)
{
  utility::nvtx_scoped_range groupby_range("groupby");
  CUDF_EXPECTS(
    std::all_of(requests.begin(),
                requests.end(),
                [this](auto const& request) { return request.values.size() == _keys.num_rows(); }),
    "Size mismatch between request values and groupby keys.");

  verify_valid_requests(requests);

  if (!cudf::groupby::detail::hash::can_use_hash_groupby(requests)) {
    cudf::groupby::groupby cudf_groupby_obj(_keys, cudf::null_policy::INCLUDE);
    return cudf_groupby_obj.aggregate(requests);
  }

  if (_keys.num_rows() == 0) { return {empty_like(_keys), empty_results(requests)}; }

  return gqe::groupby::hash::groupby(_keys, requests, active_mask, device_properties, stream, mr);
}

}  // namespace groupby
}  // namespace gqe
