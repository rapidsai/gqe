/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <memory>
#include <unordered_set>

namespace gqe {
class groupby_simple_aggregations_collector final
  : public cudf::detail::simple_aggregations_collector {
 public:
  using cudf::detail::simple_aggregations_collector::visit;

  std::vector<std::unique_ptr<cudf::aggregation>> visit(
    cudf::data_type col_type, cudf::detail::min_aggregation const&) override
  {
    std::vector<std::unique_ptr<cudf::aggregation>> aggs;
    aggs.push_back(col_type.id() == cudf::type_id::STRING ? cudf::make_argmin_aggregation()
                                                          : cudf::make_min_aggregation());
    return aggs;
  }

  std::vector<std::unique_ptr<cudf::aggregation>> visit(
    cudf::data_type col_type, cudf::detail::max_aggregation const&) override
  {
    std::vector<std::unique_ptr<cudf::aggregation>> aggs;
    aggs.push_back(col_type.id() == cudf::type_id::STRING ? cudf::make_argmax_aggregation()
                                                          : cudf::make_max_aggregation());
    return aggs;
  }
};

std::tuple<cudf::table_view,
           std::vector<cudf::aggregation::Kind>,
           std::vector<std::unique_ptr<cudf::aggregation>>>
flatten_single_pass_aggs(cudf::host_span<cudf::groupby::aggregation_request const> requests)
{
  std::vector<cudf::column_view> columns;
  std::vector<std::unique_ptr<cudf::aggregation>> aggs;
  std::vector<cudf::aggregation::Kind> agg_kinds;

  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;

    std::unordered_set<cudf::aggregation::Kind> agg_kinds_set;
    auto insert_agg = [&](cudf::column_view const& request_values,
                          std::unique_ptr<cudf::aggregation>&& agg) {
      if (agg_kinds_set.insert(agg->kind).second) {
        agg_kinds.push_back(agg->kind);
        aggs.push_back(std::move(agg));
        columns.push_back(request_values);
      }
    };

    auto values_type = cudf::is_dictionary(request.values.type())
                         ? cudf::dictionary_column_view(request.values).keys().type()
                         : request.values.type();
    for (auto&& agg : agg_v) {
      groupby_simple_aggregations_collector collector;

      for (auto& agg_s : agg->get_simple_aggregations(values_type, collector)) {
        insert_agg(request.values, std::move(agg_s));
      }
    }
  }

  return std::make_tuple(cudf::table_view(columns), std::move(agg_kinds), std::move(aggs));
}

template <typename SetType>
class hash_compound_agg_finalizer final : public cudf::detail::aggregation_finalizer {
  cudf::column_view col;
  cudf::data_type result_type;
  cudf::detail::result_cache* sparse_results;
  cudf::detail::result_cache* dense_results;
  cudf::device_span<cudf::size_type const> gather_map;
  SetType set;
  cudf::bitmask_type const* __restrict__ row_bitmask;
  rmm::cuda_stream_view stream;
  rmm::mr::device_memory_resource* mr;

 public:
  using cudf::detail::aggregation_finalizer::visit;

  hash_compound_agg_finalizer(cudf::column_view col,
                              cudf::detail::result_cache* sparse_results,
                              cudf::detail::result_cache* dense_results,
                              cudf::device_span<cudf::size_type const> gather_map,
                              SetType set,
                              cudf::bitmask_type const* row_bitmask,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
    : col(col),
      sparse_results(sparse_results),
      dense_results(dense_results),
      gather_map(gather_map),
      set(set),
      row_bitmask(row_bitmask),
      stream(stream),
      mr(mr)
  {
    result_type = cudf::is_dictionary(col.type()) ? cudf::dictionary_column_view(col).keys().type()
                                                  : col.type();
  }

  auto to_dense_agg_result(cudf::aggregation const& agg)
  {
    auto s                  = sparse_results->get_result(col, agg);
    auto dense_result_table = cudf::detail::gather(cudf::table_view({std::move(s)}),
                                                   gather_map,
                                                   cudf::out_of_bounds_policy::DONT_CHECK,
                                                   cudf::detail::negative_index_policy::NOT_ALLOWED,
                                                   stream,
                                                   mr);
    return std::move(dense_result_table->release()[0]);
  }

  // Enables conversion of ARGMIN/ARGMAX into MIN/MAX
  auto gather_argminmax(cudf::aggregation const& agg)
  {
    auto arg_result = to_dense_agg_result(agg);
    // We make a view of ARG(MIN/MAX) result without a null mask and gather
    // using this map. The values in data buffer of ARG(MIN/MAX) result
    // corresponding to null values was initialized to ARG(MIN/MAX)_SENTINEL
    // which is an out of bounds index value (-1) and causes the gathered
    // value to be null.
    cudf::column_view null_removed_map(
      cudf::data_type(cudf::type_to_id<cudf::size_type>()),
      arg_result->size(),
      static_cast<void const*>(arg_result->view().template data<cudf::size_type>()),
      nullptr,
      0);
    auto gather_argminmax =
      cudf::detail::gather(cudf::table_view({col}),
                           null_removed_map,
                           arg_result->nullable() ? cudf::out_of_bounds_policy::NULLIFY
                                                  : cudf::out_of_bounds_policy::DONT_CHECK,
                           cudf::detail::negative_index_policy::NOT_ALLOWED,
                           stream,
                           mr);
    return std::move(gather_argminmax->release()[0]);
  }

  // Declare overloads for each kind of aggregation to dispatch
  void visit(cudf::aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    dense_results->add_result(col, agg, to_dense_agg_result(agg));
  }

  void visit(cudf::detail::min_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;
    if (result_type.id() == cudf::type_id::STRING) {
      auto transformed_agg = cudf::make_argmin_aggregation();
      dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
    } else {
      dense_results->add_result(col, agg, to_dense_agg_result(agg));
    }
  }

  void visit(cudf::detail::max_aggregation const& agg) override
  {
    if (dense_results->has_result(col, agg)) return;

    if (result_type.id() == cudf::type_id::STRING) {
      auto transformed_agg = cudf::make_argmax_aggregation();
      dense_results->add_result(col, agg, gather_argminmax(*transformed_agg));
    } else {
      dense_results->add_result(col, agg, to_dense_agg_result(agg));
    }
  }
};

/**
 * @brief Gather sparse results into dense using `gather_map` and add to
 * `dense_cache`
 *
 * @see groupby_null_templated()
 */
template <typename SetType>
void sparse_to_dense_results(cudf::table_view const& keys,
                             cudf::host_span<cudf::groupby::aggregation_request const> requests,
                             cudf::detail::result_cache* sparse_results,
                             cudf::detail::result_cache* dense_results,
                             cudf::device_span<cudf::size_type const> gather_map,
                             SetType set,
                             bool keys_have_nulls,
                             cudf::null_policy include_null_keys,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  auto row_bitmask =
    cudf::detail::bitmask_and(keys, stream, rmm::mr::get_current_device_resource()).first;
  bool skip_key_rows_with_nulls =
    keys_have_nulls and include_null_keys == cudf::null_policy::EXCLUDE;
  cudf::bitmask_type const* row_bitmask_ptr =
    skip_key_rows_with_nulls ? static_cast<cudf::bitmask_type*>(row_bitmask.data()) : nullptr;

  for (auto const& request : requests) {
    auto const& agg_v = request.aggregations;
    auto const& col   = request.values;

    // Given an aggregation, this will get the result from sparse_results and
    // convert and return dense, compacted result
    auto finalizer = hash_compound_agg_finalizer(
      col, sparse_results, dense_results, gather_map, set, row_bitmask_ptr, stream, mr);
    for (auto&& agg : agg_v) {
      agg->finalize(finalizer);
    }
  }
}

template <typename RequestType>
inline std::vector<cudf::groupby::aggregation_result> extract_results(
  cudf::host_span<RequestType const> requests,
  cudf::detail::result_cache& cache,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  std::vector<cudf::groupby::aggregation_result> results(requests.size());
  std::unordered_map<std::pair<cudf::column_view, std::reference_wrapper<cudf::aggregation const>>,
                     cudf::column_view,
                     cudf::detail::pair_column_aggregation_hash,
                     cudf::detail::pair_column_aggregation_equal_to>
    repeated_result;
  for (size_t i = 0; i < requests.size(); i++) {
    for (auto&& agg : requests[i].aggregations) {
      if (cache.has_result(requests[i].values, *agg)) {
        results[i].results.emplace_back(cache.release_result(requests[i].values, *agg));
        repeated_result[{requests[i].values, *agg}] = results[i].results.back()->view();
      } else {
        auto it = repeated_result.find({requests[i].values, *agg});
        if (it != repeated_result.end()) {
          results[i].results.emplace_back(std::make_unique<cudf::column>(it->second, stream, mr));
        } else {
          CUDF_FAIL("Cannot extract result from the cache");
        }
      }
    }
  }
  return results;
}

}  // namespace gqe
