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

#include <gqe/context_reference.hpp>
#include <gqe/device_properties.hpp>
#include <gqe/qep/operator/bounded_hash_aggregate.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/state.hpp>
#include <gqe/types.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

using bool_column_wrapper   = cudf::test::fixed_width_column_wrapper<bool>;
using int32_column_wrapper  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64_column_wrapper  = cudf::test::fixed_width_column_wrapper<int64_t>;
using double_column_wrapper = cudf::test::fixed_width_column_wrapper<double>;
using float_column_wrapper  = cudf::test::fixed_width_column_wrapper<float>;

constexpr int64_t int64_sentinel = std::numeric_limits<int64_t>::max();

cudf::data_type const int32_type{cudf::type_id::INT32};
cudf::data_type const int64_type{cudf::type_id::INT64};
cudf::data_type const float32_type{cudf::type_id::FLOAT32};
cudf::data_type const float64_type{cudf::type_id::FLOAT64};

[[nodiscard]] bool supports_128_bit_cas()
{
  return gqe::device_properties::instance().get<gqe::device_properties::computeCapabilityMajor>() >=
         9;
}

constexpr auto cas_128_skip_message = "This test requires compute capability 9.0 for a 128-bit CAS";

/** @brief Assemble released column wrappers into a regular-table chunk. */
template <typename... Columns>
[[nodiscard]] qep::state_container make_table(Columns&&... input_columns)
{
  qep::state_container_builder builder;
  (builder.add_state(state_kind::cudf_column{std::forward<Columns>(input_columns)},
                     gqe::query_stream::default_stream()),
   ...);
  return builder.build();
}

/** @brief Prefix a regular-table chunk with a valid-row mask. */
[[nodiscard]] qep::state_container make_masked_table(std::unique_ptr<cudf::column> mask,
                                                     qep::state_container const& data)
{
  qep::state_container_builder builder;
  builder.add_state(state_kind::valid_mask{std::move(mask)}, gqe::query_stream::default_stream());
  for (auto const& state : data) {
    builder.add_state(state);
  }
  return builder.build();
}

/** @brief Drive the build fold to its finalized map state. */
[[nodiscard]] qep::state_container run_build(qep::bounded_hash_aggregate_build_task const& task,
                                             std::span<qep::state_container const> chunks)
{
  gqe::context_reference ctx_ref{};
  auto accumulator = task.initialize(ctx_ref);
  for (auto const& chunk : chunks) {
    task.next(chunk, accumulator, ctx_ref);
  }
  return task.finalize(std::move(accumulator), ctx_ref);
}

/** @brief Drain the scan iterator into its emitted chunks. */
[[nodiscard]] std::vector<qep::state_container> run_scan(qep::iterate_task const& task,
                                                         qep::state_container map)
{
  gqe::context_reference ctx_ref{};
  auto iterator = task.initialize(map, ctx_ref);
  std::vector<qep::state_container> chunks;
  while (auto chunk = task.next(iterator, ctx_ref)) {
    chunks.push_back(std::move(*chunk));
  }
  task.finalize(std::move(iterator), ctx_ref);
  return chunks;
}

/** @brief Concatenate scan chunks into one owned result table. */
[[nodiscard]] std::unique_ptr<cudf::table> concatenate_chunks(
  std::span<qep::state_container const> chunks)
{
  std::vector<cudf::table_view> views;
  views.reserve(chunks.size());
  for (auto const& chunk : chunks) {
    auto const view = qep::to_table_view(chunk);
    if (!view.has_value()) { throw std::logic_error{"scan chunk is not a regular table"}; }
    views.push_back(*view);
  }
  return cudf::concatenate(views);
}

/** @brief Run build then scan and materialize the full result. */
[[nodiscard]] std::unique_ptr<cudf::table> run_aggregate(
  qep::bounded_hash_aggregate_build_task const& build,
  qep::bounded_hash_aggregate_scan_task const& scan,
  std::span<qep::state_container const> chunks)
{
  auto const outputs = run_scan(scan, run_build(build, chunks));
  return concatenate_chunks(outputs);
}

/** @brief `[key, aggregate]` reference result from a cuDF groupby, NULL keys included. */
[[nodiscard]] std::unique_ptr<cudf::table> reference_groupby(
  cudf::column_view keys,
  cudf::column_view values,
  std::unique_ptr<cudf::groupby_aggregation> aggregation)
{
  cudf::groupby::groupby groupby_obj{cudf::table_view{{keys}}, cudf::null_policy::INCLUDE};
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(std::move(aggregation));
  auto [group_keys, results] = groupby_obj.aggregate(requests);

  auto columns = group_keys->release();
  columns.push_back(std::move(results[0].results[0]));
  return std::make_unique<cudf::table>(std::move(columns));
}

/** @brief Order-insensitive table comparison (hash aggregation has no output order). */
void expect_tables_equal_unordered(cudf::table_view actual, cudf::table_view expected)
{
  auto const sorted_actual   = cudf::sort(actual);
  auto const sorted_expected = cudf::sort(expected);
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_actual->view(), sorted_expected->view());
}

/**
 * @brief Run a `[key, value]` aggregate over `chunks` and compare it with a cuDF groupby.
 *
 * The chunks are the single source of truth. The groupby reference consumes their
 * concatenation, so the two inputs cannot drift apart.
 */
void expect_aggregate_matches_groupby(qep::bounded_hash_aggregate_build_task const& build,
                                      std::span<qep::state_container const> chunks,
                                      std::unique_ptr<cudf::groupby_aggregation> aggregation,
                                      cudf::size_type max_rows_per_output = 16)
{
  auto const reference_input = concatenate_chunks(chunks);
  if (reference_input->num_columns() != 2) {
    throw std::logic_error{"the groupby reference helper expects [key, value] chunks"};
  }
  auto const expected = reference_groupby(reference_input->get_column(0).view(),
                                          reference_input->get_column(1).view(),
                                          std::move(aggregation));

  qep::bounded_hash_aggregate_scan_task const scan{max_rows_per_output};
  auto const actual = run_aggregate(build, scan, chunks);
  expect_tables_equal_unordered(actual->view(), expected->view());
}

}  // namespace

// ----- Correctness against a cudf::groupby reference -----

TEST(QepBoundedHashAggregateTaskTest, SumInt64SingleChunk)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{5, 3, 5, 8, 3, 5}.release(),
                              int64_column_wrapper{1, 2, 3, 4, 5, 6}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    3};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>(), 1024);
}

/**
 * @brief Exactly representable values, so any accumulation order gives the exact sum.
 */
TEST(QepBoundedHashAggregateTaskTest, SumFloat64MultiChunk)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 3}.release(),
                              double_column_wrapper{0.5, 1.0, 1.5}.release()));
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 3}.release(),
                              double_column_wrapper{2.0, 2.5, 3.0}.release()));
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 3}.release(),
                              double_column_wrapper{3.5, 4.0, 4.5}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, float64_type},
    3};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>(), 2);
}

TEST(QepBoundedHashAggregateTaskTest, SumFloat32)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int32_column_wrapper{7, 7, 9}.release(),
                              float_column_wrapper{1.25F, 2.5F, 4.0F}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int32_type, float32_type},
    2};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

TEST(QepBoundedHashAggregateTaskTest, SumInt32InputWidensToInt64)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{4, 4, 6}.release(),
                              int32_column_wrapper{10, 20, 30}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    2};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

TEST(QepBoundedHashAggregateTaskTest, CountValidAndCountAllWithNullValues)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_table(int64_column_wrapper{1, 1, 2, 2, 2}.release(),
               int32_column_wrapper{{9, 9, 9, 9, 9}, {true, false, true, false, false}}.release()));

  qep::bounded_hash_aggregate_build_task const count_valid{
    0,
    {cudf::aggregation::COUNT_VALID, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int32_type},
    2};
  expect_aggregate_matches_groupby(
    count_valid,
    chunks,
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));

  qep::bounded_hash_aggregate_build_task const count_all{
    0,
    {cudf::aggregation::COUNT_ALL, std::nullopt, std::nullopt, gqe::nullability::nullable},
    {int64_type, int32_type},
    2};
  expect_aggregate_matches_groupby(
    count_all,
    chunks,
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
}

TEST(QepBoundedHashAggregateTaskTest, MinAndMaxInt32)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int32_column_wrapper{3, 1, 3, 1, 3}.release(),
                              int32_column_wrapper{5, -2, 9, 7, -4}.release()));

  qep::bounded_hash_aggregate_build_task const min_task{
    0,
    {cudf::aggregation::MIN, 1, std::nullopt, gqe::nullability::nullable},
    {int32_type, int32_type},
    2};
  expect_aggregate_matches_groupby(
    min_task, chunks, cudf::make_min_aggregation<cudf::groupby_aggregation>());

  qep::bounded_hash_aggregate_build_task const max_task{
    0,
    {cudf::aggregation::MAX, 1, std::nullopt, gqe::nullability::nullable},
    {int32_type, int32_type},
    2};
  expect_aggregate_matches_groupby(
    max_task, chunks, cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

/**
 * @brief Floating-point MIN/MAX go through the CAS-loop fetch_min/fetch_max. NaN values are valid
 * and must follow cuDF's device min/max selection.
 */
TEST(QepBoundedHashAggregateTaskTest, MinAndMaxFloat64WithNan)
{
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 1, 2, 2, 3}.release(),
                              double_column_wrapper{2.0, -1.0, nan, 5.0, nan}.release()));

  qep::bounded_hash_aggregate_build_task const min_task{
    0,
    {cudf::aggregation::MIN, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, float64_type},
    3};
  expect_aggregate_matches_groupby(
    min_task, chunks, cudf::make_min_aggregation<cudf::groupby_aggregation>());

  qep::bounded_hash_aggregate_build_task const max_task{
    0,
    {cudf::aggregation::MAX, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, float64_type},
    3};
  expect_aggregate_matches_groupby(
    max_task, chunks, cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

/**
 * @brief Integral PRODUCT widens into the INT64 accumulator through the CAS-loop multiply; NULL
 * values are skipped and an all-NULL group aggregates to NULL.
 */
TEST(QepBoundedHashAggregateTaskTest, ProductInt32WidensToInt64)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int32_column_wrapper{1, 1, 2, 2, 3, 3}.release(),
    int32_column_wrapper{{2, 3, 5, 7, 9, 9}, {true, true, true, true, false, false}}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::PRODUCT, 1, std::nullopt, gqe::nullability::nullable},
    {int32_type, int64_type},
    3};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

/**
 * @brief Power-of-two factors, so any accumulation order gives the exact product.
 */
TEST(QepBoundedHashAggregateTaskTest, ProductFloat64)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 1, 2, 1}.release(),
                              double_column_wrapper{0.5, 2.0, 4.0, 8.0, 2.0}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::PRODUCT, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, float64_type},
    2};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

/**
 * @brief ARGMIN/ARGMAX emit the payload of each group's winning ordering row (MIN_BY/MAX_BY).
 * NULL ordering values are skipped. Verified against a cudf::groupby ARGMIN/ARGMAX reference
 * gathered through the payload column.
 */
TEST(QepBoundedHashAggregateTaskTest, MinByAndMaxByInt64Ordering)
{
  if (!supports_128_bit_cas()) { GTEST_SKIP() << cas_128_skip_message; }

  std::vector<int64_t> const key_values{1, 1, 2, 2, 3, 3, 4};
  std::vector<int64_t> const ordering{50, 30, 70, 90, 7, 7, 20};
  std::vector<bool> const ordering_valid{true, true, true, true, false, true, true};
  std::vector<double> const payload{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5};

  auto keys = int64_column_wrapper(key_values.begin(), key_values.end());
  auto ordering_column =
    int64_column_wrapper(ordering.begin(), ordering.end(), ordering_valid.begin());
  auto payload_column = double_column_wrapper(payload.begin(), payload.end());

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper(key_values.begin(), key_values.end()).release(),
    int64_column_wrapper(ordering.begin(), ordering.end(), ordering_valid.begin()).release(),
    double_column_wrapper(payload.begin(), payload.end()).release()));

  auto const reference = [&](std::unique_ptr<cudf::groupby_aggregation>&& aggregation) {
    // cudf's ARGMIN/ARGMAX return row indices; gathering the payload column through them
    // yields the MIN_BY/MAX_BY reference.
    cudf::groupby::groupby groupby_obj{cudf::table_view{{keys}}, cudf::null_policy::INCLUDE};
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = ordering_column;
    requests[0].aggregations.push_back(std::move(aggregation));
    auto [group_keys, results] = groupby_obj.aggregate(requests);
    // An all-NULL group's ARGMIN index is NULL; replace it with a past-the-end sentinel (a
    // negative one would wrap around in cudf::gather), so the NULLIFY gather emits NULL.
    cudf::numeric_scalar<cudf::size_type> const out_of_bounds{
      static_cast<cudf::size_type>(key_values.size())};
    auto indices  = cudf::replace_nulls(results[0].results[0]->view(), out_of_bounds);
    auto gathered = cudf::gather(
      cudf::table_view{{payload_column}}, indices->view(), cudf::out_of_bounds_policy::NULLIFY);
    auto payload_result = std::move(gathered->release()[0]);
    // The NULLIFY gather always attaches a mask; drop it when empty so the strict property
    // comparison matches the operator's non-nullable output.
    if (payload_result->null_count() == 0) {
      payload_result->set_null_mask(rmm::device_buffer{}, 0);
    }
    auto columns = group_keys->release();
    columns.push_back(std::move(payload_result));
    return std::make_unique<cudf::table>(std::move(columns));
  };

  qep::bounded_hash_aggregate_scan_task const scan{16};

  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::nullable},
    {int64_type, float64_type, int64_type},
    4};
  auto const min_result   = run_aggregate(min_by, scan, chunks);
  auto const min_expected = reference(cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(min_result->view(), min_expected->view());

  qep::bounded_hash_aggregate_build_task const max_by{
    0,
    {cudf::aggregation::ARGMAX, 1, 2, gqe::nullability::nullable},
    {int64_type, float64_type, int64_type},
    4};
  auto const max_result   = run_aggregate(max_by, scan, chunks);
  auto const max_expected = reference(cudf::make_argmax_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(max_result->view(), max_expected->view());
}

/**
 * @brief MIN_BY over a FLOAT64 ordering follows the NaN-aware order: NaN never wins against a
 * real ordering value, and a group whose ordering values are all NULL emits NULL. Expected
 * payloads are fixed by construction (one deterministic winner per group).
 */
TEST(QepBoundedHashAggregateTaskTest, MinByFloat64OrderingWithNanAndNulls)
{
  if (!supports_128_bit_cas()) { GTEST_SKIP() << cas_128_skip_message; }

  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper{1, 1, 2, 2, 3, 3}.release(),
    double_column_wrapper{{nan, 3.0, 5.0, nan, 0.0, 0.0}, {true, true, true, true, false, false}}
      .release(),
    int64_column_wrapper{111, 222, 333, 444, 555, 666}.release()));

  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::nullable},
    {int64_type, int64_type, float64_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(min_by, scan, chunks);

  auto expected_keys    = int64_column_wrapper{1, 2, 3};
  auto expected_payload = int64_column_wrapper{{222, 333, 0}, {true, true, false}};
  expect_tables_equal_unordered(result->view(),
                                cudf::table_view{{expected_keys, expected_payload}});
}

/**
 * @brief A NULL payload excludes its row from the aggregation, like a NULL ordering value: the
 * winner comes from the payload-valid rows, and a group whose payloads are all NULL emits NULL.
 */
TEST(QepBoundedHashAggregateTaskTest, MinByNullPayloadRowsAreSkipped)
{
  if (!supports_128_bit_cas()) { GTEST_SKIP() << cas_128_skip_message; }

  std::vector<int64_t> const keys{1, 1, 1, 2, 2};
  std::vector<int64_t> const ordering{3, 5, 7, 9, 11};
  std::vector<int64_t> const payload{30, 50, 70, 90, 110};
  // Group 1's smallest ordering value (3) has a NULL payload, so the runner-up (5) wins. Group
  // 2's payloads are all NULL.
  std::vector<bool> const payload_valid{false, true, true, false, false};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper(keys.begin(), keys.end()).release(),
    int64_column_wrapper(ordering.begin(), ordering.end()).release(),
    int64_column_wrapper(payload.begin(), payload.end(), payload_valid.begin()).release()));

  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::nullable},
    {int64_type, int64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(min_by, scan, chunks);

  auto expected_keys    = int64_column_wrapper{1, 2};
  auto expected_payload = int64_column_wrapper{{50, 0}, {true, false}};
  expect_tables_equal_unordered(result->view(),
                                cudf::table_view{{expected_keys, expected_payload}});
}

/**
 * @brief The payload NULL check composes with the valid-row mask: a masked-out row does not
 * participate even when its payload is valid, and a NULL-payload row does not participate even
 * when the mask keeps it.
 */
TEST(QepBoundedHashAggregateTaskTest, MinByNullPayloadComposesWithTheMask)
{
  if (!supports_128_bit_cas()) { GTEST_SKIP() << cas_128_skip_message; }

  std::vector<int64_t> const keys{1, 1, 1, 1};
  std::vector<int64_t> const ordering{3, 5, 7, 9};
  std::vector<int64_t> const payload{30, 50, 70, 90};
  // Row 0 is masked out, row 1 has a NULL payload, so row 2 wins.
  std::vector<bool> const mask{false, true, true, true};
  std::vector<bool> const payload_valid{true, false, true, true};

  auto data = make_table(
    int64_column_wrapper(keys.begin(), keys.end()).release(),
    int64_column_wrapper(ordering.begin(), ordering.end()).release(),
    int64_column_wrapper(payload.begin(), payload.end(), payload_valid.begin()).release());
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_masked_table(bool_column_wrapper(mask.begin(), mask.end()).release(), data));

  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::nullable},
    {int64_type, int64_type, int64_type},
    1};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(min_by, scan, chunks);

  auto expected_keys    = int64_column_wrapper{1};
  auto expected_payload = int64_column_wrapper{70};
  expect_tables_equal_unordered(result->view(),
                                cudf::table_view{{expected_keys, expected_payload}});
}

/**
 * @brief DECIMAL128 MIN/MAX combine through the 128-bit CAS on the mirrored accumulator.
 * Covers negative values (signed 128-bit comparison), NULL values, and an all-NULL group.
 */
TEST(QepBoundedHashAggregateTaskTest, Decimal128MinAndMax)
{
  if (!supports_128_bit_cas()) { GTEST_SKIP() << cas_128_skip_message; }

  using decimal128_wrapper = cudf::test::fixed_point_column_wrapper<__int128_t>;
  auto const scale         = numeric::scale_type{-2};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 1, 2, 2, 3, 4, 4}.release(),
                              decimal128_wrapper({12345, -99999, 500, -99998, 12346, 7, 7},
                                                 {true, true, true, true, true, false, false},
                                                 scale)
                                .release()));
  auto const decimal128_type = cudf::data_type{cudf::type_id::DECIMAL128, -2};

  qep::bounded_hash_aggregate_build_task const min_task{
    0,
    {cudf::aggregation::MIN, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, decimal128_type},
    4};
  expect_aggregate_matches_groupby(
    min_task, chunks, cudf::make_min_aggregation<cudf::groupby_aggregation>());

  qep::bounded_hash_aggregate_build_task const max_task{
    0,
    {cudf::aggregation::MAX, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, decimal128_type},
    4};
  expect_aggregate_matches_groupby(
    max_task, chunks, cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

/**
 * @brief SUM_OF_SQUARES squares after widening, so INT32 inputs whose squares overflow INT32
 * still accumulate exactly into the INT64 target.
 */
TEST(QepBoundedHashAggregateTaskTest, SumOfSquaresWidensToInt64)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int32_column_wrapper{1, 1, 2, 2}.release(),
                              int32_column_wrapper{100000, 3, -100000, 4}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM_OF_SQUARES, 1, std::nullopt, gqe::nullability::nullable},
    {int32_type, int64_type},
    2};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());
}

// ----- NULL keys, sentinel keys, NULL values -----

TEST(QepBoundedHashAggregateTaskTest, NullKeysFormOneGroup)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_table(int64_column_wrapper{{1, 0, 2, 0, 1}, {true, false, true, false, true}}.release(),
               int64_column_wrapper{10, 20, 30, 40, 50}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    2};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

TEST(QepBoundedHashAggregateTaskTest, SentinelValuedKeyIsALegalGroup)
{
  std::vector<int64_t> const key_values{int64_sentinel, 1, int64_sentinel, 1};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper(key_values.begin(), key_values.end()).release(),
                              int64_column_wrapper{10, 20, 30, 40}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    1};

  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

/**
 * @brief Group 1 has only NULL values (SUM is NULL); group 2 mixes (nulls skipped). The NULL-key
 * group also carries only NULL values, covering the out-of-band value-validity path.
 */
TEST(QepBoundedHashAggregateTaskTest, AllNullValueGroupAggregatesToNull)
{
  auto keys   = int64_column_wrapper{{1, 1, 2, 2, 0}, {true, true, true, true, false}};
  auto values = int64_column_wrapper{{7, 7, 5, 7, 7}, {false, false, true, false, false}};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper{{1, 1, 2, 2, 0}, {true, true, true, true, false}}.release(),
    int64_column_wrapper{{7, 7, 5, 7, 7}, {false, false, true, false, false}}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  auto const expected =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

/**
 * @brief Every value of every group is NULL, so no combine ever fires and the whole aggregate
 * column comes back NULL while all groups still appear (ports the executor groupby suite's
 * zero_valid_values case).
 */
TEST(QepBoundedHashAggregateTaskTest, EveryValueNullYieldsAllNullSums)
{
  auto keys   = int64_column_wrapper{1, 1, 2, 3, 3};
  auto values = int64_column_wrapper{{7, 7, 7, 7, 7}, {false, false, false, false, false}};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper{1, 1, 2, 3, 3}.release(),
    int64_column_wrapper{{7, 7, 7, 7, 7}, {false, false, false, false, false}}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  EXPECT_EQ(result->num_rows(), 3);
  EXPECT_EQ(result->get_column(1).null_count(), 3);
  auto const expected =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

/**
 * @brief NaN values are valid (not NULL) and propagate through the sum like in cuDF (ports the
 * executor groupby suite's values_with_nan case). NaN addition is commutative, so the
 * result is accumulation-order independent.
 */
TEST(QepBoundedHashAggregateTaskTest, SumFloat64PropagatesNan)
{
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  auto keys            = int64_column_wrapper{1, 2, 1, 2, 3};
  auto values          = double_column_wrapper{1.0, nan, 2.0, 3.0, nan};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 1, 2, 3}.release(),
                              double_column_wrapper{1.0, nan, 2.0, 3.0, nan}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, float64_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  auto const expected =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

// ----- Masked chunks -----

/**
 * @brief A null mask entry excludes the row, like false. Reference: filter, then aggregate.
 */
TEST(QepBoundedHashAggregateTaskTest, MaskedChunksSkipExcludedRows)
{
  auto const mask_wrapper =
    bool_column_wrapper{{true, false, true, true, false}, {true, true, true, false, true}};
  auto keys   = int64_column_wrapper{1, 1, 2, 2, 2};
  auto values = int64_column_wrapper{10, 20, 30, 40, 50};

  auto data = make_table(int64_column_wrapper{1, 1, 2, 2, 2}.release(),
                         int64_column_wrapper{10, 20, 30, 40, 50}.release());
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_masked_table(
    bool_column_wrapper{{true, false, true, true, false}, {true, true, true, false, true}}
      .release(),
    data));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  auto const filtered = cudf::apply_retention_mask(cudf::table_view{{keys, values}}, mask_wrapper);
  auto const expected = reference_groupby(filtered->view().column(0),
                                          filtered->view().column(1),
                                          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

/**
 * @brief NULLs and the valid-row mask combined: the mask excludes rows with valid and with NULL
 * keys, the surviving rows still carry NULL keys and NULL values, and one surviving group
 * (key 3) keeps only NULL values. Reference: filter by the mask, then aggregate.
 */
TEST(QepBoundedHashAggregateTaskTest, MaskedChunksWithNullKeysAndNullValues)
{
  auto const mask_wrapper = bool_column_wrapper{{true, false, true, true, false, true, true, false},
                                                {true, true, true, false, true, true, true, true}};
  auto keys               = int64_column_wrapper{{1, 1, 0, 2, 0, 0, 3, 3},
                                                 {true, true, false, true, false, false, true, true}};
  auto values             = int64_column_wrapper{{10, 20, 30, 40, 50, 60, 70, 80},
                                                 {true, true, false, true, true, true, false, true}};

  auto data = make_table(int64_column_wrapper{{1, 1, 0, 2, 0, 0, 3, 3},
                                              {true, true, false, true, false, false, true, true}}
                           .release(),
                         int64_column_wrapper{{10, 20, 30, 40, 50, 60, 70, 80},
                                              {true, true, false, true, true, true, false, true}}
                           .release());
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_masked_table(bool_column_wrapper{{true, false, true, true, false, true, true, false},
                                          {true, true, true, false, true, true, true, true}}
                        .release(),
                      data));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  auto const filtered = cudf::apply_retention_mask(cudf::table_view{{keys, values}}, mask_wrapper);
  auto const expected = reference_groupby(filtered->view().column(0),
                                          filtered->view().column(1),
                                          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

// ----- Value columns declared NOT NULL -----

/**
 * @brief A value column declared NOT NULL aggregates like a nullable one, with the result
 * column non-nullable.
 *
 * The chunks exercise everything the declaration does not cover:
 *
 *  - A NULL key.
 *  - The sentinel key.
 *  - A valid-row mask with a NULL entry.
 *  - A value column that carries an all-valid null mask, which the declaration still admits.
 */
TEST(QepBoundedHashAggregateTaskTest, NotNullValuesAggregateWithoutValidityTracking)
{
  // The reference input: both chunks concatenated, with the valid-row mask of the first chunk
  // extended by all-true entries for the unmasked second chunk.
  std::vector<int64_t> const key_values{1, 0, 1, 2, 2, 1, int64_sentinel, 2};
  std::vector<bool> const key_valid{true, false, true, true, true, true, true, true};
  std::vector<int64_t> const value_values{10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<bool> const mask_values{true, true, false, true, true, true, true, true};
  std::vector<bool> const mask_valid{true, true, true, false, true, true, true, true};
  auto keys   = int64_column_wrapper(key_values.begin(), key_values.end(), key_valid.begin());
  auto values = int64_column_wrapper(value_values.begin(), value_values.end());
  auto mask   = bool_column_wrapper(mask_values.begin(), mask_values.end(), mask_valid.begin());

  std::vector<qep::state_container> chunks;
  auto masked_data =
    make_table(int64_column_wrapper{{1, 0, 1, 2}, {true, false, true, true}}.release(),
               int64_column_wrapper{{10, 20, 30, 40}, {true, true, true, true}}.release());
  chunks.push_back(make_masked_table(
    bool_column_wrapper{{true, true, false, true}, {true, true, true, false}}.release(),
    masked_data));
  chunks.push_back(
    make_table(int64_column_wrapper(key_values.begin() + 4, key_values.end()).release(),
               int64_column_wrapper(value_values.begin() + 4, value_values.end()).release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::not_null},
    {int64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);
  EXPECT_FALSE(result->get_column(1).nullable());

  auto const filtered = cudf::apply_retention_mask(cudf::table_view{{keys, values}}, mask);
  auto const expected = reference_groupby(filtered->view().column(0),
                                          filtered->view().column(1),
                                          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

/**
 * @brief The split DECIMAL128 SUM over a value column declared NOT NULL, the TPC-H Q18 shape.
 */
TEST(QepBoundedHashAggregateTaskTest, NotNullDecimal64SumWidensToDecimal128)
{
  using decimal64_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;
  auto const scale        = numeric::scale_type{-2};
  cudf::data_type const decimal128_type{cudf::type_id::DECIMAL128, -2};

  auto keys   = int64_column_wrapper{1, 2, 1, 2, 3};
  auto values = decimal64_wrapper{{125, 250, 375, -500, 1}, scale};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 1}.release(),
                              decimal64_wrapper{{125, 250, 375}, scale}.release()));
  chunks.push_back(make_table(int64_column_wrapper{2, 3}.release(),
                              decimal64_wrapper{{-500, 1}, scale}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::not_null},
    {int64_type, decimal128_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);
  EXPECT_FALSE(result->get_column(1).nullable());

  auto const narrow =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto const widened  = cudf::cast(narrow->view().column(1), decimal128_type);
  auto const expected = cudf::table_view{{narrow->view().column(0), widened->view()}};
  expect_tables_equal_unordered(result->view(), expected);
}

/**
 * @brief The NOT NULL declaration covers the ARGMIN/ARGMAX payload column too, so the kernel
 * reads neither the ordering nor the payload null mask. Winners are fixed by construction.
 */
TEST(QepBoundedHashAggregateTaskTest, NotNullMinByAndMaxByReadNoMasks)
{
  if (!supports_128_bit_cas()) { GTEST_SKIP() << cas_128_skip_message; }

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 1, 2, 2, 3}.release(),
                              int64_column_wrapper{50, 30, 70, 90, 7}.release(),
                              double_column_wrapper{1.5, 2.5, 3.5, 4.5, 5.5}.release()));
  qep::bounded_hash_aggregate_scan_task const scan{16};

  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::not_null},
    {int64_type, float64_type, int64_type},
    3};
  auto const min_result = run_aggregate(min_by, scan, chunks);
  EXPECT_FALSE(min_result->get_column(1).nullable());
  auto const min_expected_keys    = int64_column_wrapper{1, 2, 3};
  auto const min_expected_payload = double_column_wrapper{2.5, 3.5, 5.5};
  expect_tables_equal_unordered(min_result->view(),
                                cudf::table_view{{min_expected_keys, min_expected_payload}});

  qep::bounded_hash_aggregate_build_task const max_by{
    0,
    {cudf::aggregation::ARGMAX, 1, 2, gqe::nullability::not_null},
    {int64_type, float64_type, int64_type},
    3};
  auto const max_result = run_aggregate(max_by, scan, chunks);
  EXPECT_FALSE(max_result->get_column(1).nullable());
  auto const max_expected_keys    = int64_column_wrapper{1, 2, 3};
  auto const max_expected_payload = double_column_wrapper{1.5, 4.5, 5.5};
  expect_tables_equal_unordered(max_result->view(),
                                cudf::table_view{{max_expected_keys, max_expected_payload}});
}

/**
 * @brief Counts record no validity, so a NOT NULL declaration only drops the mask read, and
 * COUNT_VALID then counts every selected row.
 */
TEST(QepBoundedHashAggregateTaskTest, NotNullCountValidCountsEveryRow)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper{1, 1, 2, 3, 3, 3}.release(),
    int64_column_wrapper{{7, 7, 7, 7, 7, 7}, {true, true, true, true, true, true}}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::COUNT_VALID, 1, std::nullopt, gqe::nullability::not_null},
    {int64_type, int32_type},
    3};
  expect_aggregate_matches_groupby(
    build,
    chunks,
    cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
}

/**
 * @brief A NULL in a masked-out row does not contradict the NOT NULL declaration.
 *
 *  - The SUM chunk hides a NULL value behind the valid-row mask.
 *  - The ARGMIN chunk hides a NULL payload behind the mask.
 *  - Reference: filter by the mask, then aggregate.
 */
TEST(QepBoundedHashAggregateTaskTest, NotNullDeclarationIgnoresMaskedOutNulls)
{
  auto const mask_wrapper = bool_column_wrapper{{true, false, true, true, false}};
  auto keys               = int64_column_wrapper{1, 1, 2, 2, 2};
  auto values = int64_column_wrapper{{10, 20, 30, 40, 50}, {true, false, true, true, false}};

  std::vector<qep::state_container> chunks;
  auto data = make_table(
    int64_column_wrapper{1, 1, 2, 2, 2}.release(),
    int64_column_wrapper{{10, 20, 30, 40, 50}, {true, false, true, true, false}}.release());
  chunks.push_back(
    make_masked_table(bool_column_wrapper{{true, false, true, true, false}}.release(), data));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::not_null},
    {int64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);
  EXPECT_FALSE(result->get_column(1).nullable());

  auto const filtered = cudf::apply_retention_mask(cudf::table_view{{keys, values}}, mask_wrapper);
  auto const expected = reference_groupby(filtered->view().column(0),
                                          filtered->view().column(1),
                                          cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());

  if (!supports_128_bit_cas()) { return; }
  std::vector<qep::state_container> payload_chunks;
  auto payload_data =
    make_table(int64_column_wrapper{1, 1, 2}.release(),
               int64_column_wrapper{10, 20, 30}.release(),
               double_column_wrapper{{1.5, 2.5, 3.5}, {true, false, true}}.release());
  payload_chunks.push_back(
    make_masked_table(bool_column_wrapper{{true, false, true}}.release(), payload_data));
  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::not_null},
    {int64_type, float64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const payload_scan{16};
  auto const payload_result = run_aggregate(min_by, payload_scan, payload_chunks);
  EXPECT_FALSE(payload_result->get_column(1).nullable());
  // Key 1 keeps only row 0, whose payload is 1.5. Key 2 keeps row 2, whose payload is 3.5.
  auto const expected_keys    = int64_column_wrapper{1, 2};
  auto const expected_payload = double_column_wrapper{1.5, 3.5};
  expect_tables_equal_unordered(payload_result->view(),
                                cudf::table_view{{expected_keys, expected_payload}});
}

/**
 * @brief A NULL in a selected row of a column declared NOT NULL makes `finalize` throw.
 *
 *  - A masked chunk that selects its NULL value.
 *  - An unmasked chunk with a NULL value.
 *  - An unmasked chunk with a NULL ARGMIN payload.
 */
TEST(QepBoundedHashAggregateTaskTest, NotNullDeclarationViolatedByAChunkThrows)
{
  std::vector<qep::state_container> masked_chunks;
  auto masked_data = make_table(int64_column_wrapper{1, 1, 2}.release(),
                                int64_column_wrapper{{10, 20, 30}, {true, false, true}}.release());
  masked_chunks.push_back(
    make_masked_table(bool_column_wrapper{{false, true, true}}.release(), masked_data));
  qep::bounded_hash_aggregate_build_task const masked_build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::not_null},
    {int64_type, int64_type},
    2};
  EXPECT_THROW(std::ignore = run_build(masked_build, masked_chunks), std::runtime_error);

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 1, 2}.release(),
                              int64_column_wrapper{{10, 20, 30}, {true, false, true}}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::not_null},
    {int64_type, int64_type},
    2};
  EXPECT_THROW(std::ignore = run_build(build, chunks), std::runtime_error);

  if (!supports_128_bit_cas()) { return; }
  std::vector<qep::state_container> payload_chunks;
  payload_chunks.push_back(
    make_table(int64_column_wrapper{1, 1, 2}.release(),
               int64_column_wrapper{10, 20, 30}.release(),
               double_column_wrapper{{1.5, 2.5, 3.5}, {true, false, true}}.release()));
  qep::bounded_hash_aggregate_build_task const min_by{
    0,
    {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::not_null},
    {int64_type, float64_type, int64_type},
    2};
  EXPECT_THROW(std::ignore = run_build(min_by, payload_chunks), std::runtime_error);
}

// ----- Empty input and empty chunks -----

TEST(QepBoundedHashAggregateTaskTest, EmptyInputEmitsSchemaCorrectEmptyResult)
{
  std::vector<qep::state_container> const chunks;

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, float64_type},
    4};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const outputs = run_scan(scan, run_build(build, chunks));

  // The scan must emit at least one schema-correct chunk even for an empty map.
  ASSERT_FALSE(outputs.empty());
  auto const result = concatenate_chunks(outputs);
  EXPECT_EQ(result->num_rows(), 0);
  ASSERT_EQ(result->num_columns(), 2);
  EXPECT_EQ(result->get_column(0).type(), int64_type);
  EXPECT_EQ(result->get_column(1).type(), float64_type);
}

TEST(QepBoundedHashAggregateTaskTest, EmptyChunksAreNoOps)
{
  auto keys   = int64_column_wrapper{1, 2};
  auto values = int64_column_wrapper{3, 4};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{}.release(), int64_column_wrapper{}.release()));
  chunks.push_back(
    make_table(int64_column_wrapper{1, 2}.release(), int64_column_wrapper{3, 4}.release()));
  chunks.push_back(make_table(int64_column_wrapper{}.release(), int64_column_wrapper{}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  auto const expected =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

// ----- Scan chunking -----

/**
 * @brief Includes NULL and sentinel keys so the out-of-band append lands in exactly one chunk.
 */
TEST(QepBoundedHashAggregateTaskTest, ResultsAreIdenticalAcrossScanWindowSizes)
{
  std::vector<int64_t> const keys{1, 2, 3, 4, 5, 6, 7, 8, 0, int64_sentinel};
  std::vector<bool> const key_validity{true, true, true, true, true, true, true, true, false, true};
  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_table(int64_column_wrapper(keys.begin(), keys.end(), key_validity.begin()).release(),
               int64_column_wrapper{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    8};

  std::unique_ptr<cudf::table> baseline;
  for (auto const target_chunk_rows : {1, 3, 1 << 20}) {
    qep::bounded_hash_aggregate_scan_task const scan{target_chunk_rows};
    auto const outputs = run_scan(scan, run_build(build, chunks));
    if (target_chunk_rows == 1) {
      // A one-row target must split the scan into multiple chunks.
      EXPECT_GT(outputs.size(), 1);
    }
    auto result = concatenate_chunks(outputs);
    if (baseline == nullptr) {
      baseline = std::move(result);
    } else {
      expect_tables_equal_unordered(result->view(), baseline->view());
    }
  }
}

/**
 * @brief Includes NULL and sentinel keys, so exactly one split (the last) may emit the out-of-band
 * rows. The union of all splits' outputs must equal the unsplit result, the per-split row
 * counts must sum to it (disjointness), and each split emits exactly one chunk per the
 * splittable-trait contract, empty slices included.
 */
TEST(QepBoundedHashAggregateTaskTest, SplitScansPartitionTheMap)
{
  std::vector<int64_t> const key_values{1, 2, 3, 4, 5, 6, 7, 8, 0, int64_sentinel};
  std::vector<bool> const key_validity{true, true, true, true, true, true, true, true, false, true};
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(
    int64_column_wrapper(key_values.begin(), key_values.end(), key_validity.begin()).release(),
    int64_column_wrapper{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    8};
  qep::bounded_hash_aggregate_scan_task const scan{2};
  EXPECT_GE(scan.max_splits(), 1);

  auto const baseline = run_aggregate(build, scan, chunks);

  // Both a small split count and one that exceeds the map capacity (empty slices).
  for (auto const split_count : {3, 64}) {
    auto const map = run_build(build, chunks);

    std::vector<qep::state_container> outputs;
    cudf::size_type split_rows = 0;
    for (cudf::size_type split_index = 0; split_index < split_count; ++split_index) {
      auto const split_scan = scan.split(split_index, split_count);
      auto split_outputs    = run_scan(*split_scan, map);
      // Exactly one schema-correct chunk per split, empty slices included.
      EXPECT_EQ(split_outputs.size(), 1);
      split_rows += qep::get_row_count(split_outputs.front());
      outputs.push_back(std::move(split_outputs.front()));
    }

    auto const result = concatenate_chunks(outputs);
    EXPECT_EQ(split_rows, baseline->num_rows());
    expect_tables_equal_unordered(result->view(), baseline->view());
  }
}

TEST(QepBoundedHashAggregateTaskTest, SingleOutputEmitsTheMapAsOneChunk)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 3, 1, 2}.release(),
                              int64_column_wrapper{1, 2, 3, 4, 5}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{
    qep::bounded_hash_aggregate_scan_task::single_output};
  auto const outputs = run_scan(scan, run_build(build, chunks));

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(qep::get_row_count(outputs.front()), 3);
}

/**
 * @brief A row bound of at least 2^16 rounds the scan windows down to whole occupancy-histogram
 * segments, so every chunk's row count comes from the histogram (the sync-free path).
 * 150,000 groups make the map span three segments and the scan emit multiple windows.
 */
TEST(QepBoundedHashAggregateTaskTest, SegmentAlignedWindowsScanLargeMaps)
{
  constexpr cudf::size_type num_groups = 150000;
  std::vector<int64_t> key_values(num_groups);
  std::iota(key_values.begin(), key_values.end(), 0);
  auto keys   = int64_column_wrapper(key_values.begin(), key_values.end());
  auto values = int64_column_wrapper(key_values.begin(), key_values.end());

  std::vector<qep::state_container> chunks;
  chunks.push_back(
    make_table(int64_column_wrapper(key_values.begin(), key_values.end()).release(),
               int64_column_wrapper(key_values.begin(), key_values.end()).release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    num_groups};
  // The row bound of 100,000 rounds down to one 65,536-slot segment per window.
  qep::bounded_hash_aggregate_scan_task const scan{100000};
  auto const outputs = run_scan(scan, run_build(build, chunks));

  EXPECT_GE(outputs.size(), 3);
  cudf::size_type total_rows = 0;
  for (auto const& chunk : outputs) {
    auto const rows = qep::get_row_count(chunk);
    EXPECT_LE(rows, 65536);
    total_rows += rows;
  }
  EXPECT_EQ(total_rows, num_groups);

  auto const result = concatenate_chunks(outputs);
  auto const expected =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

// ----- Bound violation -----

TEST(QepBoundedHashAggregateTaskTest, ViolatedBoundThrowsInsteadOfHanging)
{
  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 3, 4, 5, 6, 7, 8}.release(),
                              int64_column_wrapper{1, 1, 1, 1, 1, 1, 1, 1}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    2};

  gqe::context_reference ctx_ref{};
  auto accumulator = build.initialize(ctx_ref);
  for (auto const& chunk : chunks) {
    build.next(chunk, accumulator, ctx_ref);
  }
  EXPECT_THROW(static_cast<void>(build.finalize(std::move(accumulator), ctx_ref)),
               std::runtime_error);
}

/**
 * @brief Five distinct keys against a bound of four fit the map's load-factor headroom, so no
 * saturation occurs and only the occupancy-histogram count in finalize can detect the excess.
 * Exactly four distinct keys must pass.
 */
TEST(QepBoundedHashAggregateTaskTest, ExceededBoundThrowsAtFinalize)
{
  auto const run_with_keys = [](std::vector<int64_t> const& key_values) {
    std::vector<qep::state_container> chunks;
    chunks.push_back(
      make_table(int64_column_wrapper(key_values.begin(), key_values.end()).release(),
                 int64_column_wrapper(key_values.begin(), key_values.end()).release()));
    qep::bounded_hash_aggregate_build_task const build{
      0,
      {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
      {int64_type, int64_type},
      4};
    return run_build(build, chunks);
  };

  EXPECT_NO_THROW(static_cast<void>(run_with_keys({1, 2, 3, 4})));
  EXPECT_THROW(static_cast<void>(run_with_keys({1, 2, 3, 4, 5})), std::runtime_error);
}

// ----- Concurrency -----

TEST(QepBoundedHashAggregateTaskTest, ConcurrentNextFromMultipleThreads)
{
  constexpr int num_chunks     = 8;
  constexpr int rows_per_chunk = 1000;
  constexpr int num_groups     = 37;

  std::vector<qep::state_container> chunks;
  std::vector<int64_t> all_keys;
  std::vector<int64_t> all_values;
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    std::vector<int64_t> keys(rows_per_chunk);
    std::vector<int64_t> values(rows_per_chunk);
    for (int row = 0; row < rows_per_chunk; ++row) {
      keys[row]   = (chunk * rows_per_chunk + row) % num_groups;
      values[row] = chunk + row;
      all_keys.push_back(keys[row]);
      all_values.push_back(values[row]);
    }
    chunks.push_back(make_table(int64_column_wrapper(keys.begin(), keys.end()).release(),
                                int64_column_wrapper(values.begin(), values.end()).release()));
  }

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    num_groups};

  gqe::context_reference ctx_ref{};
  auto accumulator = build.initialize(ctx_ref);
  // Acting as the executor: order `initialize` before every thread's `next` (each `std::async`
  // thread runs on its own per-thread default stream), and drain each thread's stream before
  // `finalize`.
  gqe::query_stream::default_stream().ref().sync();

  std::vector<std::future<void>> workers;
  workers.reserve(num_chunks);
  for (int chunk = 0; chunk < num_chunks; ++chunk) {
    workers.push_back(std::async(std::launch::async, [&, chunk]() {
      build.next(chunks[chunk], accumulator, ctx_ref);
      gqe::query_stream::default_stream().ref().sync();
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }

  auto map = build.finalize(std::move(accumulator), ctx_ref);
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = concatenate_chunks(run_scan(scan, std::move(map)));

  auto const keys_column   = int64_column_wrapper(all_keys.begin(), all_keys.end());
  auto const values_column = int64_column_wrapper(all_values.begin(), all_values.end());
  auto const expected      = reference_groupby(
    keys_column, values_column, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

// ----- Allocation accounting -----

TEST(QepBoundedHashAggregateTaskTest, MapIsAllocatedOnceAndFreedOnce)
{
  rmm::mr::tracking_resource_adaptor build_mr{cudf::get_current_device_resource_ref()};
  rmm::mr::tracking_resource_adaptor scan_mr{cudf::get_current_device_resource_ref()};
  gqe::context_reference ctx_ref{};
  auto const stream_handle = gqe::query_stream::default_stream();

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    64};

  auto accumulator     = build.initialize(ctx_ref, stream_handle, build_mr);
  auto const map_bytes = build_mr.get_allocated_bytes();
  EXPECT_GT(map_bytes, 0);

  // Unmasked chunks allocate nothing: per-chunk footprint must not grow with chunk count.
  for (int chunk = 0; chunk < 4; ++chunk) {
    auto const input =
      make_table(int64_column_wrapper{1, 2, 3}.release(), int64_column_wrapper{4, 5, 6}.release());
    build.next(input, accumulator, ctx_ref, stream_handle, build_mr);
    EXPECT_EQ(build_mr.get_allocated_bytes(), map_bytes);
  }

  auto map = build.finalize(std::move(accumulator), ctx_ref, stream_handle, build_mr);
  EXPECT_EQ(build_mr.get_allocated_bytes(), map_bytes);

  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto iterator = scan.initialize(map, ctx_ref, stream_handle, scan_mr);
  map.clear();

  std::vector<qep::state_container> outputs;
  while (auto chunk = scan.next(iterator, ctx_ref, stream_handle, scan_mr)) {
    outputs.push_back(std::move(*chunk));
  }
  scan.finalize(std::move(iterator), ctx_ref, stream_handle, scan_mr);
  // The executor is responsible for the state's lifetime: dropping the last container
  // reference must free the map exactly once.
  iterator.clear();
  EXPECT_EQ(build_mr.get_allocated_bytes(), 0);
  // The scan outputs remain allocated from the scan's memory resource until dropped.
  EXPECT_GT(scan_mr.get_allocated_bytes(), 0);
  outputs.clear();
  EXPECT_EQ(scan_mr.get_allocated_bytes(), 0);
}

// ----- Fixed-point and date types -----

TEST(QepBoundedHashAggregateTaskTest, Decimal64SumWidensToDecimal128)
{
  using decimal64_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;
  auto const scale        = numeric::scale_type{-2};
  cudf::data_type const decimal64_type{cudf::type_id::DECIMAL64, -2};
  cudf::data_type const decimal128_type{cudf::type_id::DECIMAL128, -2};

  auto keys   = int64_column_wrapper{1, 2, 1, 2};
  auto values = decimal64_wrapper{{125, 250, 375, -500}, scale};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(int64_column_wrapper{1, 2, 1, 2}.release(),
                              decimal64_wrapper{{125, 250, 375, -500}, scale}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, decimal128_type},
    2};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  // cuDF keeps fixed-point SUM in the source type; widen its result for comparison.
  auto const narrow =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto const widened  = cudf::cast(narrow->view().column(1), decimal128_type);
  auto const expected = cudf::table_view{{narrow->view().column(0), widened->view()}};
  expect_tables_equal_unordered(result->view(), expected);
}

/**
 * @brief DECIMAL128 SUMs that leave the int64 range, so the split accumulator's wrap counter is
 * used.
 *
 * Each round is its own chunk, so its contributions reach the map as separate atomics. Key 1
 * wraps past +2^63. Key 2 wraps past -2^63 and back. Key 3 receives contributions outside the
 * int64 range. Key 4 stays small and negative, which must not touch the wrap counter at all.
 */
TEST(QepBoundedHashAggregateTaskTest, Decimal128SumCrossesTheInt64Range)
{
  using decimal128_wrapper = cudf::test::fixed_point_column_wrapper<__int128_t>;
  auto const scale         = numeric::scale_type{0};
  cudf::data_type const decimal128_type{cudf::type_id::DECIMAL128, 0};
  __int128_t const quarter = __int128_t{1} << 62;
  __int128_t const big     = (__int128_t{3} << 64) + 7;
  std::vector<std::vector<__int128_t>> const rounds{{quarter, -quarter, big, -1},
                                                    {quarter, -quarter, -5, -2},
                                                    {quarter, -quarter, 0, -3},
                                                    {quarter, quarter, 0, -4},
                                                    {quarter, quarter, -big, -5}};
  std::vector<int64_t> const keys{1, 2, 3, 4};
  std::vector<qep::state_container> chunks;
  for (auto const& round : rounds) {
    chunks.push_back(make_table(int64_column_wrapper(keys.begin(), keys.end()).release(),
                                decimal128_wrapper(round.begin(), round.end(), scale).release()));
  }
  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, decimal128_type},
    4};
  expect_aggregate_matches_groupby(
    build, chunks, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

TEST(QepBoundedHashAggregateTaskTest, TimestampKeysUseIntegerRepDispatch)
{
  using timestamp_wrapper =
    cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, cudf::timestamp_D::rep>;
  cudf::data_type const timestamp_type{cudf::type_id::TIMESTAMP_DAYS};

  auto keys   = timestamp_wrapper{100, 200, 100, 200, 300};
  auto values = int64_column_wrapper{1, 2, 3, 4, 5};

  std::vector<qep::state_container> chunks;
  chunks.push_back(make_table(timestamp_wrapper{100, 200, 100, 200, 300}.release(),
                              int64_column_wrapper{1, 2, 3, 4, 5}.release()));

  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {timestamp_type, int64_type},
    3};
  qep::bounded_hash_aggregate_scan_task const scan{16};
  auto const result = run_aggregate(build, scan, chunks);

  EXPECT_EQ(result->get_column(0).type(), timestamp_type);
  auto const expected =
    reference_groupby(keys, values, cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  expect_tables_equal_unordered(result->view(), expected->view());
}

TEST(QepBoundedHashAggregateTaskTest, IndirectInputsMatchDirectInputsWithAndWithoutMasks)
{
  auto source =
    make_table(int64_column_wrapper{10, 20, 30}.release(), int32_column_wrapper{1, 1, 2}.release());
  auto gather_source = qep::indirect_gather_source_state::make(
    qep::indirect_table_source_state::make(std::move(source)),
    qep::make_position_list(int32_column_wrapper{2, 0, 1, 0}.release(),
                            gqe::query_stream::default_stream()));
  // Mix indirect values/keys with a direct argument column in logical row order.
  auto logical = make_table(int64_column_wrapper{300, 100, 200, 100}.release());
  logical.insert(logical.begin(), qep::indirect_column_state::make(gather_source, 0));
  logical.push_back(qep::indirect_column_state::make(gather_source, 1));
  auto direct = make_table(int64_column_wrapper{30, 10, 20, 10}.release(),
                           int64_column_wrapper{300, 100, 200, 100}.release(),
                           int32_column_wrapper{2, 1, 1, 1}.release());

  for (bool const is_masked : {false, true}) {
    std::vector<qep::state_container> logical_chunks{logical};
    std::vector<qep::state_container> direct_chunks{direct};
    if (is_masked) {
      auto mask         = bool_column_wrapper{{true, true, true, false}, {true, false, true, true}};
      logical_chunks[0] = make_masked_table(mask.release(), logical);
      auto reference_mask =
        bool_column_wrapper{{true, true, true, false}, {true, false, true, true}};
      direct_chunks[0] = make_masked_table(reference_mask.release(), direct);
    }
    for (auto const aggregation :
         {qep::bounded_hash_aggregation{.kind = cudf::aggregation::SUM, .value_index = 0},
          qep::bounded_hash_aggregation{.kind = cudf::aggregation::SUM, .value_index = 2},
          qep::bounded_hash_aggregation{.kind = cudf::aggregation::COUNT_ALL},
          qep::bounded_hash_aggregation{
            .kind = cudf::aggregation::ARGMAX, .value_index = 0, .argument_index = 1}}) {
      bool const is_arg = aggregation.kind == cudf::aggregation::ARGMAX;
      if (is_arg && !supports_128_bit_cas()) { continue; }
      std::vector<cudf::data_type> types{int32_type, int64_type};
      if (is_arg) { types.push_back(int64_type); }
      qep::bounded_hash_aggregate_build_task const build{2, aggregation, types, 2};
      auto const clone = build.clone();
      qep::bounded_hash_aggregate_scan_task const scan{16};
      auto const actual = run_aggregate(
        dynamic_cast<qep::bounded_hash_aggregate_build_task const&>(*clone), scan, logical_chunks);
      auto const expected = run_aggregate(build, scan, direct_chunks);
      expect_tables_equal_unordered(actual->view(), expected->view());
    }
  }
}

// ----- Construction and schema rejection -----

TEST(QepBoundedHashAggregateTaskTest, UnsupportedShapesAreRejectedAtConstruction)
{
  cudf::data_type const decimal128_type{cudf::type_id::DECIMAL128, -2};

  // A DECIMAL128 PRODUCT changes the scale; rejected.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::PRODUCT, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, cudf::data_type{cudf::type_id::DECIMAL128, -2}},
                 4}),
               std::logic_error);
  // ARGMIN requires the argument column and the declared ordering type.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::ARGMIN, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, float64_type, int64_type},
                 4}),
               std::logic_error);
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::ARGMIN, 1, 2, gqe::nullability::nullable},
                 {int64_type, float64_type},
                 4}),
               std::logic_error);
  // An INT32 ordering type has no 8-byte storage; rejected.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::ARGMAX, 1, 2, gqe::nullability::nullable},
                 {int64_type, float64_type, int32_type},
                 4}),
               std::logic_error);
  // DECIMAL128 keys need an undecomposable 128-bit CAS; deferred.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
                 {decimal128_type, decimal128_type},
                 4}),
               std::logic_error);
  // Integral SUM always targets INT64.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, int32_type},
                 4}),
               std::logic_error);
  // The bound must be positive.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, int64_type},
                 0}),
               std::logic_error);
  // The load factor must be in (0, 1].
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, int64_type},
                 4,
                 0.0}),
               std::logic_error);
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, int64_type},
                 4,
                 1.5}),
               std::logic_error);
  // COUNT_ALL takes no value column; everything else requires one.
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::COUNT_ALL, 1, std::nullopt, gqe::nullability::nullable},
                 {int64_type, int32_type},
                 4}),
               std::logic_error);
  // Without a value column there is nothing to declare NOT NULL.
  EXPECT_THROW(
    (qep::bounded_hash_aggregate_build_task{
      0,
      {cudf::aggregation::COUNT_ALL, std::nullopt, std::nullopt, gqe::nullability::not_null},
      {int64_type, int32_type},
      4}),
    std::logic_error);
  EXPECT_THROW((qep::bounded_hash_aggregate_build_task{
                 0,
                 {cudf::aggregation::SUM, std::nullopt, std::nullopt, gqe::nullability::nullable},
                 {int64_type, int64_type},
                 4}),
               std::logic_error);
  // Exactly [key type, aggregate result type].
  EXPECT_THROW(
    (qep::bounded_hash_aggregate_build_task{
      0, {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable}, {int64_type}, 4}),
    std::logic_error);
  // The scan's chunk-row target must be positive, and a split must be a valid slice.
  EXPECT_THROW(qep::bounded_hash_aggregate_scan_task{0}, std::logic_error);
  qep::bounded_hash_aggregate_scan_task const splittable{16};
  EXPECT_THROW(static_cast<void>(splittable.split(2, 2)), std::logic_error);
  EXPECT_THROW(static_cast<void>(splittable.split(-1, 3)), std::logic_error);
  EXPECT_THROW(static_cast<void>(splittable.split(0, 0)), std::logic_error);
}

TEST(QepBoundedHashAggregateTaskTest, MismatchedChunkSchemaIsRejected)
{
  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    4};
  gqe::context_reference ctx_ref{};
  auto accumulator = build.initialize(ctx_ref);

  // Key type contradicts the declared key output type.
  auto const wrong_key =
    make_table(int32_column_wrapper{1}.release(), int64_column_wrapper{2}.release());
  EXPECT_THROW(build.next(wrong_key, accumulator, ctx_ref), std::logic_error);

  // Value index beyond the chunk width.
  auto const too_narrow = make_table(int64_column_wrapper{1}.release());
  EXPECT_THROW(build.next(too_narrow, accumulator, ctx_ref), std::out_of_range);
}

// ----- State-kind dispatch sites -----

TEST(QepBoundedHashAggregateTaskTest, MapStateRejectsRowCountAndTableViews)
{
  std::vector<qep::state_container> const chunks;
  qep::bounded_hash_aggregate_build_task const build{
    0,
    {cudf::aggregation::SUM, 1, std::nullopt, gqe::nullability::nullable},
    {int64_type, int64_type},
    4};
  auto const map = run_build(build, chunks);

  EXPECT_THROW(static_cast<void>(qep::get_row_count(map)), std::logic_error);
  EXPECT_FALSE(qep::to_table_view(map).has_value());

  // A shallow state copy aliases the same map: the executor's accumulator snapshots must pass
  // the kind through untouched.
  auto const copy = qep::make_mutable_state_copy(map);
  EXPECT_EQ(copy.front().get(), map.front().get());
}
