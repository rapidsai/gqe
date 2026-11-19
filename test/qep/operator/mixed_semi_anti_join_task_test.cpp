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

#include "semi_anti_join_test_cases.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/operator/mixed_semi_anti_join.hpp>
#include <gqe/qep/shapes/logical_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>

#include <cudf/sorting.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

namespace gqe_test {

/** @brief Execute logical semi/anti cases through the mixed join lifecycle. */
class MixedSemiAntiTestAdapter : public QepJoinTaskTestBase {
 public:
  qep::state_container execute(qep::state_container const& left,
                               qep::state_container const& right,
                               SemiAntiJoinTestOptions const& options,
                               std::unique_ptr<gqe::expression> residual = nullptr)
  {
    auto const left_width  = qep::logical_table_view::try_from(left)->num_columns();
    auto const right_width = qep::logical_table_view::try_from(right)->num_columns();
    qep::mixed_semi_anti_join_build_task build_task{
      right_width, options.keys, options.predicate_columns, options.null_equality};
    auto build = fold(build_task, {right});
    if (!residual) {
      // Equality-only cases need a residual that accepts every candidate pair.
      residual = std::make_unique<gqe::equal_expression>(
        std::make_shared<gqe::literal_expression<int32_t>>(1),
        std::make_shared<gqe::literal_expression<int32_t>>(1));
    }
    qep::mixed_semi_anti_join_probe_task probe_task{left_width,
                                                    right_width,
                                                    options.keys,
                                                    std::move(residual),
                                                    options.join_type,
                                                    options.projections,
                                                    options.output_types};
    auto output = probe(probe_task, build, left);
    return options.projections.empty() ? output : gather(output);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Mixed, QepSemiAntiJoinContractTest, MixedSemiAntiTestAdapter);
INSTANTIATE_TYPED_TEST_SUITE_P(Mixed, QepSemiAntiJoinResidualTest, MixedSemiAntiTestAdapter);

}  // namespace gqe_test

using namespace gqe_test;
namespace qep = gqe::qep;

/** @brief Exercises mixed selection, deferred payload, null, and ownership contracts. */
class QepMixedSemiAntiJoinTaskTest : public QepJoinTaskTestBase {
 protected:
  static auto predicate(cudf::size_type left = 1, cudf::size_type right = 3)
  {
    return std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::column_reference_expression>(left),
      std::make_shared<gqe::column_reference_expression>(right));
  }
  static auto make_probe(gqe::join_type_type mode, bool count = false)
  {
    return qep::mixed_semi_anti_join_probe_task{
      2,
      2,
      {0},
      predicate(),
      mode,
      count ? std::vector<cudf::size_type>{} : std::vector<cudf::size_type>{1, 0, 1},
      std::vector<cudf::data_type>(count ? 0 : 3, cudf::data_type{cudf::type_id::INT64})};
  }
  void expect_values(qep::state_container const& output,
                     std::vector<int64_t> values,
                     std::vector<int64_t> keys)
  {
    auto gathered = gather(output);
    auto sorted   = cudf::sort(*qep::to_table_view(gathered));
    int64_column_wrapper expected_values(values.begin(), values.end());
    int64_column_wrapper expected_keys(keys.begin(), keys.end());
    cudf::table_view expected{{expected_values, expected_keys, expected_values}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
  }
};

TEST_F(QepMixedSemiAntiJoinTaskTest, EqualityNullPolicyIsIndependentOfResidualNulls)
{
  // Columns are [key, value], and the residual is left.value < right.value.
  // Left rows are (null, 1), (1, 2), and (1, null); right rows are (null, 10) and (1, 10).
  // The null-key row depends on key equality policy. The null-value row never matches.
  for (auto policy : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    std::vector<std::unique_ptr<cudf::column>> right;
    right.push_back(int64_column_wrapper{{0, 1}, {false, true}}.release());
    right.push_back(int64_column_wrapper{10, 10}.release());
    std::vector<std::unique_ptr<cudf::column>> left;
    left.push_back(int64_column_wrapper{{0, 1, 1}, {false, true, true}}.release());
    left.push_back(int64_column_wrapper{{1, 2, 3}, {true, true, false}}.release());
    qep::mixed_semi_anti_join_build_task task{2, {0}, {1}, policy};
    auto build = fold(task, {table_state(std::move(right))});
    auto input = table_state(std::move(left));
    auto semi  = probe(make_probe(gqe::join_type_type::left_semi, true), build, input);
    auto anti  = probe(make_probe(gqe::join_type_type::left_anti, true), build, input);
    if (policy == cudf::null_equality::EQUAL) {
      EXPECT_EQ(qep::try_row_count(semi), 2) << "Both the null-key row and (1, 2) match.";
      EXPECT_EQ(qep::try_row_count(anti), 1) << "Only (1, null) has no true residual match.";
    } else {
      EXPECT_EQ(qep::try_row_count(semi), 1) << "Only (1, 2) matches.";
      EXPECT_EQ(qep::try_row_count(anti), 2) << "Neither the null-key nor null-value row matches.";
    }
  }
}

TEST_F(QepMixedSemiAntiJoinTaskTest, NoChunksAndSchemaBearingEmptyInputs)
{
  qep::mixed_semi_anti_join_build_task task{2, {0}, {1}, cudf::null_equality::UNEQUAL};
  for (bool has_build_schema : {false, true}) {
    SCOPED_TRACE(has_build_schema ? "Schema-bearing empty build" : "No build chunks");
    std::vector<qep::state_container> chunks;
    if (has_build_schema) { chunks.push_back(two_int64_columns({}, {})); }
    auto build = fold(task, chunks);
    for (bool masked_indirect : {false, true}) {
      SCOPED_TRACE(masked_indirect ? "Masked indirect probe" : "Direct probe");
      qep::state_container input;
      std::vector<int64_t> expected_values;
      std::vector<int64_t> expected_keys;
      if (masked_indirect) {
        // Indirection produces values [20, 10, 30, 20]. The false and null mask entries
        // remove the middle two rows, leaving two references to the source row (key 2, value 20).
        input = with_valid_mask(
          bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(),
          indirect_columns(two_int64_columns({1, 2, 3}, {10, 20, 30}), {1, 0, 2, 1}));
        expected_values = {20, 20};
        expected_keys   = {2, 2};
      } else {
        input           = two_int64_columns({1, 2}, {10, 20});
        expected_values = {10, 20};
        expected_keys   = {1, 2};
      }

      // An empty right side gives semi no matches; anti retains every valid left row.
      expect_values(probe(make_probe(gqe::join_type_type::left_semi), build, input), {}, {});
      expect_values(probe(make_probe(gqe::join_type_type::left_anti), build, input),
                    expected_values,
                    expected_keys);
      auto semi_count = probe(make_probe(gqe::join_type_type::left_semi, true), build, input);
      auto anti_count = probe(make_probe(gqe::join_type_type::left_anti, true), build, input);
      EXPECT_EQ(qep::try_row_count(semi_count), 0) << "No left row can match an empty build.";
      EXPECT_EQ(qep::try_row_count(anti_count), 2) << "Both valid left rows survive the anti join.";

      // An empty probe produces zero rows in either mode, regardless of the build representation.
      for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
        auto empty = probe(make_probe(mode, true), build, two_int64_columns({}, {}));
        EXPECT_EQ(qep::try_row_count(empty), 0);
      }
    }
  }

  auto nonempty_build = fold(task, {two_int64_columns({1}, {10})});
  auto empty_anti     = probe(
    make_probe(gqe::join_type_type::left_anti, true), nonempty_build, two_int64_columns({}, {}));
  EXPECT_EQ(qep::try_row_count(empty_anti), 0)
    << "A nonempty build cannot add rows to an empty probe.";
}

TEST_F(QepMixedSemiAntiJoinTaskTest, MaskedIndirectPayloadsShareSelectionAndSurviveInputs)
{
  qep::mixed_semi_anti_join_build_task task{2, {0}, {1}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {two_int64_columns({1}, {10}), two_int64_columns({2, 2}, {20, 30})});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{1, 2, 3, 2}.release());
  columns.push_back(int64_column_wrapper{1, 2, 3, 4}.release());
  columns.push_back(string_column_wrapper{"café", "東京", "skip", "four"}.release());
  auto input  = indirect_columns(table_state(std::move(columns)), {3, 0, 2, 1, 0});
  auto source = std::get<qep::state_kind::indirect_column>(*input[2])
                  .state()
                  ->gather_source()
                  ->relation_source();
  input = with_valid_mask(
    bool_column_wrapper{{true, true, true, true, false}, {true, true, false, true, true}}.release(),
    std::move(input));
  qep::mixed_semi_anti_join_probe_task probe_task{3,
                                                  2,
                                                  {0},
                                                  predicate(1, 4),
                                                  gqe::join_type_type::left_semi,
                                                  {2, 1, 2},
                                                  {cudf::data_type{cudf::type_id::STRING},
                                                   cudf::data_type{cudf::type_id::INT64},
                                                   cudf::data_type{cudf::type_id::STRING}}};
  cudf::get_default_stream().sync();
  gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
  auto const stream = stream_context.create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor resource{cudf::get_current_device_resource_ref()};
  auto output = probe(probe_task, build, input, stream, resource);
  input.clear();
  build.clear();
  stream.ref().sync();
  auto selected = std::get<qep::state_kind::indirect_column>(*output[0]).state()->gather_source();
  EXPECT_EQ(selected->relation_source(), source);
  EXPECT_EQ(selected,
            std::get<qep::state_kind::indirect_column>(*output[2]).state()->gather_source());
  EXPECT_TRUE(resource.get_outstanding_allocations().contains(
    const_cast<void*>(selected->position_list()->column().view().head())));
  auto gathered = gather(output);
  auto sorted   = cudf::sort(*qep::to_table_view(gathered));
  string_column_wrapper expected_strings{"café", "four", "東京"};
  int64_column_wrapper expected_values{1, 4, 2};
  cudf::table_view expected{{expected_strings, expected_values, expected_strings}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
}

TEST_F(QepMixedSemiAntiJoinTaskTest, MaskedBuildAndProbeKeepOperandAlignment)
{
  qep::mixed_semi_anti_join_build_task task{2, {0}, {1}, cudf::null_equality::UNEQUAL};
  auto build = fold(task,
                    {masked_two_int64_columns(
                      bool_column_wrapper{false, true, true}.release(), {1, 2, 3}, {100, 20, 1})});
  auto input = masked_two_int64_columns(
    bool_column_wrapper{true, false, true, true}.release(), {1, 9, 2, 3}, {1, 9, 2, 3});
  expect_values(probe(make_probe(gqe::join_type_type::left_semi), build, input), {2}, {2});
  expect_values(probe(make_probe(gqe::join_type_type::left_anti), build, input), {1, 3}, {1, 3});
}

TEST_F(QepMixedSemiAntiJoinTaskTest, LiteralPredicateAndClonesOwnTheirStateAcrossProbes)
{
  qep::mixed_semi_anti_join_build_task task{2, {0}, {}, cudf::null_equality::UNEQUAL};
  auto cloned_build = task.clone();
  auto build        = fold(static_cast<qep::mixed_semi_anti_join_build_task const&>(*cloned_build),
                           {two_int64_columns({1}, {99})});
  auto residual =
    std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::literal_expression<int64_t>>(10));
  auto task_probe = std::make_unique<qep::mixed_semi_anti_join_probe_task>(
    2,
    2,
    std::vector<cudf::size_type>{0},
    std::move(residual),
    gqe::join_type_type::left_semi,
    std::vector<cudf::size_type>{1, 0, 1},
    std::vector<cudf::data_type>(3, cudf::data_type{cudf::type_id::INT64}));
  auto clone = task_probe->clone();
  task_probe.reset();
  auto const& copied = static_cast<qep::mixed_semi_anti_join_probe_task const&>(*clone);
  auto state         = copied.initialize(build, ctx_ref);
  build.clear();
  auto first = copied.next(two_int64_columns({1, 1}, {5, 20}), state, ctx_ref);
  ASSERT_TRUE(first.has_value());
  expect_values(*first, {5}, {1});
  auto second = copied.next(two_int64_columns({1, 2}, {3, 4}), state, ctx_ref);
  ASSERT_TRUE(second.has_value());
  expect_values(*second, {3}, {1});
}

TEST_F(QepMixedSemiAntiJoinTaskTest, InitializedPredicatesRemainIndependentAcrossBuildLayouts)
{
  // Logical schemas are left [key, value] and right [key, high_limit, low_limit].
  // Combined predicate coordinates are [left.key, left.value, right.key, high_limit, low_limit].
  constexpr cudf::size_type left_value       = 1;
  constexpr cudf::size_type right_high_limit = 3;
  constexpr cudf::size_type right_low_limit  = 4;
  auto make_right_input                      = [] {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{1}.release());
    columns.push_back(int64_column_wrapper{10}.release());
    columns.push_back(int64_column_wrapper{2}.release());
    return table_state(std::move(columns));
  };
  // Physical build layouts are [key, high_limit] and [key, low_limit, high_limit].
  qep::mixed_semi_anti_join_build_task high_limit_build{3, {0}, {1}, cudf::null_equality::UNEQUAL};
  qep::mixed_semi_anti_join_build_task reordered_limits_build{
    3, {0}, {2, 1}, cudf::null_equality::UNEQUAL};
  auto high_limit_state       = fold(high_limit_build, {make_right_input()});
  auto reordered_limits_state = fold(reordered_limits_build, {make_right_input()});
  auto make_limit_probe       = [&](cudf::size_type logical_limit_index) {
    return qep::mixed_semi_anti_join_probe_task{
      2,
      3,
            {0},
      predicate(left_value, logical_limit_index),
      gqe::join_type_type::left_semi,
            {1, 0, 1},
      std::vector<cudf::data_type>(3, cudf::data_type{cudf::type_id::INT64})};
  };
  auto high_limit_probe              = make_limit_probe(right_high_limit);
  auto low_limit_probe               = make_limit_probe(right_low_limit);
  auto high_limit_on_narrow_build    = high_limit_probe.initialize(high_limit_state, ctx_ref);
  auto high_limit_on_reordered_build = high_limit_probe.initialize(reordered_limits_state, ctx_ref);
  auto low_limit_on_reordered_build  = low_limit_probe.initialize(reordered_limits_state, ctx_ref);
  high_limit_state.clear();
  reordered_limits_state.clear();
  auto input = two_int64_columns({1, 1}, {5, 1});

  // Both left values are below 10, regardless of the high limit's physical column position.
  for (auto const* state : {&high_limit_on_narrow_build, &high_limit_on_reordered_build}) {
    auto output = high_limit_probe.next(input, *state, ctx_ref);
    ASSERT_TRUE(output.has_value());
    expect_values(*output, {1, 5}, {1, 1});
  }
  // The independent predicate uses the low limit 2, so only left value 1 matches.
  auto output = low_limit_probe.next(input, low_limit_on_reordered_build, ctx_ref);
  ASSERT_TRUE(output.has_value());
  expect_values(*output, {1}, {1});

  // A schema-bearing build must include the requested limit, even when it has no rows.
  auto missing_low_limit = fold(high_limit_build, {make_right_input()});
  EXPECT_THROW((void)low_limit_probe.initialize(missing_low_limit, ctx_ref), std::invalid_argument);
  std::vector<std::unique_ptr<cudf::column>> empty_columns;
  for (int index = 0; index < 3; ++index) {
    empty_columns.push_back(int64_column_wrapper{}.release());
  }
  auto empty_missing_low_limit = fold(high_limit_build, {table_state(std::move(empty_columns))});
  EXPECT_THROW((void)low_limit_probe.initialize(empty_missing_low_limit, ctx_ref),
               std::invalid_argument);
  // No-chunk builds bypass the predicate, so they need no residual operand columns.
  auto no_chunk_build = fold(high_limit_build, {});
  EXPECT_NO_THROW((void)low_limit_probe.initialize(no_chunk_build, ctx_ref));
}

TEST_F(QepMixedSemiAntiJoinTaskTest, SparseReorderedOperandsKeepLogicalPredicateCoordinates)
{
  std::vector<std::unique_ptr<cudf::column>> right;
  right.push_back(int64_column_wrapper{10, 20}.release());
  right.push_back(int64_column_wrapper{99, 99}.release());
  right.push_back(int64_column_wrapper{1, 1}.release());
  qep::mixed_semi_anti_join_build_task build_task(3, {2}, {0}, cudf::null_equality::UNEQUAL);
  auto build = fold(build_task, {table_state(std::move(right))});

  std::vector<std::unique_ptr<cudf::column>> left;
  left.push_back(int64_column_wrapper{5, 15, 25}.release());
  left.push_back(int64_column_wrapper{99, 99, 99}.release());
  left.push_back(int64_column_wrapper{99, 99, 99}.release());
  left.push_back(int64_column_wrapper{1, 1, 1}.release());
  auto input = table_state(std::move(left));
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    // Materialized schemas are left [3, 0] and right [2, 0], while the residual stays [0, 4].
    qep::mixed_semi_anti_join_probe_task task(
      4,
      3,
      {3},
      predicate(0, 4),
      mode,
      {0, 3, 0},
      std::vector<cudf::data_type>(3, cudf::data_type{cudf::type_id::INT64}));
    expect_values(probe(task, build, input),
                  mode == gqe::join_type_type::left_semi ? std::vector<int64_t>{5, 15}
                                                         : std::vector<int64_t>{25},
                  mode == gqe::join_type_type::left_semi ? std::vector<int64_t>{1, 1}
                                                         : std::vector<int64_t>{1});
  }
}
