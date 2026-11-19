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

#include <gqe/qep/operator/equality_semi_anti_join.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>

#include <barrier>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gqe_test {

namespace qep = gqe::qep;

/** @brief Exercises the equality semi/anti build, probe, and gather task lifecycle. */
class QepEqualitySemiAntiJoinTaskTest : public QepJoinTaskTestBase {};

/** @brief Execute logical semi/anti cases by building right and probing left. */
class EqualitySemiAntiTestAdapter : public QepJoinTaskTestBase {
 public:
  qep::state_container execute(qep::state_container const& left,
                               qep::state_container const& right,
                               SemiAntiJoinTestOptions const& options)
  {
    qep::equality_semi_anti_join_build_task build_task{options.keys, options.null_equality};
    auto build = fold(build_task, {right});
    qep::equality_semi_anti_join_probe_task probe_task{
      options.keys, options.join_type, options.projections, options.output_types};
    auto output = probe(probe_task, build, left);
    return options.projections.empty() ? output : gather(output);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Equality, QepSemiAntiJoinContractTest, EqualitySemiAntiTestAdapter);
// Equality joins do not accept residual predicates.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QepSemiAntiJoinResidualTest);

/** @brief Exercises STRING equality keys and projected payloads for both join kinds. */
class QepEqualitySemiAntiJoinStringTest : public QepJoinTaskTestBase,
                                          public testing::WithParamInterface<gqe::join_type_type> {
};

TEST_P(QepEqualitySemiAntiJoinStringTest, StringKeysSelectProjectedStringPayloads)
{
  auto const string_type = cudf::data_type{cudf::type_id::STRING};
  std::vector<std::unique_ptr<cudf::column>> build_columns;
  build_columns.push_back(string_column_wrapper{"café", ""}.release());
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(build_task, {table_state(std::move(build_columns))});

  std::vector<std::unique_ptr<cudf::column>> probe_columns;
  probe_columns.push_back(string_column_wrapper{"pear", "café", "", "plum"}.release());
  probe_columns.push_back(
    string_column_wrapper{"pear payload", "café payload", "empty key", "plum payload"}.release());
  qep::equality_semi_anti_join_probe_task probe_task{
    {0}, GetParam(), {1, 0}, {string_type, string_type}};
  auto output = gather(probe(probe_task, build, table_state(std::move(probe_columns))));
  auto table  = qep::to_table_view(output);
  ASSERT_TRUE(table.has_value());
  ASSERT_EQ(table->num_columns(), 2);

  auto const is_semi = GetParam() == gqe::join_type_type::left_semi;
  auto expected_keys =
    is_semi ? string_column_wrapper{"café", ""} : string_column_wrapper{"pear", "plum"};
  auto expected_payload = is_semi ? string_column_wrapper{"café payload", "empty key"}
                                  : string_column_wrapper{"pear payload", "plum payload"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(0), expected_payload);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(table->column(1), expected_keys);
}

INSTANTIATE_TEST_SUITE_P(JoinKinds,
                         QepEqualitySemiAntiJoinStringTest,
                         testing::Values(gqe::join_type_type::left_semi,
                                         gqe::join_type_type::left_anti),
                         [](auto const& info) {
                           return info.param == gqe::join_type_type::left_semi ? "LeftSemi"
                                                                               : "LeftAnti";
                         });

TEST_F(QepEqualitySemiAntiJoinTaskTest, AbsentBuildChunksPreserveSemiAntiSemantics)
{
  auto const type = cudf::data_type{cudf::type_id::INT64};
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(build_task, {});
  auto input = two_int64_columns({1, 2}, {100, 200});
  qep::equality_semi_anti_join_probe_task semi_task{
    {0}, gqe::join_type_type::left_semi, {1}, {type}};
  auto semi = gather(probe(semi_task, build, input));
  EXPECT_EQ(qep::to_table_view(semi)->num_rows(), 0);
  qep::equality_semi_anti_join_probe_task anti_task{
    {0}, gqe::join_type_type::left_anti, {1}, {type}};
  auto anti = gather(probe(anti_task, build, input));
  int64_column_wrapper expected{100, 200};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(anti)->column(0), expected);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, MaskedIndirectInputsPreservePayloadSourcesAcrossJoins)
{
  auto const type = cudf::data_type{cudf::type_id::INT64};
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto build_input = indirect_columns(two_int64_columns({9, 1, 3}, {90, 10, 30}), {2, 0, 1});
  build_input =
    with_valid_mask(bool_column_wrapper{true, false, true}.release(), std::move(build_input));
  auto build = fold(build_task, {build_input});
  auto input = indirect_columns(two_int64_columns({1, 2, 3, 4}, {10, 20, 30, 40}), {3, 2, 0, 1});
  auto const source = qep::try_get_indirect_column(input[0])->gather_source()->relation_source();
  input = with_valid_mask(bool_column_wrapper{true, true, false, true}.release(), std::move(input));
  // Build keys after selection are {3, 1}. Probe rows are {(4, 40), (3, 30), (2, 20)}.
  // Semi keeps (3, 30). Anti keeps (4, 40) and (2, 20).
  for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    qep::equality_semi_anti_join_probe_task task{{0}, mode, {1, 0, 1}, {type, type, type}};
    auto output       = probe(task, build, input);
    auto const* first = qep::try_get_indirect_column(output[0]);
    EXPECT_EQ(first->gather_source()->relation_source(), source);
    EXPECT_EQ(first->gather_source(), qep::try_get_indirect_column(output[2])->gather_source());
    auto gathered = gather(output);
    auto expected = mode == gqe::join_type_type::left_semi ? two_int64_columns({30}, {3})
                                                           : two_int64_columns({40, 20}, {4, 2});
    auto table    = *qep::to_table_view(gathered);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(0), qep::to_table_view(expected)->column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(table.column(1), qep::to_table_view(expected)->column(1));

    // A second semi join consumes the first join's deferred output on both sides.
    qep::equality_semi_anti_join_build_task chained_build_task{{1}, cudf::null_equality::UNEQUAL};
    auto chained_build = fold(chained_build_task, {output});
    qep::equality_semi_anti_join_probe_task chained_probe{
      {1}, gqe::join_type_type::left_semi, {0, 2}, {type, type}};
    auto chained = gather(probe(chained_probe, chained_build, output));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(chained)->column(0), table.column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(chained)->column(1), table.column(2));
  }
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, MixedPayloadSourcesPreserveNullification)
{
  auto const type = cudf::data_type{cudf::type_id::INT64};
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(build_task, {two_int64_columns({1, 3}, {10, 30})});
  auto input = two_int64_columns({1, 2, 3}, {100, 200, 300});
  auto payload =
    indirect_columns(two_int64_columns({7}, {70}), {99, 0, 0}, cudf::out_of_bounds_policy::NULLIFY);
  input.push_back(payload[1]);
  input = with_valid_mask(bool_column_wrapper{true, false, true}.release(), std::move(input));
  qep::equality_semi_anti_join_probe_task projected{
    {0}, gqe::join_type_type::left_semi, {1, 2}, {type, type}};
  auto output   = probe(projected, build, input);
  auto gathered = gather(output);
  int64_column_wrapper direct_expected{100, 300};
  int64_column_wrapper indirect_expected{{0, 70}, {false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(gathered)->column(0), direct_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(qep::to_table_view(gathered)->column(1), indirect_expected);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, ConcurrentBuildAppendsRetainEveryChunk)
{
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto accumulator  = build_task.initialize(ctx_ref);
  auto first_chunk  = two_int64_columns({1, 2}, {1, 2});
  auto second_chunk = two_int64_columns({3, 4}, {3, 4});
  std::barrier start_build{3};

  auto first  = std::async(std::launch::async, [&] {
    start_build.arrive_and_wait();
    build_task.next(first_chunk, accumulator, ctx_ref);
  });
  auto second = std::async(std::launch::async, [&] {
    start_build.arrive_and_wait();
    build_task.next(second_chunk, accumulator, ctx_ref);
  });
  start_build.arrive_and_wait();
  first.get();
  second.get();
  auto build = build_task.finalize(std::move(accumulator), ctx_ref);

  qep::equality_semi_anti_join_probe_task probe_task{{0}, gqe::join_type_type::left_semi, {}, {}};
  auto output = probe(probe_task, build, two_int64_columns({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}));
  EXPECT_EQ(qep::try_row_count(output), 4);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, FinalizedStateSupportsConcurrentProbes)
{
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(build_task, {two_int64_columns({1, 2, 3}, {10, 20, 30})});
  cudf::get_default_stream().sync();
  qep::equality_semi_anti_join_probe_task probe_task{{0}, gqe::join_type_type::left_semi, {}, {}};
  std::barrier start_probe{3};

  auto run_probe = [&](int64_t key) {
    gqe::query_context stream_context{gqe::make_optimization_parameters(true)};
    auto const stream = stream_context.create_stream("test/stream");
    start_probe.arrive_and_wait();
    auto result = probe(probe_task, build, two_int64_columns({key}, {key * 100}), stream);
    stream.ref().sync();
    return qep::try_row_count(result);
  };
  auto first  = std::async(std::launch::async, run_probe, int64_t{1});
  auto second = std::async(std::launch::async, run_probe, int64_t{3});
  start_probe.arrive_and_wait();
  EXPECT_EQ(first.get(), 1);
  EXPECT_EQ(second.get(), 1);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, RejectsInvalidConstructorConfiguration)
{
  EXPECT_THROW(
    std::ignore = qep::equality_semi_anti_join_build_task({}, cudf::null_equality::UNEQUAL),
    std::invalid_argument);
  EXPECT_THROW(
    std::ignore = qep::equality_semi_anti_join_build_task({-1}, cudf::null_equality::UNEQUAL),
    std::invalid_argument);
  EXPECT_THROW(std::ignore = qep::equality_semi_anti_join_probe_task(
                 {}, gqe::join_type_type::left_semi, {}, {}),
               std::invalid_argument);
  EXPECT_THROW(
    std::ignore = qep::equality_semi_anti_join_probe_task(
      {0}, gqe::join_type_type::left_semi, {-1}, {cudf::data_type{cudf::type_id::INT64}}),
    std::invalid_argument);
  EXPECT_THROW(std::ignore = qep::equality_semi_anti_join_probe_task(
                 {0}, gqe::join_type_type::left_semi, {0}, {}),
               std::invalid_argument);
  for (auto const type : {gqe::join_type_type::inner,
                          gqe::join_type_type::left,
                          gqe::join_type_type::full,
                          gqe::join_type_type::single}) {
    EXPECT_THROW(std::ignore = qep::equality_semi_anti_join_probe_task({0}, type, {}, {}),
                 std::invalid_argument);
  }
  EXPECT_THROW(std::ignore = qep::equality_semi_anti_join_probe_task(
                 {0}, gqe::join_type_type::left_semi, {0}, {cudf::data_type{cudf::type_id::LIST}}),
               std::invalid_argument);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, RejectsInvalidKeyAndProjectionMetadata)
{
  qep::equality_semi_anti_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  qep::equality_semi_anti_join_build_task bad_build_key{{2}, cudf::null_equality::UNEQUAL};
  auto input = two_int64_columns({1}, {100});
  EXPECT_THROW(std::ignore = fold(bad_build_key, {input}), std::invalid_argument);
  auto build = fold(build_task, {two_int64_columns({1}, {10})});
  qep::equality_semi_anti_join_probe_task probe_task{{0}, gqe::join_type_type::left_semi, {}, {}};
  std::vector<std::unique_ptr<cudf::column>> wrong_type;
  wrong_type.push_back(int32_column_wrapper{1}.release());
  wrong_type.push_back(int64_column_wrapper{100}.release());
  EXPECT_THROW(std::ignore = probe(probe_task, build, table_state(std::move(wrong_type))),
               std::logic_error);

  qep::equality_semi_anti_join_probe_task bad_probe_key{
    {2}, gqe::join_type_type::left_semi, {}, {}};
  EXPECT_THROW(std::ignore = probe(bad_probe_key, build, input), std::invalid_argument);

  qep::equality_semi_anti_join_probe_task wrong_key_count{
    {0, 1}, gqe::join_type_type::left_semi, {}, {}};
  EXPECT_THROW(std::ignore = probe(wrong_key_count, build, input), std::logic_error);

  qep::equality_semi_anti_join_probe_task bad_projection{
    {0}, gqe::join_type_type::left_semi, {2}, {cudf::data_type{cudf::type_id::INT64}}};
  EXPECT_THROW(std::ignore = probe(bad_projection, build, input), std::logic_error);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, RejectsInvalidBuildStateShape)
{
  qep::equality_semi_anti_join_probe_task probe_task({0}, gqe::join_type_type::left_semi, {}, {});
  EXPECT_THROW(std::ignore = probe_task.initialize(qep::make_row_count_container(1), ctx_ref),
               std::logic_error);
}

TEST_F(QepEqualitySemiAntiJoinTaskTest, RejectsNestedKeysBeforeBuildMutationOrProbeMaterialization)
{
  for (auto const type : {cudf::type_id::LIST, cudf::type_id::STRUCT}) {
    for (bool const is_empty : {false, true}) {
      std::vector<std::unique_ptr<cudf::column>> columns;
      if (type == cudf::type_id::LIST) {
        columns.push_back(cudf::test::lists_column_wrapper<int64_t>{{1, 2}}.release());
      } else {
        std::vector<std::unique_ptr<cudf::column>> children;
        children.push_back(int64_column_wrapper{1}.release());
        columns.push_back(cudf::test::structs_column_wrapper(std::move(children)).release());
      }
      if (is_empty) { columns.front() = cudf::empty_like(columns.front()->view()); }
      auto nested = table_state(std::move(columns));
      qep::equality_semi_anti_join_build_task build_task({0}, cudf::null_equality::UNEQUAL);
      auto accumulator = build_task.initialize(ctx_ref);
      EXPECT_THROW(build_task.next(nested, accumulator, ctx_ref), std::invalid_argument);
      // Rejected input must not leave a nested chunk in the accumulator.
      build_task.next(two_int64_columns({1}, {10}), accumulator, ctx_ref);
      auto build    = build_task.finalize(std::move(accumulator), ctx_ref);
      auto no_build = fold(build_task, {});
      for (auto const mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
        qep::equality_semi_anti_join_probe_task probe_task({0}, mode, {}, {});
        EXPECT_THROW(std::ignore = probe(probe_task, build, nested), std::invalid_argument);
        EXPECT_THROW(std::ignore = probe(probe_task, no_build, nested), std::invalid_argument);
        EXPECT_EQ(qep::try_row_count(probe(probe_task, build, two_int64_columns({1}, {10}))),
                  mode == gqe::join_type_type::left_semi ? 1 : 0);
      }
    }
  }
}

}  // namespace gqe_test
