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

#include "join_task_test_util.hpp"

#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/qep/operator/conditional_semi_anti_join.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>

#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

using namespace gqe_test;
namespace qep = gqe::qep;

/** @brief Exercises conditional selection, deferred payload, null, and ownership contracts. */
class QepConditionalSemiAntiJoinTaskTest : public QepJoinTaskTestBase {
 protected:
  static auto predicate(cudf::size_type left = 1, cudf::size_type right = 3)
  {
    return std::make_unique<gqe::less_expression>(
      std::make_shared<gqe::column_reference_expression>(left),
      std::make_shared<gqe::column_reference_expression>(right));
  }
  static auto make_probe(gqe::join_type_type mode, bool count = false)
  {
    return qep::conditional_semi_anti_join_probe_task{
      2,
      2,
      predicate(),
      mode,
      count ? std::vector<cudf::size_type>{} : std::vector<cudf::size_type>{1, 0, 1},
      std::vector<cudf::data_type>(count ? 0 : 3, cudf::data_type{cudf::type_id::INT64})};
  }
  void expect_values(qep::state_container const& output,
                     std::vector<int64_t> values,
                     std::vector<int64_t> ids)
  {
    auto gathered = gather(output);
    auto sorted   = cudf::sort(*qep::to_table_view(gathered));
    int64_column_wrapper expected_values(values.begin(), values.end());
    int64_column_wrapper expected_ids(ids.begin(), ids.end());
    cudf::table_view expected{{expected_values, expected_ids, expected_values}};
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
  }
};

TEST_F(QepConditionalSemiAntiJoinTaskTest, DuplicateMatchesAndNullableResiduals)
{
  std::vector<std::unique_ptr<cudf::column>> right;
  right.push_back(int64_column_wrapper{1, 1, 2, 3}.release());
  right.push_back(int64_column_wrapper{{10, 20, 99, 5}, {true, true, false, true}}.release());
  qep::conditional_semi_anti_join_build_task task{2, {1}};
  auto build = fold(task, {table_state(std::move(right))});
  std::vector<std::unique_ptr<cudf::column>> left;
  left.push_back(int64_column_wrapper{1, 2, 3, 4, 5}.release());
  left.push_back(int64_column_wrapper{{1, 2, 30, 4, 0}, {true, true, true, true, false}}.release());
  auto input = table_state(std::move(left));
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    auto output = gather(probe(make_probe(mode), build, input));
    auto sorted = cudf::sort(*qep::to_table_view(output));
    // Gather preserves nullable metadata even when every selected value is valid.
    // Anti keeps both the unmatched value 30 and the NULL left operand.
    auto values   = mode == gqe::join_type_type::left_semi
                      ? int64_column_wrapper{{1, 2, 4}, {true, true, true}}
                      : int64_column_wrapper{{30, 0}, {true, false}};
    auto ids      = mode == gqe::join_type_type::left_semi ? int64_column_wrapper{1, 2, 4}
                                                           : int64_column_wrapper{3, 5};
    auto expected = cudf::sort(cudf::table_view{{values, ids, values}});
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected->view());
    EXPECT_EQ(qep::try_row_count(probe(make_probe(mode, true), build, input)),
              mode == gqe::join_type_type::left_semi ? 3 : 2);
  }
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, NullableStringPredicatesWithSlicedInputs)
{
  // The prefix on the right would incorrectly match "prefix" if its slice offset were ignored.
  string_column_wrapper right_owner{{"prefix", "café", "東京", "absent", "suffix"},
                                    {true, true, true, false, true}};
  string_column_wrapper left_owner{{"prefix", "café", "東京", "absent", "café", "prefix", "suffix"},
                                   {true, true, true, true, false, true, true}};
  int64_column_wrapper id_owner{0, 1, 2, 3, 4, 5, 6};
  auto right =
    qep::state_container_builder()
      .add_state(qep::state_kind::cudf_column_view{cudf::slice(right_owner, {1, 4}).front()})
      .build();
  auto left =
    qep::state_container_builder()
      .add_state(qep::state_kind::cudf_column_view{cudf::slice(id_owner, {1, 6}).front()})
      .add_state(qep::state_kind::cudf_column_view{cudf::slice(left_owner, {1, 6}).front()})
      .build();
  auto build = fold(qep::conditional_semi_anti_join_build_task{1, {0}}, {right});
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    SCOPED_TRACE(mode == gqe::join_type_type::left_semi ? "Semi" : "Anti");
    qep::conditional_semi_anti_join_probe_task task{
      2,
      1,
      std::make_unique<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(1),
        std::make_shared<gqe::column_reference_expression>(2)),
      mode,
      {0},
      {cudf::data_type{cudf::type_id::INT64}}};
    auto output = gather(probe(task, build, left));
    auto sorted = cudf::sort(*qep::to_table_view(output));
    // NULL on either side is not a match, including the NULL left "café" row.
    auto ids = mode == gqe::join_type_type::left_semi ? int64_column_wrapper{1, 2}
                                                      : int64_column_wrapper{3, 4, 5};
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), cudf::table_view{{ids}});
  }
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, NullPredicatesAreNotMatches)
{
  std::vector<std::unique_ptr<cudf::column>> right;
  right.push_back(int64_column_wrapper{0}.release());
  right.push_back(int64_column_wrapper{{10}, {false}}.release());
  qep::conditional_semi_anti_join_build_task task{2, {1}};
  auto build = fold(task, {table_state(std::move(right))});
  auto input = two_int64_columns({1, 2}, {1, 20});
  expect_values(probe(make_probe(gqe::join_type_type::left_semi), build, input), {}, {});
  expect_values(probe(make_probe(gqe::join_type_type::left_anti), build, input), {1, 20}, {1, 2});
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, NoChunksAndSchemaBearingEmptyInputs)
{
  qep::conditional_semi_anti_join_build_task task{2, {1}};
  for (int build_case : {0, 1, 2}) {
    SCOPED_TRACE(build_case);
    std::vector<qep::state_container> chunks;
    if (build_case == 1) { chunks.push_back(two_int64_columns({}, {})); }
    if (build_case == 2) {
      // Physical rows exist, but neither false nor NULL mask entries survive compaction.
      chunks.push_back(masked_two_int64_columns(
        bool_column_wrapper{{false, true}, {true, false}}.release(), {1, 2}, {100, 200}));
    }
    auto build = fold(task, chunks);
    for (bool masked_indirect : {false, true}) {
      SCOPED_TRACE(masked_indirect ? "Masked indirect probe" : "Direct probe");
      qep::state_container input;
      std::vector<int64_t> expected_values;
      std::vector<int64_t> expected_ids;
      if (masked_indirect) {
        // Indirection produces values [20, 10, 30, 20]. The false and null mask entries
        // remove the middle two rows, leaving two references to the source row (id 2, value 20).
        input = with_valid_mask(
          bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(),
          indirect_columns(two_int64_columns({1, 2, 3}, {10, 20, 30}), {1, 0, 2, 1}));
        expected_values = {20, 20};
        expected_ids    = {2, 2};
      } else {
        input           = two_int64_columns({1, 2}, {10, 20});
        expected_values = {10, 20};
        expected_ids    = {1, 2};
      }

      // An empty right side gives semi no matches; anti retains every valid left row.
      expect_values(probe(make_probe(gqe::join_type_type::left_semi), build, input), {}, {});
      expect_values(probe(make_probe(gqe::join_type_type::left_anti), build, input),
                    expected_values,
                    expected_ids);
      auto semi_count = probe(make_probe(gqe::join_type_type::left_semi, true), build, input);
      auto anti_count = probe(make_probe(gqe::join_type_type::left_anti, true), build, input);
      EXPECT_EQ(qep::try_row_count(semi_count), 0) << "No left row can match an empty build.";
      EXPECT_EQ(qep::try_row_count(anti_count), 2) << "Both valid left rows survive the anti join.";
    }
    // An empty probe produces zero rows in either mode, regardless of the build representation.
    for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
      auto empty = probe(make_probe(mode, true), build, two_int64_columns({}, {}));
      EXPECT_EQ(qep::try_row_count(empty), 0);
    }
  }

  auto nonempty_build = fold(task, {two_int64_columns({1}, {10})});
  auto empty_anti     = probe(
    make_probe(gqe::join_type_type::left_anti, true), nonempty_build, two_int64_columns({}, {}));
  EXPECT_EQ(qep::try_row_count(empty_anti), 0)
    << "A nonempty build cannot add rows to an empty probe.";
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, InitializationChecksBuildOperandsExceptForNoChunks)
{
  qep::conditional_semi_anti_join_build_task narrow{2, {0}};
  auto missing       = fold(narrow, {two_int64_columns({99}, {10})});
  auto missing_empty = fold(narrow, {two_int64_columns({}, {})});
  auto absent        = fold(narrow, {});
  auto input         = two_int64_columns({1, 2}, {5, 20});
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    auto task = make_probe(mode);
    EXPECT_THROW((void)task.initialize(missing, ctx_ref), std::invalid_argument);
    EXPECT_THROW((void)task.initialize(missing_empty, ctx_ref), std::invalid_argument);
    EXPECT_NO_THROW((void)task.initialize(absent, ctx_ref));
    expect_values(
      probe(task, absent, input),
      mode == gqe::join_type_type::left_semi ? std::vector<int64_t>{} : std::vector<int64_t>{5, 20},
      mode == gqe::join_type_type::left_semi ? std::vector<int64_t>{} : std::vector<int64_t>{1, 2});

    // Extra operands and their physical ordering do not change logical predicate coordinates.
    for (auto columns :
         {std::vector<cudf::size_type>{0, 1}, std::vector<cudf::size_type>{1, 0, 1}}) {
      qep::conditional_semi_anti_join_build_task complete{2, columns};
      auto clone = complete.clone();
      auto build = fold(static_cast<qep::conditional_semi_anti_join_build_task const&>(*clone),
                        {two_int64_columns({99}, {10})});
      expect_values(
        probe(task, build, input),
        mode == gqe::join_type_type::left_semi ? std::vector<int64_t>{5} : std::vector<int64_t>{20},
        mode == gqe::join_type_type::left_semi ? std::vector<int64_t>{1} : std::vector<int64_t>{2});
    }
  }
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, MaskedIndirectPayloadsShareSelectionAndSurviveInputs)
{
  qep::conditional_semi_anti_join_build_task task{2, {1}};
  auto build = fold(task, {two_int64_columns({1}, {10}), two_int64_columns({2, 2}, {20, 30})});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(int64_column_wrapper{1, 2, 3, 2}.release());
  columns.push_back(int64_column_wrapper{1, 2, 3, 4}.release());
  columns.push_back(string_column_wrapper{"café", "東京", "skip", "four"}.release());
  auto input  = indirect_columns(table_state(std::move(columns)), {3, 0, 2, 1, 0, 2});
  auto source = std::get<qep::state_kind::indirect_column>(*input[2])
                  .state()
                  ->gather_source()
                  ->relation_source();
  input = with_valid_mask(bool_column_wrapper{{true, true, true, true, true, false},
                                              {true, true, false, true, true, true}}
                            .release(),
                          std::move(input));
  qep::conditional_semi_anti_join_probe_task probe_task{3,
                                                        2,
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
  // Repeated logical references to café survive independently and share deferred payload.
  string_column_wrapper expected_strings{"café", "café", "four", "東京"};
  int64_column_wrapper expected_values{1, 1, 4, 2};
  cudf::table_view expected{{expected_strings, expected_values, expected_strings}};
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted->view(), expected);
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, MaskedBuildAndProbeKeepOperandAlignment)
{
  qep::conditional_semi_anti_join_build_task task{2, {1}};
  auto build = fold(task,
                    {masked_two_int64_columns(
                      bool_column_wrapper{false, true, true}.release(), {1, 2, 3}, {100, 20, 1})});
  auto input = masked_two_int64_columns(
    bool_column_wrapper{true, false, true, true}.release(), {1, 9, 2, 3}, {1, 9, 2, 50});
  // 50 would match the masked-out right value 100 if build compaction ignored its mask.
  expect_values(probe(make_probe(gqe::join_type_type::left_semi), build, input), {1, 2}, {1, 2});
  expect_values(probe(make_probe(gqe::join_type_type::left_anti), build, input), {50}, {3});
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, LiteralPredicateAndClonesOwnTheirStateAcrossProbes)
{
  auto cloned_build = qep::conditional_semi_anti_join_build_task{2, {1}}.clone();
  auto build = fold(static_cast<qep::conditional_semi_anti_join_build_task const&>(*cloned_build),
                    {two_int64_columns({1}, {99})});
  auto residual =
    std::make_unique<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::literal_expression<int64_t>>(10));
  // Temporary originals are destroyed before either clone is exercised.
  auto clone =
    qep::conditional_semi_anti_join_probe_task{
      2,
      2,
      std::make_unique<gqe::logical_and_expression>(std::move(residual), predicate()),
      gqe::join_type_type::left_semi,
      {1, 0, 1},
      std::vector<cudf::data_type>(3, cudf::data_type{cudf::type_id::INT64})}
      .clone();
  auto const& copied = static_cast<qep::conditional_semi_anti_join_probe_task const&>(*clone);
  expect_values(probe(copied, build, two_int64_columns({1, 1}, {5, 20})), {5}, {1});
  expect_values(probe(copied, build, two_int64_columns({1, 2}, {3, 4})), {3, 4}, {1, 2});
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, InvalidMetadataIsRejectedBeforeExecution)
{
  EXPECT_THROW((qep::conditional_semi_anti_join_build_task{0, {}}), std::invalid_argument);
  EXPECT_THROW((qep::conditional_semi_anti_join_build_task{2, {2}}), std::invalid_argument);
  EXPECT_THROW((qep::conditional_semi_anti_join_probe_task{
                 2, 2, predicate(), gqe::join_type_type::inner, {}, {}}),
               std::invalid_argument);
  EXPECT_THROW((qep::conditional_semi_anti_join_probe_task{
                 2, 2, predicate(1, 4), gqe::join_type_type::left_semi, {}, {}}),
               std::invalid_argument);
  EXPECT_THROW(
    (qep::conditional_semi_anti_join_probe_task{2,
                                                2,
                                                predicate(),
                                                gqe::join_type_type::left_semi,
                                                {2},
                                                {cudf::data_type{cudf::type_id::INT64}}}),
    std::invalid_argument);
}

TEST_F(QepConditionalSemiAntiJoinTaskTest, PredicatesMustReferenceBothInputs)
{
  EXPECT_THROW((qep::conditional_semi_anti_join_build_task{2, {}}), std::invalid_argument);
  for (auto mode : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    for (int side : {-1, 0, 1}) {
      std::shared_ptr<gqe::expression> operand =
        side < 0
          ? std::shared_ptr<gqe::expression>{std::make_shared<gqe::literal_expression<int64_t>>(1)}
          : std::shared_ptr<gqe::expression>{
              std::make_shared<gqe::column_reference_expression>(side * 2)};
      auto residual = std::make_unique<gqe::less_expression>(
        operand, std::make_shared<gqe::literal_expression<int64_t>>(2));
      EXPECT_THROW(
        (qep::conditional_semi_anti_join_probe_task{2, 2, std::move(residual), mode, {}, {}}),
        std::invalid_argument);
    }
  }
}
