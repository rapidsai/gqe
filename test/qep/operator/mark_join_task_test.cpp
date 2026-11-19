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
#include <gqe/qep/operator/mark_join.hpp>
#include <gqe/qep/shapes/indirect_table.hpp>
#include <gqe/qep/shapes/row_count.hpp>

#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <algorithm>
#include <future>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace gqe_test {
namespace qep = gqe::qep;

/** @brief Provides build, probe, completion, and scan helpers for mark-join tasks. */
class QepMarkJoinTaskTest : public QepJoinTaskTestBase {
 protected:
  qep::state_container scan(qep::state_container const& build,
                            qep::state_container complete,
                            gqe::join_type_type type,
                            bool is_count_only                        = false,
                            std::vector<cudf::size_type> projections  = {1, 1},
                            std::vector<cudf::data_type> output_types = {
                              cudf::data_type{cudf::type_id::INT64},
                              cudf::data_type{cudf::type_id::INT64}})
  {
    auto input = build;
    input.insert(input.end(), complete.begin(), complete.end());
    qep::mark_join_scan_task task{
      type,
      is_count_only ? std::vector<cudf::size_type>{} : std::move(projections),
      is_count_only ? std::vector<cudf::data_type>{} : std::move(output_types)};
    auto iterator = task.initialize(input, ctx_ref);
    std::vector<qep::state_container> chunks;
    while (auto result = task.next(iterator, ctx_ref)) {
      chunks.push_back(gather(*result));
    }
    EXPECT_FALSE(chunks.empty());
    task.finalize(std::move(iterator), ctx_ref);
    if (is_count_only) {
      EXPECT_EQ(chunks.size(), 1);
      return std::move(chunks.front());
    }
    std::vector<cudf::table_view> tables;
    for (auto const& chunk : chunks) {
      tables.push_back(*qep::to_table_view(chunk));
    }
    return qep::state_container_builder()
      .add_state(std::move(*cudf::concatenate(tables)), gqe::query_stream::default_stream())
      .build();
  }

  qep::state_container run(qep::state_container const& build,
                           std::vector<qep::state_container> const& chunks,
                           gqe::join_type_type type,
                           bool is_count_only                         = false,
                           std::vector<cudf::size_type> keys          = {0},
                           std::unique_ptr<gqe::expression> predicate = nullptr,
                           std::vector<cudf::size_type> projections   = {1, 1},
                           std::vector<cudf::data_type> output_types  = {
                             cudf::data_type{cudf::type_id::INT64},
                             cudf::data_type{cudf::type_id::INT64}})
  {
    qep::mark_join_probe_task probe{std::move(keys), std::move(predicate)};
    qep::mark_join_complete_task barrier;
    auto state    = probe.initialize(build, ctx_ref);
    auto complete = barrier.initialize(ctx_ref);
    for (auto const& chunk : chunks) {
      auto token = probe.next(chunk, state, ctx_ref);
      EXPECT_TRUE(token.has_value());
      barrier.next(*token, complete, ctx_ref);
    }
    probe.finalize(std::move(state), ctx_ref);
    return scan(build,
                barrier.finalize(std::move(complete), ctx_ref),
                type,
                is_count_only,
                std::move(projections),
                std::move(output_types));
  }

  void expect_values(qep::state_container const& result, std::initializer_list<int64_t> values)
  {
    auto table = qep::to_table_view(result);
    ASSERT_TRUE(table.has_value());
    auto sorted = cudf::sort(*table);
    int64_column_wrapper expected(values);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->view().column(0), expected);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->view().column(1), expected);
  }
};

/**
 * @brief Execute logical semi/anti cases by building left, probing right, completing, and scanning.
 */
class MarkJoinTestAdapter : public QepMarkJoinTaskTest {
 public:
  qep::state_container execute(qep::state_container const& left,
                               qep::state_container const& right,
                               SemiAntiJoinTestOptions const& options,
                               std::unique_ptr<gqe::expression> residual = nullptr)
  {
    qep::mark_join_build_task build_task{
      options.keys, options.null_equality, options.predicate_columns};
    auto build = this->fold(build_task, {left});
    return this->run(build,
                     {right},
                     options.join_type,
                     options.projections.empty(),
                     options.keys,
                     std::move(residual),
                     options.projections,
                     options.output_types);
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Mark, QepSemiAntiJoinContractTest, MarkJoinTestAdapter);
INSTANTIATE_TYPED_TEST_SUITE_P(Mark, QepSemiAntiJoinResidualTest, MarkJoinTestAdapter);

TEST_F(QepMarkJoinTaskTest, EmptyProbeStreamAndAbsentBuildChunks)
{
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {two_int64_columns({1, 2}, {10, 20})});
  expect_values(run(build, {}, gqe::join_type_type::left_anti), {10, 20});
  expect_values(run(build, {}, gqe::join_type_type::left_semi), {});
  auto empty = fold(task, {});
  expect_values(run(empty, {}, gqe::join_type_type::left_anti), {});
  EXPECT_EQ(qep::try_row_count(run(empty, {}, gqe::join_type_type::left_semi, true)), 0);
}

TEST_F(QepMarkJoinTaskTest, SharedBuildDoesNotShareMarks)
{
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {two_int64_columns({1, 2}, {10, 20})});
  expect_values(run(build, {two_int64_columns({1}, {0})}, gqe::join_type_type::left_semi), {10});
  expect_values(run(build, {two_int64_columns({2}, {0})}, gqe::join_type_type::left_semi), {20});
  expect_values(run(build, {}, gqe::join_type_type::left_anti), {10, 20});
}

TEST_F(QepMarkJoinTaskTest, SlicedNullableProbeKeys)
{
  for (auto nulls : {cudf::null_equality::UNEQUAL, cudf::null_equality::EQUAL}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{{1, 9, 2}, {true, false, true}}.release());
    columns.push_back(int64_column_wrapper{10, 20, 30}.release());
    qep::mark_join_build_task task{{0}, nulls};
    auto build = fold(task, {table_state(std::move(columns))});

    int64_column_wrapper keys{{2, 0, 1, 2}, {true, false, true, true}};
    int64_column_wrapper payload{0, 0, 0, 0};
    auto probe = qep::state_container_builder()
                   .add_state(cudf::table_view{
                     {cudf::slice(keys, {1, 3}).front(), cudf::slice(payload, {1, 3}).front()}})
                   .build();
    auto semi = run(build, {probe}, gqe::join_type_type::left_semi);
    auto anti = run(build, {probe}, gqe::join_type_type::left_anti);
    if (nulls == cudf::null_equality::EQUAL) {
      expect_values(semi, {10, 20});
      expect_values(anti, {30});
    } else {
      expect_values(semi, {10});
      expect_values(anti, {20, 30});
    }
  }
}

TEST_F(QepMarkJoinTaskTest, ConcurrentProbeStreamsAndCompletionDeduplication)
{
  auto const first  = get_query_ctx()->create_stream("test/first");
  auto const second = get_query_ctx()->create_stream("test/second");
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto chunk = two_int64_columns({1, 2, 3}, {10, 20, 30});
  cudf::get_default_stream().sync();
  auto build = fold(task, {chunk}, first);
  qep::mark_join_probe_task probe{{0}};
  auto state = probe.initialize(build, ctx_ref, first);
  qep::mark_join_complete_task barrier;
  auto complete = barrier.initialize(ctx_ref);
  auto a        = two_int64_columns({1, 1}, {0, 0});
  auto b        = two_int64_columns({1, 3, 3}, {0, 0, 0});
  first.ref().sync();
  cudf::get_default_stream().sync();
  auto worker = [&](auto const& input, gqe::query_stream const& stream) {
    auto token = probe.next(input, state, ctx_ref, stream);
    barrier.next(*token, complete, ctx_ref, stream);
    barrier.next(*token, complete, ctx_ref, stream);
    stream.ref().sync();
  };
  auto future = std::async(std::launch::async, [&] { worker(a, first); });
  worker(b, second);
  future.get();
  probe.finalize(std::move(state), ctx_ref);
  expect_values(
    scan(build, barrier.finalize(std::move(complete), ctx_ref), gqe::join_type_type::left_semi),
    {10, 30});
}

TEST_F(QepMarkJoinTaskTest, RejectsInvalidConfiguration)
{
  EXPECT_THROW((qep::mark_join_build_task{{}, cudf::null_equality::UNEQUAL}),
               std::invalid_argument);
  EXPECT_THROW((qep::mark_join_probe_task{{-1}}), std::invalid_argument);
  EXPECT_THROW((qep::mark_join_scan_task{gqe::join_type_type::inner, {}, {}}),
               std::invalid_argument);
}

TEST_F(QepMarkJoinTaskTest, RejectsInvalidResidualReferencesAndTypes)
{
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {two_int64_columns({1}, {10})});
  std::vector<qep::state_container> right{two_int64_columns({1}, {20})};
  for (auto const index : {-1, 4}) {
    EXPECT_THROW(std::ignore = run(build,
                                   right,
                                   gqe::join_type_type::left_semi,
                                   false,
                                   {0},
                                   std::make_unique<gqe::equal_expression>(
                                     std::make_shared<gqe::column_reference_expression>(index),
                                     std::make_shared<gqe::column_reference_expression>(0))),
                 std::invalid_argument);
  }
  EXPECT_THROW(std::ignore = run(build,
                                 right,
                                 gqe::join_type_type::left_semi,
                                 false,
                                 {0},
                                 std::make_unique<gqe::column_reference_expression>(1)),
               std::invalid_argument);
}

TEST_F(QepMarkJoinTaskTest, UnionsIndependentProbeAccumulators)
{
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {two_int64_columns({1, 2, 3}, {10, 20, 30})});
  qep::mark_join_probe_task probe{{0}};
  qep::mark_join_complete_task barrier;
  auto complete = barrier.initialize(ctx_ref);
  // Key 1 is matched by both independent maps and must only contribute once.
  for (auto const& chunk : {two_int64_columns({1, 1}, {0, 0}), two_int64_columns({1, 2}, {0, 0})}) {
    auto accumulator = probe.initialize(build, ctx_ref);
    auto token       = probe.next(chunk, accumulator, ctx_ref);
    barrier.next(*token, complete, ctx_ref);
    probe.finalize(std::move(accumulator), ctx_ref);
  }
  auto completed = barrier.finalize(std::move(complete), ctx_ref);
  expect_values(scan(build, completed, gqe::join_type_type::left_semi), {10, 20});
  expect_values(scan(build, completed, gqe::join_type_type::left_anti), {30});
  EXPECT_EQ(qep::try_row_count(scan(build, completed, gqe::join_type_type::left_semi, true)), 2);
  EXPECT_EQ(qep::try_row_count(scan(build, completed, gqe::join_type_type::left_anti, true)), 1);
}

TEST_F(QepMarkJoinTaskTest, RetainedStorageUsesSuppliedMemoryResource)
{
  auto const stream = get_query_ctx()->create_stream("test/stream");
  rmm::mr::tracking_resource_adaptor mr{cudf::get_current_device_resource_ref()};
  {
    qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
    auto left  = two_int64_columns({1, 2}, {10, 20});
    auto right = two_int64_columns({1}, {0});
    cudf::get_default_stream().sync();
    auto build = fold(task, {left}, stream, mr);
    qep::mark_join_probe_task probe{{0}};
    auto state = probe.initialize(build, ctx_ref, stream, mr);
    EXPECT_FALSE(mr.get_outstanding_allocations().empty());
    auto token = probe.next(right, state, ctx_ref, stream, mr);
    qep::mark_join_complete_task barrier;
    auto complete = barrier.initialize(ctx_ref, stream, mr);
    barrier.next(*token, complete, ctx_ref, stream, mr);
    probe.finalize(std::move(state), ctx_ref, stream, mr);
    auto done  = barrier.finalize(std::move(complete), ctx_ref, stream, mr);
    auto input = build;
    input.insert(input.end(), done.begin(), done.end());
    qep::mark_join_scan_task scan{
      gqe::join_type_type::left_semi, {1}, {cudf::data_type{cudf::type_id::INT64}}};
    auto iterator      = scan.initialize(input, ctx_ref, stream, mr);
    auto output        = scan.next(iterator, ctx_ref, stream, mr);
    auto const& column = *std::get<qep::state_kind::indirect_column>(*(*output)[0]).state();
    EXPECT_TRUE(mr.get_outstanding_allocations().contains(
      const_cast<void*>(column.gather_source()->position_list()->column().view().head())));
    scan.finalize(std::move(iterator), ctx_ref, stream, mr);
  }
  stream.ref().sync();
  EXPECT_TRUE(mr.get_outstanding_allocations().empty());
}

TEST_F(QepMarkJoinTaskTest, ScanAllocatesOutputsOnNextAndRetainsInputs)
{
  auto const init_stream = get_query_ctx()->create_stream("test/init_stream");
  auto const next_stream = get_query_ctx()->create_stream("test/next_stream");
  rmm::mr::tracking_resource_adaptor init_mr{cudf::get_current_device_resource_ref()};
  rmm::mr::tracking_resource_adaptor next_mr{cudf::get_current_device_resource_ref()};
  qep::mark_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
  auto build =
    fold(build_task,
         {with_valid_mask(bool_column_wrapper{true, false}.release(),
                          indirect_columns(two_int64_columns({1, 2, 3}, {10, 20, 30}), {2, 0})),
          indirect_columns(two_int64_columns({4, 5}, {40, 50}), {1, 0})});
  qep::mark_join_complete_task complete_task;
  auto complete = fold(complete_task, {});
  auto input    = build;
  input.insert(input.end(), complete.begin(), complete.end());
  cudf::get_default_stream().sync();
  qep::mark_join_scan_task scan_task{
    gqe::join_type_type::left_anti,
    {1, 1},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  auto iterator = scan_task.initialize(input, ctx_ref, init_stream, init_mr);
  init_stream.ref().sync();
  input.clear();
  build.clear();
  complete.clear();

  for (int chunk_index = 0; chunk_index < 2; ++chunk_index) {
    auto output = scan_task.next(iterator, ctx_ref, next_stream, next_mr);
    ASSERT_TRUE(output.has_value());
    auto const* column = qep::try_get_indirect_column((*output)[0]);
    ASSERT_NE(column, nullptr);
    auto* positions =
      const_cast<void*>(column->gather_source()->position_list()->column().view().head());
    EXPECT_TRUE(next_mr.get_outstanding_allocations().contains(positions));
    EXPECT_FALSE(init_mr.get_outstanding_allocations().contains(positions));
    next_stream.ref().sync();
    if (chunk_index == 0) {
      expect_values(gather(*output), {30});
    } else {
      expect_values(gather(*output), {40, 50});
    }
    output.reset();
    EXPECT_TRUE(next_mr.get_outstanding_allocations().empty());
  }
  EXPECT_FALSE(scan_task.next(iterator, ctx_ref, next_stream, next_mr).has_value());
  scan_task.finalize(std::move(iterator), ctx_ref, init_stream, init_mr);
  next_stream.ref().sync();
  init_stream.ref().sync();
  EXPECT_TRUE(init_mr.get_outstanding_allocations().empty());
}

TEST_F(QepMarkJoinTaskTest, SplitScanPreservesEachSourceAndHandlesUnusedSlots)
{
  for (auto type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    qep::mark_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL};
    auto build =
      fold(build_task,
           {masked_two_int64_columns(bool_column_wrapper{true, false}.release(), {1, 2}, {10, 20}),
            two_int64_columns({}, {}),
            indirect_columns(two_int64_columns({3, 4}, {30, 40}), {1, 0})});
    qep::mark_join_probe_task probe_task{{0}};
    auto probe_state = probe_task.initialize(build, ctx_ref);
    auto token       = probe_task.next(two_int64_columns({1, 4}, {0, 0}), probe_state, ctx_ref);
    ASSERT_TRUE(token.has_value());
    qep::mark_join_complete_task complete_task;
    auto complete = fold(complete_task, {*token});
    probe_task.finalize(std::move(probe_state), ctx_ref);
    auto input = build;
    input.insert(input.end(), complete.begin(), complete.end());
    qep::mark_join_scan_task scan_task{
      type, {1, 1}, std::vector<cudf::data_type>(2, cudf::data_type{cudf::type_id::INT64})};
    EXPECT_EQ(scan_task.required_split_count_from_fold_input(), 0u);
    EXPECT_THROW(std::ignore = scan_task.split(-1, 4), std::invalid_argument);
    EXPECT_THROW(std::ignore = scan_task.split(0, 0), std::invalid_argument);
    // Projected scans require the fold's input-adapter count, not any count below max_splits().
    auto too_few = scan_task.split(0, 2);
    EXPECT_THROW(std::ignore = too_few->initialize(input, ctx_ref), std::logic_error);

    // Model four input adapters with an early exit leaving three retained chunks, one empty.
    for (cudf::size_type i = 0; i < 4; ++i) {
      auto split    = scan_task.split(i, 4);
      auto cloned   = qep::clone_qep_task_as<qep::iterate_task>(*split);
      auto iterator = cloned->initialize(input, ctx_ref);
      auto output   = cloned->next(iterator, ctx_ref);
      ASSERT_TRUE(output.has_value());
      if (i < 3) { EXPECT_NE(qep::try_get_indirect_column((*output)[0]), nullptr); }
      EXPECT_FALSE(cloned->next(iterator, ctx_ref).has_value());
      cloned->finalize(std::move(iterator), ctx_ref);
      auto table = gather(*output);
      if (i == 0 && type == gqe::join_type_type::left_semi) {
        expect_values(table, {10});
      } else if (i == 2) {
        expect_values(table,
                      type == gqe::join_type_type::left_semi ? std::initializer_list<int64_t>{40}
                                                             : std::initializer_list<int64_t>{30});
      } else {
        expect_values(table, {});
      }
    }

    qep::mark_join_scan_task count_scan{type, {}, {}};
    EXPECT_FALSE(count_scan.required_split_count_from_fold_input().has_value());
    EXPECT_EQ(count_scan.max_splits(), 1);
    EXPECT_THROW(std::ignore = count_scan.split(0, 2), std::invalid_argument);
    auto split    = count_scan.split(0, 1);
    auto iterator = split->initialize(input, ctx_ref);
    auto output   = split->next(iterator, ctx_ref);
    ASSERT_TRUE(output.has_value());
    EXPECT_EQ(qep::try_row_count(*output), type == gqe::join_type_type::left_semi ? 2 : 1);
    EXPECT_FALSE(split->next(iterator, ctx_ref).has_value());
    split->finalize(std::move(iterator), ctx_ref);
  }
}

TEST_F(QepMarkJoinTaskTest, RejectsResidualScratchExceedingBlockLimit)
{
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL, {1}};
  auto build = fold(task, {two_int64_columns({1, 1, 2}, {10, 30, 40})});
  // A balanced expression keeps enough intermediates live to exceed the scratch budget of a
  // 1024-thread block. Reject it before attempting an invalid kernel launch.
  auto make_sum = [](auto&& self, int depth) -> std::shared_ptr<gqe::expression> {
    if (depth == 0) { return std::make_shared<gqe::column_reference_expression>(1); }
    return std::make_shared<gqe::add_expression>(self(self, depth - 1), self(self, depth - 1));
  };
  auto predicate = std::make_unique<gqe::greater_expression>(
    make_sum(make_sum, 8), std::make_shared<gqe::literal_expression<int64_t>>(256 * 20));
  EXPECT_THROW(run(build,
                   {two_int64_columns({1, 2}, {0, 0})},
                   gqe::join_type_type::left_semi,
                   false,
                   {0},
                   std::move(predicate)),
               std::invalid_argument);
}

TEST_F(QepMarkJoinTaskTest, MultiBlockDuplicatesAndSlicedProbe)
{
  std::vector<int64_t> keys(131075);
  std::vector<int64_t> payload(keys.size());
  for (std::size_t i = 0; i < keys.size(); ++i) {
    keys[i]    = static_cast<int64_t>(i % 17);
    payload[i] = static_cast<int64_t>(i);
  }
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {two_int64_columns(keys, payload)});
  int64_column_wrapper right_keys{99, 3, 5, 99};
  int64_column_wrapper right_payload{0, 0, 0, 0};
  auto right = qep::state_container_builder()
                 .add_state(cudf::table_view{{cudf::slice(right_keys, {1, 3}).front(),
                                              cudf::slice(right_payload, {1, 3}).front()}})
                 .build();
  auto result   = run(build, {right}, gqe::join_type_type::left_semi, true);
  auto expected = std::ranges::count(keys, int64_t{3}) + std::ranges::count(keys, int64_t{5});
  EXPECT_EQ(qep::try_row_count(result), expected);
  for (auto type : {gqe::join_type_type::left_semi, gqe::join_type_type::left_anti}) {
    auto actual = run(build, {right}, type);
    std::vector<int64_t> expected_payload;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      bool const is_match = keys[i] == 3 || keys[i] == 5;
      if (is_match == (type == gqe::join_type_type::left_semi)) {
        expected_payload.push_back(payload[i]);
      }
    }
    int64_column_wrapper expected_column(expected_payload.begin(), expected_payload.end());
    auto sorted_actual = cudf::sort(qep::to_table_view(actual)->select({0}));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_actual->view().column(0), expected_column);
  }
}

TEST_F(QepMarkJoinTaskTest, NativeMasksPreserveRowsAcrossMixedBuildChunks)
{
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build =
    fold(task,
         {masked_two_int64_columns(
            bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(),
            {1, 2, 3, 4},
            {10, 20, 30, 40}),
          two_int64_columns({1, 5}, {11, 50}),
          two_int64_columns({}, {})});
  auto right = masked_two_int64_columns(
    bool_column_wrapper{{true, true, true, false}, {true, true, false, true}}.release(),
    {1, 2, 4, 5},
    {0, 0, 0, 0});
  expect_values(run(build, {right}, gqe::join_type_type::left_semi), {10, 11});
  expect_values(run(build, {right}, gqe::join_type_type::left_anti), {40, 50});
  expect_values(run(build, {}, gqe::join_type_type::left_anti), {10, 11, 40, 50});
  EXPECT_EQ(qep::try_row_count(run(build, {right}, gqe::join_type_type::left_anti, true)), 2);
}

TEST_F(QepMarkJoinTaskTest, InactiveNullKeysNeverParticipateUnderEitherNullPolicy)
{
  for (auto nulls : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column_wrapper{{0, 0, 0, 1}, {false, false, false, true}}.release());
    columns.push_back(int64_column_wrapper{10, 20, 30, 40}.release());
    auto left = with_valid_mask(
      bool_column_wrapper{{true, false, true, true}, {true, true, false, true}}.release(),
      table_state(std::move(columns)));
    qep::mark_join_build_task task{{0}, nulls};
    auto build = fold(task, {left});
    std::vector<std::unique_ptr<cudf::column>> right_columns;
    right_columns.push_back(int64_column_wrapper{{0, 1}, {false, true}}.release());
    right_columns.push_back(int64_column_wrapper{0, 0}.release());
    auto right = with_valid_mask(bool_column_wrapper{true, false}.release(),
                                 table_state(std::move(right_columns)));
    if (nulls == cudf::null_equality::EQUAL) {
      expect_values(run(build, {right}, gqe::join_type_type::left_semi), {10});
      expect_values(run(build, {right}, gqe::join_type_type::left_anti), {40});
    } else {
      expect_values(run(build, {right}, gqe::join_type_type::left_semi), {});
      expect_values(run(build, {right}, gqe::join_type_type::left_anti), {10, 40});
    }
    auto inactive = fold(
      task,
      {masked_two_int64_columns(bool_column_wrapper{false, false}.release(), {1, 2}, {10, 20})});
    expect_values(run(inactive, {}, gqe::join_type_type::left_anti), {});
    expect_values(run(inactive, {right}, gqe::join_type_type::left_semi), {});
  }
}

TEST_F(QepMarkJoinTaskTest, SlicedNullableMasksUseTheirOwnOffsets)
{
  bool_column_wrapper mask{{false, true, false, true, true, true},
                           {true, true, true, false, true, true}};
  auto mask_slice = cudf::slice(mask, {1, 5}).front();
  auto left       = two_int64_columns({1, 2, 3, 4}, {10, 20, 30, 40});
  left.insert(left.begin(), qep::make_shared_state(qep::state_kind::valid_mask_view{mask_slice}));
  qep::mark_join_build_task task{{0}, cudf::null_equality::UNEQUAL};
  auto build = fold(task, {left});
  auto right = two_int64_columns({4, 1, 2, 3}, {0, 0, 0, 0});
  right.insert(right.begin(), qep::make_shared_state(qep::state_kind::valid_mask_view{mask_slice}));
  expect_values(run(build, {right}, gqe::join_type_type::left_semi), {40});
  expect_values(run(build, {right}, gqe::join_type_type::left_anti), {10});
}

TEST_F(QepMarkJoinTaskTest, MaskedIndirectInputsRetainPayloadSources)
{
  auto left   = indirect_columns(two_int64_columns({1, 2, 3}, {10, 20, 30}), {2, 0, 1, 0});
  auto source = qep::try_get_indirect_column(left[1])->gather_source()->relation_source();
  left = with_valid_mask(bool_column_wrapper{true, false, true, true}.release(), std::move(left));
  auto right =
    with_valid_mask(bool_column_wrapper{true, false, true}.release(),
                    indirect_columns(two_int64_columns({1, 2, 3}, {5, 20, 35}), {2, 1, 0}));
  qep::mark_join_build_task build_task{{0}, cudf::null_equality::UNEQUAL, {1}};
  auto build     = fold(build_task, {left});
  auto predicate = std::make_unique<gqe::greater_expression>(
    std::make_shared<gqe::column_reference_expression>(1),
    std::make_shared<gqe::column_reference_expression>(3));
  qep::mark_join_probe_task probe_task{{0}, std::move(predicate)};
  auto probe_state = probe_task.initialize(build, ctx_ref);
  auto token       = probe_task.next(right, probe_state, ctx_ref);
  qep::mark_join_complete_task complete_task;
  auto complete = fold(complete_task, {*token});
  auto input    = build;
  input.insert(input.end(), complete.begin(), complete.end());
  qep::mark_join_scan_task scan_task{
    gqe::join_type_type::left_semi,
    {1, 1},
    {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::INT64}}};
  auto iterator = scan_task.initialize(input, ctx_ref);
  auto output   = scan_task.next(iterator, ctx_ref);
  ASSERT_TRUE(output.has_value());
  auto const* column = qep::try_get_indirect_column((*output)[0]);
  ASSERT_NE(column, nullptr);
  EXPECT_EQ(column->gather_source()->relation_source(), source);
  EXPECT_EQ(column->gather_source(), qep::try_get_indirect_column((*output)[1])->gather_source());
  EXPECT_FALSE(scan_task.next(iterator, ctx_ref).has_value());
  scan_task.finalize(std::move(iterator), ctx_ref);
  probe_task.finalize(std::move(probe_state), ctx_ref);
  left.clear();
  right.clear();
  input.clear();
  build.clear();
  complete.clear();
  token.reset();
  expect_values(gather(*output), {10});
}

}  // namespace gqe_test
