/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

enum class cache_strategy { recompute, use_cache };

/*
 * This test suite constructs the input tables with handpicked values. Both the left and the right
 * table have 2 columns, 1 key column and 1 payload column.
 */
class SingleKeyColumnJoinTest : public ::testing::Test {
 protected:
  SingleKeyColumnJoinTest()
    : task_manager_ctx{},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }
  void construct_join_task(gqe::join_type_type join_type,
                           std::vector<cudf::size_type> projection_indices,
                           cache_strategy strategy)
  {
    int64_column_wrapper left_key({2, 1, 1, 3, 4, 1});
    int64_column_wrapper left_payload({0, 1, 2, 3, 4, 5});
    int64_column_wrapper right_key({3, 1, 5, 1, 2});
    int64_column_wrapper right_payload({0, 1, 2, 3, 4});

    std::vector<std::unique_ptr<cudf::column>> left_table_columns;
    left_table_columns.push_back(left_key.release());
    left_table_columns.push_back(left_payload.release());
    auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));

    std::vector<std::unique_ptr<cudf::column>> right_table_columns;
    right_table_columns.push_back(right_key.release());
    right_table_columns.push_back(right_payload.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));

    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;

    auto left_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table));
    auto right_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, right_task_id, stage_id, std::move(right_table));
    auto join_condition = std::make_unique<gqe::equal_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2));

    // this mimics separate_materialization in task_graph.cpp
    const bool late_materialize =
      join_type == gqe::join_type_type::left_semi || join_type == gqe::join_type_type::left_anti;
    std::shared_ptr<gqe::join_hash_map_cache> hash_map_cache = nullptr;
    if (strategy == cache_strategy::use_cache) {
      if (join_type == gqe::join_type_type::left_semi ||
          join_type == gqe::join_type_type::left_anti) {
        hash_map_cache = std::make_shared<gqe::join_hash_map_cache>(
          gqe::join_hash_map_cache::build_location::left);
      } else {
        hash_map_cache = std::make_shared<gqe::join_hash_map_cache>(
          gqe::join_hash_map_cache::build_location::right);
      }
    }

    join_task = std::make_shared<gqe::join_task>(ctx_ref,
                                                 join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 projection_indices,
                                                 hash_map_cache,
                                                 !late_materialize);

    if (late_materialize) {
      std::vector<std::shared_ptr<gqe::task>> tasks{{join_task}};
      late_materialization = std::make_shared<gqe::materialize_join_from_position_lists_task>(
        ctx_ref,
        join_task_id,
        stage_id,
        std::move(left_task),  // left table
        std::move(tasks),      // positions lists
        join_type,             // join type
        projection_indices,    // projection indices
        hash_map_cache,        // hash map
        true);                 // use mark join
    }
  }

  static void verify_inner_join(std::optional<cudf::table_view> join_result)
  {
    ASSERT_EQ(join_result.has_value(), true);
    auto join_result_sorted = cudf::sort(*join_result);

    int64_column_wrapper ref_result_col0({1, 1, 1, 1, 1, 1, 2, 3});
    int64_column_wrapper ref_result_col1({1, 1, 2, 2, 5, 5, 0, 3});
    int64_column_wrapper ref_result_col2({1, 3, 1, 3, 1, 3, 4, 0});

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    ref_result_columns.push_back(ref_result_col0.release());
    ref_result_columns.push_back(ref_result_col1.release());
    ref_result_columns.push_back(ref_result_col2.release());

    auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
  }

  static void verify_left_join(std::optional<cudf::table_view> join_result)
  {
    ASSERT_EQ(join_result.has_value(), true);
    auto join_result_sorted = cudf::sort(*join_result);

    int64_column_wrapper ref_result_col0({1, 1, 1, 1, 1, 1, 2, 3, 4});
    int64_column_wrapper ref_result_col1({1, 1, 2, 2, 5, 5, 0, 3, 4});
    int64_column_wrapper ref_result_col2({1, 3, 1, 3, 1, 3, 4, 0, 0},
                                         {true, true, true, true, true, true, true, true, false});

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    ref_result_columns.push_back(ref_result_col0.release());
    ref_result_columns.push_back(ref_result_col1.release());
    ref_result_columns.push_back(ref_result_col2.release());

    auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
  }

  static void verify_left_semi_join(std::optional<cudf::table_view> join_result)
  {
    ASSERT_EQ(join_result.has_value(), true);
    auto join_result_sorted = cudf::sort(*join_result);

    int64_column_wrapper ref_result_col0({1, 1, 1, 2, 3});
    int64_column_wrapper ref_result_col1({1, 2, 5, 0, 3});

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    ref_result_columns.push_back(ref_result_col0.release());
    ref_result_columns.push_back(ref_result_col1.release());

    auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
  }

  static void verify_left_anti_join(std::optional<cudf::table_view> join_result)
  {
    ASSERT_EQ(join_result.has_value(), true);
    auto join_result_sorted = cudf::sort(*join_result);

    int64_column_wrapper ref_result_col0({4});
    int64_column_wrapper ref_result_col1({4});

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    ref_result_columns.push_back(ref_result_col0.release());
    ref_result_columns.push_back(ref_result_col1.release());

    auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
  }

  static void verify_full_join(std::optional<cudf::table_view> join_result)
  {
    ASSERT_EQ(join_result.has_value(), true);
    auto join_result_sorted = cudf::sort(*join_result);

    int64_column_wrapper ref_result_col0(
      {0, 1, 1, 1, 1, 1, 1, 2, 3, 4},
      {false, true, true, true, true, true, true, true, true, true});
    int64_column_wrapper ref_result_col1(
      {0, 1, 1, 2, 2, 5, 5, 0, 3, 4},
      {false, true, true, true, true, true, true, true, true, true});
    int64_column_wrapper ref_result_col2(
      {5, 1, 1, 1, 1, 1, 1, 2, 3, 0},
      {true, true, true, true, true, true, true, true, true, false});
    int64_column_wrapper ref_result_col3(
      {2, 1, 3, 1, 3, 1, 3, 4, 0, 0},
      {true, true, true, true, true, true, true, true, true, false});

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    ref_result_columns.push_back(ref_result_col0.release());
    ref_result_columns.push_back(ref_result_col1.release());
    ref_result_columns.push_back(ref_result_col2.release());
    ref_result_columns.push_back(ref_result_col3.release());

    auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
  }
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::shared_ptr<gqe::join_task> join_task;
  std::shared_ptr<gqe::materialize_join_from_position_lists_task> late_materialization;
};

TEST_F(SingleKeyColumnJoinTest, InnerJoin)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, cache_strategy::recompute);
  join_task->execute();

  verify_inner_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinTest, InnerJoinCache)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, cache_strategy::use_cache);
  join_task->execute();

  verify_inner_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinTest, LeftJoin)
{
  construct_join_task(gqe::join_type_type::left, {0, 1, 3}, cache_strategy::recompute);
  join_task->execute();

  verify_left_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinTest, LeftJoinCache)
{
  construct_join_task(gqe::join_type_type::left, {0, 1, 3}, cache_strategy::use_cache);
  join_task->execute();

  verify_left_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinTest, LeftSemiJoin)
{
  construct_join_task(gqe::join_type_type::left_semi, {0, 1}, cache_strategy::recompute);
  join_task->execute();
  late_materialization->execute();
  verify_left_semi_join(late_materialization->result());
}

TEST_F(SingleKeyColumnJoinTest, LeftSemiJoinCache)
{
  construct_join_task(gqe::join_type_type::left_semi, {0, 1}, cache_strategy::use_cache);
  join_task->execute();
  late_materialization->execute();
  verify_left_semi_join(late_materialization->result());
}

TEST_F(SingleKeyColumnJoinTest, LeftAntiJoin)
{
  construct_join_task(gqe::join_type_type::left_anti, {0, 1}, cache_strategy::recompute);
  join_task->execute();
  late_materialization->execute();
  verify_left_anti_join(late_materialization->result());
}

TEST_F(SingleKeyColumnJoinTest, LeftAntiJoinCache)
{
  construct_join_task(gqe::join_type_type::left_anti, {0, 1}, cache_strategy::use_cache);
  join_task->execute();
  late_materialization->execute();

  verify_left_anti_join(late_materialization->result());
}

TEST_F(SingleKeyColumnJoinTest, FullJoin)
{
  construct_join_task(gqe::join_type_type::full, {0, 1, 2, 3}, cache_strategy::recompute);
  join_task->execute();

  verify_full_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinTest, FullJoinCache)
{
  construct_join_task(gqe::join_type_type::full, {0, 1, 2, 3}, cache_strategy::use_cache);
  join_task->execute();

  verify_full_join(join_task->result());
}

class SingleKeyColumnNullsEqualJoinTest : public ::testing::Test {
 protected:
  SingleKeyColumnNullsEqualJoinTest()
    : task_manager_ctx{},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }
  void construct_join_task(gqe::join_type_type join_type,
                           std::vector<cudf::size_type> projection_indices,
                           bool nulls_equal)
  {
    int64_column_wrapper left_key({0, 1, 1, 3, 4, 1}, {false, true, true, true, true, true});
    int64_column_wrapper left_payload({0, 1, 2, 3, 4, 5});
    int64_column_wrapper right_key({3, 1, 5, 1, 0}, {true, true, true, true, false});
    int64_column_wrapper right_payload({0, 1, 2, 3, 4});

    std::vector<std::unique_ptr<cudf::column>> left_table_columns;
    left_table_columns.push_back(left_key.release());
    left_table_columns.push_back(left_payload.release());
    auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));

    std::vector<std::unique_ptr<cudf::column>> right_table_columns;
    right_table_columns.push_back(right_key.release());
    right_table_columns.push_back(right_payload.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));

    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;

    auto left_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table));
    auto right_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, right_task_id, stage_id, std::move(right_table));
    std::unique_ptr<gqe::expression> join_condition;
    if (nulls_equal) {
      join_condition = std::make_unique<gqe::nulls_equal_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(2));
    } else {
      join_condition = std::make_unique<gqe::equal_expression>(
        std::make_shared<gqe::column_reference_expression>(0),
        std::make_shared<gqe::column_reference_expression>(2));
    }

    join_task = std::make_unique<gqe::join_task>(ctx_ref,
                                                 join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 std::move(projection_indices));
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::unique_ptr<gqe::join_task> join_task;
};

TEST_F(SingleKeyColumnNullsEqualJoinTest, NullsEqual)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, true);

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({0, 1, 1, 1, 1, 1, 1, 3},
                                       {false, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col1({0, 1, 1, 2, 2, 5, 5, 3});
  int64_column_wrapper ref_result_col2({4, 1, 3, 1, 3, 1, 3, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(SingleKeyColumnNullsEqualJoinTest, NullsNotEqual)
{
  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, false);

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 1, 1, 1, 3});
  int64_column_wrapper ref_result_col1({1, 1, 2, 2, 5, 5, 3});
  int64_column_wrapper ref_result_col2({1, 3, 1, 3, 1, 3, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST(HashMapCache, HashJoin)
{
  int64_column_wrapper left_key({2, 1, 1, 3, 4, 1});
  int64_column_wrapper right_key({3, 1, 5, 1, 2});

  std::vector<std::unique_ptr<cudf::column>> left_table_columns;
  left_table_columns.push_back(left_key.release());
  auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));
  auto left_view  = left_table->view();

  std::vector<std::unique_ptr<cudf::column>> right_table_columns;
  right_table_columns.push_back(right_key.release());
  auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));
  auto right_view  = right_table->view();

  gqe::join_hash_map_cache cache(gqe::join_hash_map_cache::build_location::left);

  constexpr std::size_t num_threads = 4;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::vector<std::size_t> left_num_matches(num_threads, 0);
  std::vector<std::size_t> right_num_matches(num_threads, 0);

  for (std::size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    threads.emplace_back(
      [thread_idx, left_view, right_view, &cache, &left_num_matches, &right_num_matches]() {
        auto const hash_join =
          cache.hash_map(left_view, gqe::join_algorithm::HASH_JOIN, cudf::null_equality::UNEQUAL);
        auto [right_indices, left_indices] =
          hash_join->probe(right_view, gqe::join_type_type::inner);

        left_num_matches[thread_idx]  = left_indices->size();
        right_num_matches[thread_idx] = right_indices->size();
      });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (auto const& num_matches : left_num_matches) {
    EXPECT_EQ(num_matches, 8);
  }

  for (auto const& num_matches : right_num_matches) {
    EXPECT_EQ(num_matches, 8);
  }
}

TEST(HashMapCache, UniqueKeyJoin)
{
  int64_column_wrapper left_key({1, 2, 3, 4, 5});
  int64_column_wrapper right_key({3, 1, 5, 1, 2});

  std::vector<std::unique_ptr<cudf::column>> left_table_columns;
  left_table_columns.push_back(left_key.release());
  auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));
  auto left_view  = left_table->view();

  std::vector<std::unique_ptr<cudf::column>> right_table_columns;
  right_table_columns.push_back(right_key.release());
  auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));
  auto right_view  = right_table->view();

  gqe::join_hash_map_cache cache(gqe::join_hash_map_cache::build_location::left);

  constexpr std::size_t num_threads = 4;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::vector<std::size_t> left_num_matches(num_threads, 0);
  std::vector<std::size_t> right_num_matches(num_threads, 0);

  for (std::size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    threads.emplace_back(
      [thread_idx, left_view, right_view, &cache, &left_num_matches, &right_num_matches]() {
        auto const hash_join = cache.hash_map(
          left_view, gqe::join_algorithm::UNIQUE_KEY_JOIN, cudf::null_equality::UNEQUAL);
        auto [right_indices, left_indices] =
          hash_join->probe(right_view, gqe::join_type_type::inner);

        left_num_matches[thread_idx]  = left_indices->size();
        right_num_matches[thread_idx] = right_indices->size();
      });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (auto const& num_matches : left_num_matches) {
    EXPECT_EQ(num_matches, 5);
  }

  for (auto const& num_matches : right_num_matches) {
    EXPECT_EQ(num_matches, 5);
  }
}

class NonEqualityJoinConditionTest : public ::testing::Test {
 protected:
  NonEqualityJoinConditionTest()
    : task_manager_ctx{},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }
  void construct_join_task(gqe::join_type_type join_type,
                           std::vector<cudf::size_type> projection_indices,
                           std::unique_ptr<gqe::expression> join_condition,
                           cache_strategy strategy = cache_strategy::recompute)
  {
    int64_column_wrapper left_key({2, 1, 1, 3, 4, 1});
    int64_column_wrapper left_payload({0, 1, 2, 3, 4, 5});
    int64_column_wrapper right_key({3, 1, 5, 1, 2});
    int64_column_wrapper right_payload({0, 1, 2, 3, 4});

    std::vector<std::unique_ptr<cudf::column>> left_table_columns;
    left_table_columns.push_back(left_key.release());
    left_table_columns.push_back(left_payload.release());
    auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));

    std::vector<std::unique_ptr<cudf::column>> right_table_columns;
    right_table_columns.push_back(right_key.release());
    right_table_columns.push_back(right_payload.release());
    auto right_table = std::make_unique<cudf::table>(std::move(right_table_columns));

    constexpr int32_t left_task_id  = 0;
    constexpr int32_t right_task_id = 1;
    constexpr int32_t join_task_id  = 2;
    constexpr int32_t stage_id      = 0;

    auto left_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_task_id, stage_id, std::move(left_table));
    auto right_task = std::make_shared<gqe::test::executed_task>(
      ctx_ref, right_task_id, stage_id, std::move(right_table));

    const bool late_materialize =
      join_type == gqe::join_type_type::left_semi || join_type == gqe::join_type_type::left_anti;

    std::shared_ptr<gqe::join_hash_map_cache> hash_map_cache = nullptr;
    if (strategy == cache_strategy::use_cache) {
      if (join_type == gqe::join_type_type::left_semi ||
          join_type == gqe::join_type_type::left_anti) {
        hash_map_cache = std::make_shared<gqe::join_hash_map_cache>(
          gqe::join_hash_map_cache::build_location::left);
      } else {
        hash_map_cache = std::make_shared<gqe::join_hash_map_cache>(
          gqe::join_hash_map_cache::build_location::right);
      }
    }

    join_task = std::make_shared<gqe::join_task>(ctx_ref,
                                                 join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 projection_indices,
                                                 hash_map_cache,
                                                 !late_materialize);
    if (late_materialize) {
      std::vector<std::shared_ptr<gqe::task>> tasks{{join_task}};
      late_materialization = std::make_shared<gqe::materialize_join_from_position_lists_task>(
        ctx_ref,
        join_task_id,
        stage_id,
        std::move(left_task),  // left table
        tasks,                 // positions lists
        join_type,             // join type
        projection_indices,    // projection indices
        hash_map_cache,        // hash map
        true);                 // use mark join
    }
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::shared_ptr<gqe::join_task> join_task;
  std::shared_ptr<gqe::materialize_join_from_position_lists_task> late_materialization;
};

TEST_F(NonEqualityJoinConditionTest, MixedConditionsInnerJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::not_expression>(std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::literal_expression<int64_t>>(2)))));

  construct_join_task(gqe::join_type_type::inner, {0, 1, 3}, std::move(join_condition));

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 1, 2});
  int64_column_wrapper ref_result_col1({1, 1, 2, 5, 0});
  int64_column_wrapper ref_result_col2({1, 3, 3, 3, 4});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, MixedConditionsLeftJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::not_expression>(std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::literal_expression<int64_t>>(2)))));

  construct_join_task(gqe::join_type_type::left, {0, 1, 3}, std::move(join_condition));

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 1, 2, 3, 4});
  int64_column_wrapper ref_result_col1({1, 1, 2, 5, 0, 3, 4});
  int64_column_wrapper ref_result_col2({1, 3, 3, 3, 4, 0, 0},
                                       {true, true, true, true, true, false, false});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, MixedConditionsLeftSemiJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::not_expression>(std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::literal_expression<int64_t>>(2)))));

  construct_join_task(gqe::join_type_type::left_semi, {0, 1}, std::move(join_condition));

  join_task->execute();
  late_materialization->execute();

  auto join_result = late_materialization->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 2});
  int64_column_wrapper ref_result_col1({1, 2, 5, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, MixedConditionsLeftSemiJoinCache)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::not_expression>(std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::literal_expression<int64_t>>(2)))));

  construct_join_task(
    gqe::join_type_type::left_semi, {0, 1}, std::move(join_condition), cache_strategy::use_cache);

  join_task->execute();
  late_materialization->execute();

  auto join_result = late_materialization->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 2});
  int64_column_wrapper ref_result_col1({1, 2, 5, 0});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, MixedConditionsLeftAntiJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::not_expression>(std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::literal_expression<int64_t>>(2)))));

  construct_join_task(gqe::join_type_type::left_anti, {0, 1}, std::move(join_condition));

  join_task->execute();
  late_materialization->execute();

  auto join_result = late_materialization->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({3, 4});
  int64_column_wrapper ref_result_col1({3, 4});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, MixedConditionsFullJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::equal_expression>(std::make_shared<gqe::column_reference_expression>(0),
                                            std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::not_expression>(std::make_shared<gqe::greater_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(1),
      std::make_shared<gqe::multiply_expression>(
        std::make_shared<gqe::column_reference_expression>(3),
        std::make_shared<gqe::literal_expression<int64_t>>(2)))));

  construct_join_task(gqe::join_type_type::full, {0, 1, 3}, std::move(join_condition));

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({0, 0, 1, 1, 1, 1, 2, 3, 4},
                                       {false, false, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col1({0, 0, 1, 1, 2, 5, 0, 3, 4},
                                       {false, false, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col2({0, 2, 1, 3, 3, 3, 4, 0, 0},
                                       {true, true, true, true, true, true, true, false, false});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, NoEqualityConditionsInnerJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::column_reference_expression>(3)));

  construct_join_task(gqe::join_type_type::inner, {0, 1, 2, 3}, std::move(join_condition));

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({2, 2, 3});
  int64_column_wrapper ref_result_col1({0, 0, 3});
  int64_column_wrapper ref_result_col2({1, 1, 2});
  int64_column_wrapper ref_result_col3({1, 3, 4});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());
  ref_result_columns.push_back(ref_result_col3.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, NoEqualityConditionsLeftJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::column_reference_expression>(3)));

  construct_join_task(gqe::join_type_type::left, {0, 1, 2, 3}, std::move(join_condition));

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 2, 2, 3, 4});
  int64_column_wrapper ref_result_col1({1, 2, 5, 0, 0, 3, 4});
  int64_column_wrapper ref_result_col2({0, 0, 0, 1, 1, 2, 0},
                                       {false, false, false, true, true, true, false});
  int64_column_wrapper ref_result_col3({0, 0, 0, 1, 3, 4, 0},
                                       {false, false, false, true, true, true, false});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());
  ref_result_columns.push_back(ref_result_col3.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, NoEqualityConditionsLeftSemiJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::column_reference_expression>(3)));

  construct_join_task(gqe::join_type_type::left_semi, {0, 1}, std::move(join_condition));

  join_task->execute();
  late_materialization->execute();

  auto join_result = late_materialization->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({2, 3});
  int64_column_wrapper ref_result_col1({0, 3});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, NoEqualityConditionsLeftAntiJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::column_reference_expression>(3)));

  construct_join_task(gqe::join_type_type::left_anti, {0, 1}, std::move(join_condition));

  join_task->execute();
  late_materialization->execute();

  auto join_result = late_materialization->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({1, 1, 1, 4});
  int64_column_wrapper ref_result_col1({1, 2, 5, 4});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

TEST_F(NonEqualityJoinConditionTest, NoEqualityConditionsFullJoin)
{
  auto join_condition = std::make_unique<gqe::logical_and_expression>(
    std::make_shared<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::column_reference_expression>(2)),
    std::make_shared<gqe::less_expression>(std::make_shared<gqe::column_reference_expression>(1),
                                           std::make_shared<gqe::column_reference_expression>(3)));

  construct_join_task(gqe::join_type_type::full, {0, 1, 2, 3}, std::move(join_condition));

  join_task->execute();

  auto join_result = join_task->result();
  ASSERT_EQ(join_result.has_value(), true);
  auto join_result_sorted = cudf::sort(*join_result);

  int64_column_wrapper ref_result_col0({0, 0, 1, 1, 1, 2, 2, 3, 4},
                                       {false, false, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col1({0, 0, 1, 2, 5, 0, 0, 3, 4},
                                       {false, false, true, true, true, true, true, true, true});
  int64_column_wrapper ref_result_col2({3, 5, 0, 0, 0, 1, 1, 2, 0},
                                       {true, true, false, false, false, true, true, true, false});
  int64_column_wrapper ref_result_col3({0, 2, 0, 0, 0, 1, 3, 4, 0},
                                       {true, true, false, false, false, true, true, true, false});

  std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
  ref_result_columns.push_back(ref_result_col0.release());
  ref_result_columns.push_back(ref_result_col1.release());
  ref_result_columns.push_back(ref_result_col2.release());
  ref_result_columns.push_back(ref_result_col3.release());

  auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
}

class MaterializeJoinFromPositionListsTest : public ::testing::Test {
 protected:
  MaterializeJoinFromPositionListsTest()
    : task_manager_ctx{},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }
  void construct_materialize_task(gqe::join_type_type join_type)
  {
    constexpr int32_t stage_id = 0;

    cudf::test::fixed_width_column_wrapper<int64_t> left_table_col({10, 11, 12, 13, 14, 15});
    std::vector<std::unique_ptr<cudf::column>> left_table_columns;
    left_table_columns.push_back(left_table_col.release());

    auto left_table = std::make_unique<cudf::table>(std::move(left_table_columns));
    constexpr int32_t left_table_task_id = 0;
    auto left_table_task                 = std::make_shared<gqe::test::executed_task>(
      ctx_ref, left_table_task_id, stage_id, std::move(left_table));

    cudf::test::fixed_width_column_wrapper<cudf::size_type> position_list_1_col({2, 4, 5});
    std::vector<std::unique_ptr<cudf::column>> position_list_1_columns;
    position_list_1_columns.push_back(position_list_1_col.release());

    auto position_list_1 = std::make_unique<cudf::table>(std::move(position_list_1_columns));
    constexpr int32_t position_list_1_task_id = 1;
    auto position_list_1_task                 = std::make_shared<gqe::test::executed_task>(
      ctx_ref, position_list_1_task_id, stage_id, std::move(position_list_1));

    cudf::test::fixed_width_column_wrapper<cudf::size_type> position_list_2_col({2, 3, 5});
    std::vector<std::unique_ptr<cudf::column>> position_list_2_columns;
    position_list_2_columns.push_back(position_list_2_col.release());

    auto position_list_2 = std::make_unique<cudf::table>(std::move(position_list_2_columns));
    constexpr int32_t position_list_2_task_id = 2;
    auto position_list_2_task                 = std::make_shared<gqe::test::executed_task>(
      ctx_ref, position_list_2_task_id, stage_id, std::move(position_list_2));

    std::vector<std::shared_ptr<gqe::task>> inputs;
    inputs.push_back(std::move(position_list_1_task));
    inputs.push_back(std::move(position_list_2_task));

    constexpr int32_t materialize_task_id           = 3;
    std::vector<cudf::size_type> projection_indices = {0};

    materialize_task =
      std::make_unique<gqe::materialize_join_from_position_lists_task>(ctx_ref,
                                                                       materialize_task_id,
                                                                       stage_id,
                                                                       std::move(left_table_task),
                                                                       std::move(inputs),
                                                                       join_type,
                                                                       projection_indices);
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::unique_ptr<gqe::materialize_join_from_position_lists_task> materialize_task;
};

TEST_F(MaterializeJoinFromPositionListsTest, LeftSemi)
{
  construct_materialize_task(gqe::join_type_type::left_semi);
  materialize_task->execute();

  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_0({12, 13, 14, 15});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  auto materialize_task_result = materialize_task->result();
  ASSERT_EQ(materialize_task_result.has_value(), true);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(materialize_task_result.value(), ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(materialize_task_result.value(), ref_table->view());
}

TEST_F(MaterializeJoinFromPositionListsTest, LeftAnti)
{
  construct_materialize_task(gqe::join_type_type::left_anti);
  materialize_task->execute();

  cudf::test::fixed_width_column_wrapper<int64_t> ref_col_0({12, 15});

  std::vector<std::unique_ptr<cudf::column>> ref_columns;
  ref_columns.push_back(ref_col_0.release());
  auto ref_table = std::make_unique<cudf::table>(std::move(ref_columns));

  auto materialize_task_result = materialize_task->result();
  ASSERT_EQ(materialize_task_result.has_value(), true);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(materialize_task_result.value(), ref_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(materialize_task_result.value(), ref_table->view());
}

// TODO: Add a test on multi column join keys
