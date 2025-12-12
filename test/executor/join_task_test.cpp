/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/join.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/expression/binary_op.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/expression.hpp>
#include <gqe/expression/literal.hpp>
#include <gqe/expression/unary_op.hpp>
#include <gqe/memory_resource/memory_utilities.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/kernel_fusion_helpers.hpp>

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

struct join_test_data {
  std::string name;
  std::vector<int64_t> left_key_data;
  std::vector<int64_t> left_payload_data;
  std::vector<int64_t> right_key_data;
  std::vector<int64_t> right_payload_data;
};

/*
 * This test suite constructs the input tables with handpicked values. Both the left and the right
 * table have 2 columns, 1 key column and 1 payload column.
 */
class SingleKeyColumnJoinTest
  : public ::testing::TestWithParam<
      std::tuple<gqe::join_type_type, cache_strategy, join_test_data, bool, bool>> {
 protected:
  SingleKeyColumnJoinTest()
    : params(true),
      task_manager_ctx(params, gqe::memory_resource::create_static_memory_pool()),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  gqe::join_type_type get_join_type() const { return std::get<0>(GetParam()); }
  cache_strategy get_cache_strategy() const { return std::get<1>(GetParam()); }
  const join_test_data& get_test_data() const { return std::get<2>(GetParam()); }
  bool get_use_late_materialization() const { return std::get<3>(GetParam()); }
  bool get_mark_join() const { return std::get<4>(GetParam()); }

  std::vector<int64_t> left_key_data() const { return get_test_data().left_key_data; }
  std::vector<int64_t> left_payload_data() const { return get_test_data().left_payload_data; }
  std::vector<int64_t> right_key_data() const { return get_test_data().right_key_data; }
  std::vector<int64_t> right_payload_data() const { return get_test_data().right_payload_data; }

  std::vector<cudf::size_type> get_projection_indices()
  {
    auto const join_type = get_join_type();
    switch (join_type) {
      case gqe::join_type_type::inner:  // fallthrough
      case gqe::join_type_type::left: return {0, 1, 3};
      case gqe::join_type_type::left_semi:  // fallthrough
      case gqe::join_type_type::left_anti: return {0, 1};
      case gqe::join_type_type::full: return {0, 1, 2, 3};
      default: throw std::logic_error("Unknown join type");
    }
  }

  bool supports_late_materialization() const
  {
    auto const join_type = get_join_type();
    return join_type == gqe::join_type_type::left_semi ||
           join_type == gqe::join_type_type::left_anti;
  }

  // Returns true if this combination of parameters is supported.  Returns false if this combination
  // of parameters is expected to throw an exception.
  bool is_supported_combination() const
  {
    if (!supports_late_materialization() && get_use_late_materialization()) {
      // Late materialization is only supported for left semi and left anti joins.
      return false;
    }
    if (supports_late_materialization() && !get_use_late_materialization() && get_mark_join() &&
        get_cache_strategy() == cache_strategy::use_cache) {
      // Immediate materialization is only supported for left semi and left anti joins that are not
      // from cache.
      return false;
    }
    return true;
  }

  void construct_join_task()
  {
    auto left_key_vec      = left_key_data();
    auto left_payload_vec  = left_payload_data();
    auto right_key_vec     = right_key_data();
    auto right_payload_vec = right_payload_data();

    int64_column_wrapper left_key(left_key_vec.begin(), left_key_vec.end());
    int64_column_wrapper left_payload(left_payload_vec.begin(), left_payload_vec.end());
    int64_column_wrapper right_key(right_key_vec.begin(), right_key_vec.end());
    int64_column_wrapper right_payload(right_payload_vec.begin(), right_payload_vec.end());

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
    const bool late_materialize                              = get_use_late_materialization();
    std::shared_ptr<gqe::join_hash_map_cache> hash_map_cache = nullptr;
    if (get_cache_strategy() == cache_strategy::use_cache) {
      if (supports_late_materialization()) {
        hash_map_cache = std::make_shared<gqe::join_hash_map_cache>(
          gqe::join_hash_map_cache::build_location::left);
      } else {
        hash_map_cache = std::make_shared<gqe::join_hash_map_cache>(
          gqe::join_hash_map_cache::build_location::right);
      }
    }

    auto const join_type = get_join_type();
    auto const mark_join = get_mark_join();
    join_task            = std::make_shared<gqe::join_task>(ctx_ref,
                                                 join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 get_projection_indices(),
                                                 hash_map_cache,
                                                 !late_materialize,
                                                 gqe::unique_keys_policy::none,
                                                 false,  // perfect_hashing
                                                 /*left_filter_condition=*/nullptr,
                                                 /*right_filter_condition=*/nullptr,
                                                 mark_join);

    if (late_materialize) {
      std::vector<std::shared_ptr<gqe::task>> tasks{{join_task}};
      late_materialization = std::make_shared<gqe::materialize_join_from_position_lists_task>(
        ctx_ref,
        join_task_id,
        stage_id,
        std::move(left_task),      // left table
        std::move(tasks),          // positions lists
        join_type,                 // join type
        get_projection_indices(),  // projection indices
        hash_map_cache,            // hash map
        mark_join);                // use mark join
    }
  }

  void verify_join()
  {
    auto const join_type = get_join_type();
    std::optional<cudf::table_view> join_result;
    if (get_use_late_materialization()) {
      join_result = late_materialization->result();
    } else {
      join_result = join_task->result();
    }
    ASSERT_EQ(join_result.has_value(), true);
    auto join_result_sorted = cudf::sort(*join_result);

    auto left_key_vec      = left_key_data();
    auto left_payload_vec  = left_payload_data();
    auto right_key_vec     = right_key_data();
    auto right_payload_vec = right_payload_data();

    std::unordered_multimap<int64_t, int64_t> left_table;
    for (size_t i = 0; i < left_key_vec.size(); i++) {
      left_table.insert({left_key_vec[i], left_payload_vec[i]});
    }
    std::unordered_multimap<int64_t, int64_t> right_table;
    for (size_t i = 0; i < right_key_vec.size(); i++) {
      right_table.insert({right_key_vec[i], right_payload_vec[i]});
    }
    std::vector<std::vector<int64_t>> result_tables(4);
    std::vector<std::vector<bool>> result_validity(4);

    for (size_t i = 0; i < left_key_vec.size(); i++) {
      auto left_key              = left_key_vec[i];
      auto left_key_validity     = true;
      auto left_payload          = left_payload_vec[i];
      auto left_payload_validity = true;
      auto match_range           = right_table.equal_range(left_key);
      std::vector<std::pair<int64_t, int64_t>> matches(match_range.first, match_range.second);
      auto has_match              = matches.size() > 0;
      auto right_key_validity     = true;
      auto right_payload_validity = true;
      if ((!has_match &&
           (join_type == gqe::join_type_type::left_anti || join_type == gqe::join_type_type::left ||
            join_type == gqe::join_type_type::full)) ||
          (has_match && (join_type == gqe::join_type_type::left_semi))) {
        // Pretend to have exactly one invalid match.
        matches                = std::vector<std::pair<int64_t, int64_t>>{{0, 0}};
        right_key_validity     = false;
        right_payload_validity = false;
      } else if (has_match && (join_type == gqe::join_type_type::left_anti)) {
        // Pretend to have zero matches.
        matches = std::vector<std::pair<int64_t, int64_t>>{};
      }
      // Record matches.
      for (auto const& match : matches) {
        auto right_key     = match.first;
        auto right_payload = match.second;
        result_tables[0].push_back(left_key);
        result_tables[1].push_back(left_payload);
        result_tables[2].push_back(right_key);
        result_tables[3].push_back(right_payload);
        result_validity[0].push_back(left_key_validity);
        result_validity[1].push_back(left_payload_validity);
        result_validity[2].push_back(right_key_validity);
        result_validity[3].push_back(right_payload_validity);
      }
    }
    if (join_type == gqe::join_type_type::full) {
      // All need to add rows on the right side that are not in the left side.
      for (auto it = right_table.begin(); it != right_table.end(); it++) {
        if (left_table.find(it->first) == left_table.end()) {
          auto right_key              = it->first;
          auto right_key_validity     = true;
          auto right_payload          = it->second;
          auto right_payload_validity = true;
          result_tables[0].push_back(0);
          result_tables[1].push_back(0);
          result_tables[2].push_back(right_key);
          result_tables[3].push_back(right_payload);
          result_validity[0].push_back(false);
          result_validity[1].push_back(false);
          result_validity[2].push_back(right_key_validity);
          result_validity[3].push_back(right_payload_validity);
        }
      }
    }

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    auto projection_indices = get_projection_indices();
    for (auto const& index : projection_indices) {
      ref_result_columns.push_back(int64_column_wrapper(result_tables[index].begin(),
                                                        result_tables[index].end(),
                                                        result_validity[index].begin())
                                     .release());
    }

    auto ref_result_table        = std::make_unique<cudf::table>(std::move(ref_result_columns));
    auto ref_result_table_sorted = cudf::sort(ref_result_table->view());

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(),
                                            ref_result_table_sorted->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table_sorted->view());
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::shared_ptr<gqe::join_task> join_task;
  std::shared_ptr<gqe::materialize_join_from_position_lists_task> late_materialization;
};

TEST_P(SingleKeyColumnJoinTest, JoinTest)
{
  try {
    construct_join_task();
    join_task->execute();
    if (get_use_late_materialization()) { late_materialization->execute(); }
    verify_join();
  } catch (const std::exception& e) {
    if (is_supported_combination()) {
      // If the combination is supported but we threw an exception, fail the test.
      FAIL() << "Expected no exception, but got: " << e.what();
    }
    // Otherwise, we expected an exception and we got one, so we're good.
    return;
  }
  // If the combination is not supported and we didn't throw an exception, fail the test.
  if (!is_supported_combination()) { FAIL() << "Expected an exception, but got none."; }
}

INSTANTIATE_TEST_SUITE_P(
  SingleKeyColumnJoinTest,
  SingleKeyColumnJoinTest,
  ::testing::Combine(
    ::testing::Values(gqe::join_type_type::inner,
                      gqe::join_type_type::left,
                      gqe::join_type_type::left_semi,
                      gqe::join_type_type::left_anti,
                      gqe::join_type_type::full),
    ::testing::Values(cache_strategy::recompute, cache_strategy::use_cache),
    ::testing::Values(
      join_test_data{
        "basic", {2, 1, 1, 3, 4, 1}, {0, 1, 2, 3, 4, 5}, {3, 1, 5, 1, 2}, {0, 1, 2, 3, 4}},
      join_test_data{"simple", {1, 2, 3}, {10, 20, 30}, {2, 3, 4}, {200, 300, 400}},
      join_test_data{
        "duplicates", {1, 1, 1, 2, 2}, {10, 11, 12, 20, 21}, {1, 1, 3}, {100, 101, 300}},
      join_test_data{"empty_left", {}, {}, {3, 1, 5, 1, 2}, {0, 1, 2, 3, 4}},
      join_test_data{"empty_right", {2, 1, 1, 3, 4, 1}, {0, 1, 2, 3, 4, 5}, {}, {}},
      join_test_data{"empty_left_and_right", {}, {}, {}, {}}),
    ::testing::Bool(),
    ::testing::Bool()),
  [](const ::testing::TestParamInfo<
     std::tuple<gqe::join_type_type, cache_strategy, join_test_data, bool, bool>>& info) {
    std::string join_type_name;
    switch (std::get<0>(info.param)) {
      case gqe::join_type_type::inner: join_type_name = "inner"; break;
      case gqe::join_type_type::left: join_type_name = "left"; break;
      case gqe::join_type_type::left_semi: join_type_name = "left_semi"; break;
      case gqe::join_type_type::left_anti: join_type_name = "left_anti"; break;
      case gqe::join_type_type::full: join_type_name = "full"; break;
      default: join_type_name = "unknown"; break;
    }
    std::string cache_name;
    switch (std::get<1>(info.param)) {
      case cache_strategy::recompute: cache_name = "recompute"; break;
      case cache_strategy::use_cache: cache_name = "use_cache"; break;
      default: cache_name = "unknown"; break;
    }
    std::string late_mat_name  = std::get<3>(info.param) ? "late_mat" : "immediate_mat";
    std::string mark_join_name = std::get<4>(info.param) ? "mark_join" : "no_mark_join";
    return join_type_name + "_" + cache_name + "_" + std::get<2>(info.param).name + "_" +
           late_mat_name + "_" + mark_join_name;
  });

class SingleKeyColumnNullsEqualJoinTest : public ::testing::Test {
 protected:
  SingleKeyColumnNullsEqualJoinTest()
    : params(true),
      task_manager_ctx(params, gqe::memory_resource::create_static_memory_pool()),
      query_ctx(params),
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

  gqe::optimization_parameters params;
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
        cudf::column_view empty_mask;
        auto const& impl =
          cache.get_or_create<gqe::hash_join_impl>(left_view, cudf::null_equality::UNEQUAL);
        auto [right_indices, left_indices] =
          impl.probe(right_view, empty_mask, gqe::join_type_type::inner);

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
        cudf::column_view empty_mask;
        auto const& impl = cache.get_or_create<gqe::unique_key_join_impl>(
          left_view, cudf::column_view(), cudf::null_equality::UNEQUAL);
        auto [right_indices, left_indices] =
          impl.probe(right_view, empty_mask, gqe::join_type_type::inner);

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

TEST(HashMapCache, UniqueKeyJoinWithFilter)
{
  int64_column_wrapper left_key({1, 2, 3, 4, 5});
  int64_column_wrapper right_key({3, 1, 5, 1, 2});
  std::vector<bool> probe_mask_data1             = {false, true, true, true, true};
  std::vector<bool> probe_mask_data2             = {false, false, true, true, true};
  std::vector<bool> probe_mask_data3             = {false, false, false, true, true};
  std::vector<bool> probe_mask_data4             = {false, false, false, false, true};
  std::vector<std::vector<bool>> probe_mask_data = {
    probe_mask_data1, probe_mask_data2, probe_mask_data3, probe_mask_data4};

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

  // Filter out left_key == 3
  auto build_filter = std::make_unique<gqe::not_equal_expression>(
    std::make_shared<gqe::column_reference_expression>(0),
    std::make_shared<gqe::literal_expression<int64_t>>(3));

  for (std::size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    threads.emplace_back([thread_idx,
                          left_view,
                          right_view,
                          &cache,
                          &left_num_matches,
                          &right_num_matches,
                          build_filter = build_filter.get(),
                          &probe_mask_data]() {
      // Build filter mask inside hash map cache to avoid rebuilding by each thread
      auto const& impl = cache.get_or_create_fn<gqe::unique_key_join_impl>([&]() {
        auto [mask_view, mask_column] =
          gqe::utility::get_mask_from_filter(gqe::optimization_parameters(true),
                                             left_view,
                                             build_filter->clone(),
                                             /*column_reference_offset=*/0);
        return std::make_unique<gqe::unique_key_join_impl>(
          left_view, mask_view, cudf::null_equality::UNEQUAL);
      });
      cudf::test::fixed_width_column_wrapper<bool> probe_mask(probe_mask_data[thread_idx].begin(),
                                                              probe_mask_data[thread_idx].end());
      auto [right_indices, left_indices] =
        impl.probe(right_view, probe_mask, gqe::join_type_type::inner);

      left_num_matches[thread_idx]  = left_indices->size();
      right_num_matches[thread_idx] = right_indices->size();
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(left_num_matches[0], 4);
  EXPECT_EQ(left_num_matches[1], 3);
  EXPECT_EQ(left_num_matches[2], 2);
  EXPECT_EQ(left_num_matches[3], 1);

  EXPECT_EQ(right_num_matches[0], 4);
  EXPECT_EQ(right_num_matches[1], 3);
  EXPECT_EQ(right_num_matches[2], 2);
  EXPECT_EQ(right_num_matches[3], 1);
}

class NonEqualityJoinConditionTest : public ::testing::Test {
 protected:
  NonEqualityJoinConditionTest()
    : params(true),
      task_manager_ctx(params, gqe::memory_resource::create_static_memory_pool()),
      query_ctx(params),
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

  gqe::optimization_parameters params;
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
    : params(true),
      task_manager_ctx(params, gqe::memory_resource::create_static_memory_pool()),
      query_ctx(params),
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

  gqe::optimization_parameters params;
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

/*
 * This test suite constructs the input tables with handpicked values. Both the left and the right
 * table have 2 columns, 1 key column and 1 payload column.
 */
class SingleKeyColumnJoinWithFilterTest : public ::testing::Test {
 protected:
  SingleKeyColumnJoinWithFilterTest()
    : params(true),
      task_manager_ctx(params),
      query_ctx(params),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }
  void construct_join_task(gqe::join_type_type join_type,
                           std::vector<cudf::size_type> projection_indices,
                           cache_strategy strategy,
                           bool perfect_hashing)
  {
    // Left table keys are unique here.
    int64_column_wrapper left_key({1, 2, 3, 4, 5, 6});
    int64_column_wrapper left_payload({5, 4, 3, 2, 1, 0});
    int64_column_wrapper right_key({3, 1, 5, 3, 2});
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

    // Left key > 2
    auto left_filter_condition = std::make_unique<gqe::greater_expression>(
      std::make_shared<gqe::column_reference_expression>(0),
      std::make_shared<gqe::literal_expression<int64_t>>(2));

    // Right key != 2
    auto right_filter_condition = std::make_unique<gqe::not_equal_expression>(
      std::make_shared<gqe::column_reference_expression>(2),
      std::make_shared<gqe::literal_expression<int64_t>>(2));

    std::shared_ptr<gqe::join_hash_map_cache> hash_map_cache = nullptr;
    if (strategy == cache_strategy::use_cache) {
      hash_map_cache =
        std::make_shared<gqe::join_hash_map_cache>(gqe::join_hash_map_cache::build_location::left);
    }

    const bool late_materialize =
      join_type == gqe::join_type_type::left_semi || join_type == gqe::join_type_type::left_anti;

    join_task = std::make_shared<gqe::join_task>(ctx_ref,
                                                 join_task_id,
                                                 stage_id,
                                                 left_task,
                                                 right_task,
                                                 join_type,
                                                 std::move(join_condition),
                                                 projection_indices,
                                                 hash_map_cache,
                                                 !late_materialize,
                                                 gqe::unique_keys_policy::left,
                                                 perfect_hashing,
                                                 std::move(left_filter_condition),
                                                 std::move(right_filter_condition),
                                                 /*mark_join=*/true);

    if (late_materialize) {
      std::vector<std::shared_ptr<gqe::task>> tasks{{join_task}};
      late_materialization = std::make_shared<gqe::materialize_join_from_position_lists_task>(
        ctx_ref,
        join_task_id + 1,
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

    int64_column_wrapper ref_result_col0({3, 3, 5});
    int64_column_wrapper ref_result_col1({3, 3, 1});
    int64_column_wrapper ref_result_col2({0, 3, 2});

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

    int64_column_wrapper ref_result_col0({3, 5});
    int64_column_wrapper ref_result_col1({3, 1});

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

    int64_column_wrapper ref_result_col0({4, 6});
    int64_column_wrapper ref_result_col1({2, 0});

    std::vector<std::unique_ptr<cudf::column>> ref_result_columns;
    ref_result_columns.push_back(ref_result_col0.release());
    ref_result_columns.push_back(ref_result_col1.release());

    auto ref_result_table = std::make_unique<cudf::table>(std::move(ref_result_columns));

    CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(join_result_sorted->view(), ref_result_table->view());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(join_result_sorted->view(), ref_result_table->view());
  }

  gqe::optimization_parameters params;
  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
  std::shared_ptr<gqe::join_task> join_task;
  std::shared_ptr<gqe::materialize_join_from_position_lists_task> late_materialization;
};

TEST_F(SingleKeyColumnJoinWithFilterTest, InnerJoinWithFilter)
{
  construct_join_task(
    gqe::join_type_type::inner, {0, 1, 3}, cache_strategy::recompute, /*perfect_hashing=*/false);
  join_task->execute();

  verify_inner_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinWithFilterTest, PerfectJoinWithFilter)
{
  construct_join_task(
    gqe::join_type_type::inner, {0, 1, 3}, cache_strategy::recompute, /*perfect_hashing=*/true);
  join_task->execute();

  verify_inner_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinWithFilterTest, CachedInnerJoinWithFilter)
{
  construct_join_task(
    gqe::join_type_type::inner, {0, 1, 3}, cache_strategy::use_cache, /*perfect_hashing=*/false);
  join_task->execute();

  verify_inner_join(join_task->result());
}

TEST_F(SingleKeyColumnJoinWithFilterTest, CachedLeftSemiMarkJoinWithFilter)
{
  construct_join_task(
    gqe::join_type_type::left_semi, {0, 1}, cache_strategy::use_cache, /*perfect_hashing=*/false);
  join_task->execute();
  late_materialization->execute();

  verify_left_semi_join(late_materialization->result());
}

TEST_F(SingleKeyColumnJoinWithFilterTest, CachedLeftAntiMarkJoinWithFilter)
{
  construct_join_task(
    gqe::join_type_type::left_anti, {0, 1}, cache_strategy::use_cache, /*perfect_hashing=*/false);
  join_task->execute();
  late_materialization->execute();

  verify_left_anti_join(late_materialization->result());
}

// TODO: Add a test on multi column join keys
