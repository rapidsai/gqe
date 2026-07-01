/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "utilities.hpp"

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/partition.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe_test/base_fixture.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <gtest/gtest.h>

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using int64_column_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

class ShuffleTaskTest : public gqe::test::BaseFixture {
 protected:
  ShuffleTaskTest() : ctx_ref{get_task_manager_ctx(), get_query_ctx()} {}

  void construct_input_task(int32_t const stage_id, gqe::context_reference& ctx_ref)
  {
    constexpr int32_t input_task_id = 0;

    int64_column_wrapper input_col_0({1, 1, 3, 4, 9, 3, 7, 3, 9, 4});
    int64_column_wrapper input_col_1(
      {0, 12, 13, 0, 12, 16, 0, 16, 13, 20},
      {false, true, true, false, true, true, false, true, true, true});

    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.push_back(input_col_0.release());
    input_columns.push_back(input_col_1.release());

    input_table = std::make_unique<cudf::table>(std::move(input_columns));
    input_task  = std::make_shared<gqe::test::executed_task>(
      ctx_ref, input_task_id, stage_id, std::make_unique<cudf::table>(input_table->view()));
  }

  /**
   * @brief Verify that @p partition_task_ produced a valid hash partitioning of @p input_view.
   *
   * The checks are intentionally hash-function-agnostic so the test does not break
   * when cuDF changes its row-hash implementation (cf. rapidsai/cudf#20796, which
   * shifted the per-row hash value and therefore the bucket assignment of every key):
   *
   *   1. Offsets cover [0, num_rows], have `num_partitions + 1` entries, and are
   *      monotone non-decreasing.
   *   2. The output table is a row-permutation of the input (compared as multisets
   *      via sort).
   *   3. Rows are grouped by hash key: any distinct value of the partition column
   *      lands in at most one partition slice (i.e. equal keys are co-located,
   *      which is the actual contract of `cudf::hash_partition`).
   */
  void assert_valid_hash_partition(gqe::partition_task& partition_task_,
                                   cudf::table_view const& input_view,
                                   int32_t partition_col,
                                   int32_t num_partitions)
  {
    auto result_opt = partition_task_.result();
    ASSERT_TRUE(result_opt.has_value());
    auto const output_view = result_opt.value();

    EXPECT_EQ(output_view.num_rows(), input_view.num_rows());
    EXPECT_EQ(output_view.num_columns(), input_view.num_columns());

    EXPECT_EQ(partition_task_.partition_offset(0), 0);
    EXPECT_EQ(partition_task_.partition_offset(num_partitions), input_view.num_rows());
    for (int32_t i = 0; i < num_partitions; ++i) {
      EXPECT_LE(partition_task_.partition_offset(i), partition_task_.partition_offset(i + 1))
        << "partition_offsets not monotone at boundary " << i;
    }

    auto const sorted_output = cudf::sort(output_view);
    auto const sorted_input  = cudf::sort(input_view);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_output->view(), sorted_input->view());

    // For each non-empty partition slice, take the distinct values of the partition
    // column. Concatenate all of them and re-distinct: if any value appeared in more
    // than one slice the global distinct count would be strictly less than the sum
    // of the per-slice distinct counts.
    std::vector<std::unique_ptr<cudf::column>> per_slice_distinct;
    cudf::size_type sum_per_slice = 0;
    for (int32_t i = 0; i < num_partitions; ++i) {
      auto const start = partition_task_.partition_offset(i);
      auto const end   = partition_task_.partition_offset(i + 1);
      if (start == end) { continue; }
      auto const key_slice = cudf::slice(output_view.column(partition_col), {start, end}).front();
      auto distinct_table  = cudf::distinct(cudf::table_view{{key_slice}},
                                            {0},
                                           cudf::duplicate_keep_option::KEEP_ANY,
                                           cudf::null_equality::EQUAL);
      sum_per_slice += distinct_table->num_rows();
      per_slice_distinct.push_back(std::move(distinct_table->release().front()));
    }

    if (per_slice_distinct.empty()) { return; }

    std::vector<cudf::column_view> distinct_views;
    distinct_views.reserve(per_slice_distinct.size());
    for (auto const& col : per_slice_distinct) {
      distinct_views.push_back(col->view());
    }
    auto const all_distinct    = cudf::concatenate(distinct_views);
    auto const global_distinct = cudf::distinct(cudf::table_view{{all_distinct->view()}},
                                                {0},
                                                cudf::duplicate_keep_option::KEEP_ANY,
                                                cudf::null_equality::EQUAL);
    EXPECT_EQ(global_distinct->num_rows(), sum_per_slice)
      << "hash_partition placed the same key value in more than one partition";
  }

  gqe::context_reference ctx_ref;
  std::unique_ptr<cudf::table> input_table;
  std::shared_ptr<gqe::test::executed_task> input_task;
};

TEST_F(ShuffleTaskTest, ShuffleOnNonNullColumns)
{
  constexpr int32_t stage_id        = 0;
  constexpr int32_t shuffle_task_id = 1;
  constexpr int32_t partition_col   = 0;
  constexpr int32_t num_partitions  = 4;

  construct_input_task(stage_id, ctx_ref);
  auto const input_view = input_table->view();

  std::vector<std::unique_ptr<gqe::expression>> shuffle_cols;
  shuffle_cols.push_back(std::make_unique<gqe::column_reference_expression>(partition_col));

  auto partition_task = std::make_unique<gqe::partition_task>(ctx_ref,
                                                              shuffle_task_id,
                                                              stage_id,
                                                              std::move(input_task),
                                                              std::move(shuffle_cols),
                                                              num_partitions);

  partition_task->execute();
  assert_valid_hash_partition(*partition_task, input_view, partition_col, num_partitions);
}

TEST_F(ShuffleTaskTest, ShuffleOnNullableColumns)
{
  constexpr int32_t stage_id        = 0;
  constexpr int32_t shuffle_task_id = 1;
  constexpr int32_t partition_col   = 1;
  constexpr int32_t num_partitions  = 3;

  construct_input_task(stage_id, ctx_ref);
  auto const input_view = input_table->view();

  std::vector<std::unique_ptr<gqe::expression>> shuffle_cols;
  shuffle_cols.push_back(std::make_unique<gqe::column_reference_expression>(partition_col));

  auto partition_task = std::make_unique<gqe::partition_task>(ctx_ref,
                                                              shuffle_task_id,
                                                              stage_id,
                                                              std::move(input_task),
                                                              std::move(shuffle_cols),
                                                              num_partitions);

  partition_task->execute();
  assert_valid_hash_partition(*partition_task, input_view, partition_col, num_partitions);
}
