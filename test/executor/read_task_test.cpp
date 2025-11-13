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

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class ReadTest : public ::testing::Test {
 protected:
  ReadTest()
    : task_manager_ctx{},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{&task_manager_ctx, &query_ctx}
  {
  }

  gqe::task_manager_context task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
};

TEST_F(ReadTest, FixedDataReadTaskTest)
{
  // Construct a test table with fixed data
  cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2, 3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1({6, 5, 4, 3, 2, 1});
  cudf::test::strings_column_wrapper col_2({"apple", "orange", "duck", "big", "random", "c++"});

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(col_0.release());
  columns.push_back(col_1.release());
  columns.push_back(col_2.release());

  auto const num_columns = columns.size();
  auto test_table        = std::make_unique<cudf::table>(std::move(columns));

  std::vector<std::string> const column_names = {"col_0", "col_1", "col_2"};

  // Write the test table to disk using libcudf's Parquet writer

  cudf::io::table_input_metadata metadata(test_table->view());
  ASSERT_EQ(metadata.column_metadata.size(), num_columns);
  ASSERT_EQ(column_names.size(), num_columns);

  for (std::size_t column_idx = 0; column_idx < num_columns; column_idx++)
    metadata.column_metadata[column_idx].set_name(column_names[column_idx]);

  auto filepath = temp_env->get_temp_filepath("FixedDataTable.parquet");
  auto options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(filepath), test_table->view());
  options.metadata(metadata);
  cudf::io::write_parquet(options);

  // Load the test table from disk using a read task
  std::vector<std::string> filepaths{filepath};
  std::vector<cudf::data_type> column_types = {cudf::data_type(cudf::type_id::INT64),
                                               cudf::data_type(cudf::type_id::INT32),
                                               cudf::data_type(cudf::type_id::STRING)};

  auto read_task = std::make_unique<gqe::storage::parquet_read_task>(
    ctx_ref, 0, 0, filepaths, column_names, column_types);
  read_task->execute();
  auto result = read_task->result();

  // Compare the loaded table against the original test table

  ASSERT_EQ(result.has_value(), true);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(result.value(), test_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.value(), test_table->view());
}

TEST_F(ReadTest, FixedDataReadTaskTestMultiTask)
{
  // Construct a test table with fixed data

  cudf::test::fixed_width_column_wrapper<int64_t> col_0_first({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1_first({6, 5, 4});
  cudf::test::strings_column_wrapper col_2_first({"apple", "orange", "duck"});

  cudf::test::fixed_width_column_wrapper<int64_t> col_0_second({4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1_second({3, 2, 1});
  cudf::test::strings_column_wrapper col_2_second({"big", "random", "c++"});

  cudf::test::fixed_width_column_wrapper<int64_t> col_0({1, 2, 3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<int32_t> col_1({6, 5, 4, 3, 2, 1});
  cudf::test::strings_column_wrapper col_2({"apple", "orange", "duck", "big", "random", "c++"});

  std::vector<std::unique_ptr<cudf::column>> columns_first;
  columns_first.push_back(col_0_first.release());
  columns_first.push_back(col_1_first.release());
  columns_first.push_back(col_2_first.release());

  std::vector<std::unique_ptr<cudf::column>> columns_second;
  columns_second.push_back(col_0_second.release());
  columns_second.push_back(col_1_second.release());
  columns_second.push_back(col_2_second.release());

  std::vector<std::unique_ptr<cudf::column>> columns_combined;
  columns_combined.push_back(col_0.release());
  columns_combined.push_back(col_1.release());
  columns_combined.push_back(col_2.release());

  auto const num_columns_first = columns_first.size();
  auto test_table_first        = std::make_unique<cudf::table>(std::move(columns_first));

  auto const num_columns_second = columns_second.size();
  auto test_table_second        = std::make_unique<cudf::table>(std::move(columns_second));

  auto test_table_combined = std::make_unique<cudf::table>(std::move(columns_combined));

  std::vector<std::string> const column_names = {"col_0", "col_1", "col_2"};

  // Write the test table to disk using libcudf's Parquet writer

  cudf::io::table_input_metadata metadata_first(test_table_first->view());
  cudf::io::table_input_metadata metadata_second(test_table_second->view());
  ASSERT_EQ(metadata_first.column_metadata.size(), num_columns_first);
  ASSERT_EQ(metadata_second.column_metadata.size(), num_columns_second);
  ASSERT_EQ(column_names.size(), num_columns_first);
  ASSERT_EQ(column_names.size(), num_columns_second);

  for (std::size_t column_idx = 0; column_idx < num_columns_first; column_idx++)
    metadata_first.column_metadata[column_idx].set_name(column_names[column_idx]);

  for (std::size_t column_idx = 0; column_idx < num_columns_second; column_idx++)
    metadata_second.column_metadata[column_idx].set_name(column_names[column_idx]);

  auto filepath_first  = temp_env->get_temp_filepath("FixedDataTableFirst.parquet");
  auto filepath_second = temp_env->get_temp_filepath("FixedDataTableSecond.parquet");
  auto options_first   = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(filepath_first), test_table_first->view());
  auto options_second = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(filepath_second), test_table_second->view());
  options_first.metadata(metadata_first);
  options_second.metadata(metadata_second);
  cudf::io::write_parquet(options_first);
  cudf::io::write_parquet(options_second);

  // Load the test table from disk using a read task
  std::vector<std::string> filepaths{filepath_first, filepath_second};
  std::vector<cudf::data_type> column_types = {cudf::data_type(cudf::type_id::INT64),
                                               cudf::data_type(cudf::type_id::INT32),
                                               cudf::data_type(cudf::type_id::STRING)};

  auto read_task = std::make_unique<gqe::storage::parquet_read_task>(
    ctx_ref, 0, 0, filepaths, column_names, column_types);
  read_task->execute();
  auto result = read_task->result();

  // Compare the loaded table against the original test table
  ASSERT_EQ(result.has_value(), true);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(result.value(), test_table_combined->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.value(), test_table_combined->view());
}

TEST_F(ReadTest, FixedDataReadTaskTestPartialFilter)
{
  std::size_t num_partitions = 5;
  std::vector<int64_t> partition_vals{1, 3, 4, 5, 6};
  std::size_t num_columns_to_read = 3;

  std::vector<cudf::test::fixed_width_column_wrapper<int64_t>> col_0;
  std::vector<cudf::test::fixed_width_column_wrapper<int32_t>> col_1;
  std::vector<cudf::test::strings_column_wrapper> col_2;

  // 5 files
  col_0.push_back(cudf::test::fixed_width_column_wrapper<int64_t>({1, 1}));
  col_1.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({6, 5}));
  col_2.push_back(cudf::test::strings_column_wrapper({"apple", "orange"}));

  col_0.push_back(cudf::test::fixed_width_column_wrapper<int64_t>({3}));
  col_1.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({4}));
  col_2.push_back(cudf::test::strings_column_wrapper({"duck"}));

  col_0.push_back(cudf::test::fixed_width_column_wrapper<int64_t>({4}));
  col_1.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({3}));
  col_2.push_back(cudf::test::strings_column_wrapper({"big"}));

  col_0.push_back(cudf::test::fixed_width_column_wrapper<int64_t>({5}));
  col_1.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({2}));
  col_2.push_back(cudf::test::strings_column_wrapper({"random"}));

  col_0.push_back(cudf::test::fixed_width_column_wrapper<int64_t>({6}));
  col_1.push_back(cudf::test::fixed_width_column_wrapper<int32_t>({1}));
  col_2.push_back(cudf::test::strings_column_wrapper({"c++"}));

  std::vector<std::string> const column_names = {"col_0", "col_1", "col_2"};

  std::vector<std::string> filepaths(num_partitions);
  for (std::size_t i = 0; i < num_partitions; ++i) {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(col_0[i].release());
    columns.push_back(col_1[i].release());
    columns.push_back(col_2[i].release());
    auto test_table = std::make_unique<cudf::table>(std::move(columns));
    cudf::io::table_input_metadata metadata(test_table->view());

    ASSERT_EQ(metadata.column_metadata.size(), num_columns_to_read);
    ASSERT_EQ(column_names.size(), num_columns_to_read);

    for (std::size_t column_idx = 0; column_idx < num_columns_to_read; ++column_idx)
      metadata.column_metadata[column_idx].set_name(column_names[column_idx]);

    auto filepath = temp_env->get_temp_filepath("col_0=" + std::to_string(partition_vals[i]));
    filepaths[i]  = filepath;

    auto options =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info(filepath), test_table->view());
    options.metadata(metadata);
    cudf::io::write_parquet(options);
  }

  cudf::test::fixed_width_column_wrapper<int32_t> haystack_col({1, 3, 5});
  std::vector<std::unique_ptr<cudf::column>> haystack_columns;
  haystack_columns.push_back(haystack_col.release());

  auto const num_haystack_columns_to_read = haystack_columns.size();
  auto haystack_table = std::make_unique<cudf::table>(std::move(haystack_columns));

  std::vector<std::string> const haystack_column_names = {"haystack_col"};

  // Write the test table to disk using libcudf's Parquet writer

  cudf::io::table_input_metadata haystack_metadata(haystack_table->view());

  for (std::size_t column_idx = 0; column_idx < num_haystack_columns_to_read; column_idx++)
    haystack_metadata.column_metadata[column_idx].set_name(haystack_column_names[column_idx]);

  auto haystack_filepath = temp_env->get_temp_filepath("HaystackDataTable.parquet");
  auto haystack_options  = cudf::io::parquet_writer_options::builder(
    cudf::io::sink_info(haystack_filepath), haystack_table->view());
  haystack_options.metadata(haystack_metadata);
  cudf::io::write_parquet(haystack_options);

  // Load the haystack table from disk
  std::vector<cudf::data_type> haystack_column_types = {cudf::data_type(cudf::type_id::INT32)};
  std::vector<std::string> haystack_filepaths{haystack_filepath};
  auto haystack_task = std::make_shared<gqe::storage::parquet_read_task>(
    ctx_ref, 0, 0, haystack_filepaths, haystack_column_names, haystack_column_types);

  // Load the test table from disk using a read task
  std::vector<cudf::data_type> column_types = {cudf::data_type(cudf::type_id::INT64),
                                               cudf::data_type(cudf::type_id::INT32),
                                               cudf::data_type(cudf::type_id::STRING)};

  auto read_task = std::make_unique<gqe::storage::parquet_read_task>(
    ctx_ref,
    0,
    0,
    filepaths,
    column_names,
    std::move(column_types),
    std::make_unique<gqe::in_predicate_expression>(
      std::vector<std::shared_ptr<gqe::expression>>(
        {std::make_shared<gqe::column_reference_expression>(0)}),
      0),
    std::vector<std::shared_ptr<gqe::task>>{std::move(haystack_task)});

  read_task->execute();
  auto result = read_task->result();

  // Construct the reference result
  cudf::test::fixed_width_column_wrapper<int64_t> res_col_0({1, 1, 3, 5});
  cudf::test::fixed_width_column_wrapper<int32_t> res_col_1({6, 5, 4, 2});
  cudf::test::strings_column_wrapper res_col_2({"apple", "orange", "duck", "random"});

  std::vector<std::unique_ptr<cudf::column>> res_columns;
  res_columns.push_back(res_col_0.release());
  res_columns.push_back(res_col_1.release());
  res_columns.push_back(res_col_2.release());

  auto res_table = std::make_unique<cudf::table>(std::move(res_columns));

  ASSERT_EQ(result.has_value(), true);
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(result.value(), res_table->view());
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.value(), res_table->view());
}
