/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/read.hpp>

#include <cudf/column/column.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <memory>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

TEST(FixedDataReadTaskTest, MixTypes)
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
  options.metadata(&metadata);
  cudf::io::write_parquet(options);

  // Load the test table from disk using a read task

  auto read_task =
    std::make_unique<gqe::read_task>(0, 0, filepath, gqe::file_format_type::parquet, column_names);
  read_task->execute();
  auto result = read_task->result();

  // Compare the loaded table against the original test table

  ASSERT_EQ(result.has_value(), true);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(result.value(), test_table->view());
}
