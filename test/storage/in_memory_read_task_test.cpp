/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/context_reference.hpp>
#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <gqe/expression/column_reference.hpp>
#include <gqe/expression/literal.hpp>
#include <iterator>
#include <memory>

struct test_parameters {
  test_parameters(bool read_zero_copy_enable,
                  gqe::compression_format comp_format,
                  bool use_overlap_mtx = false)
    : read_zero_copy_enable(read_zero_copy_enable),
      comp_format(comp_format),
      use_overlap_mtx(use_overlap_mtx)
  {
  }

  bool read_zero_copy_enable;
  gqe::compression_format comp_format;
  bool use_overlap_mtx;
};

class InMemoryReadTest : public testing::TestWithParam<test_parameters> {
 public:
  InMemoryReadTest()

  {
    auto const params = GetParam();

    gqe::optimization_parameters opms(true);
    opms.read_zero_copy_enable              = params.read_zero_copy_enable;
    opms.in_memory_table_compression_format = params.comp_format;
    opms.use_overlap_mtx                    = params.use_overlap_mtx;

    query_ctx        = std::make_unique<gqe::query_context>(opms);
    task_manager_ctx = std::make_unique<gqe::task_manager_context>();
  }

  void SetUp() override
  {
    // Setup data
    cudf::test::fixed_width_column_wrapper<int32_t> col_0({6, 5, 4, 3, 2, 1});
    cudf::test::fixed_width_column_wrapper<float> col_1({1.0, 5.0, 3.0, 9.0, 7.0, 0.0});
    std::vector<std::unique_ptr<cudf::column>> test_columns;
    test_columns.push_back(col_0.release());
    test_columns.push_back(col_1.release());

    // Setup row group
    auto comp_format        = query_ctx->parameters.in_memory_table_compression_format;
    auto nvcomp_data_format = query_ctx->parameters.in_memory_table_compression_data_type;
    auto const chunk_size   = query_ctx->parameters.compression_chunk_size;

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    std::transform(
      test_columns.cbegin(),
      test_columns.cend(),
      std::back_inserter(columns),
      [comp_format, nvcomp_data_format, chunk_size](
        auto const& col) mutable -> std::unique_ptr<gqe::storage::column_base> {
        if (comp_format == gqe::compression_format::none) {
          return std::make_unique<gqe::storage::contiguous_column>(cudf::column(*col));
        } else {
          auto cudf_col = cudf::column(*col);
          auto dtype    = cudf_col.type().id();
          if ((comp_format == gqe::compression_format::best_compression_ratio) or
              (comp_format == gqe::compression_format::best_decompression_speed)) {
            best_compression_config(
              dtype,
              comp_format,
              nvcomp_data_format,
              (comp_format == gqe::compression_format::best_compression_ratio ? 0 : 1));
          }
          return std::make_unique<gqe::storage::compressed_column>(
            std::move(cudf_col),
            comp_format,
            cudf::get_default_stream(),
            rmm::mr::get_current_device_resource(),
            nvcomp_data_format,
            chunk_size);
        }
      });
    gqe::storage::row_group row_group(std::move(columns));

    // Setup test table
    test_table = std::make_unique<cudf::table>(std::move(test_columns));

    // Setup table types
    std::vector<std::string> col_names     = {"int", "float"};
    std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT32),
                                              cudf::data_type(cudf::type_id::FLOAT32)};

    // Setup GQE table
    table = std::make_unique<gqe::storage::in_memory_table>(
      gqe::memory_kind::device{rmm::cuda_device_id(0)}, col_names, col_types);

    // Add row groups to table
    table->get_row_group_appender()(std::move(row_group));
  }

  void TearDown() override { table = nullptr; }

  std::unique_ptr<gqe::task_manager_context> task_manager_ctx;
  std::unique_ptr<gqe::query_context> query_ctx;
  std::unique_ptr<gqe::storage::in_memory_table> table;
  std::unique_ptr<cudf::table> test_table;
};

TEST_P(InMemoryReadTest, ReadFirstCol)
{
  constexpr int32_t stage_id             = 0;
  const std::vector<int32_t> task_ids    = {0};
  std::vector<std::string> col_names     = {"int"};
  std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT32)};

  std::vector<gqe::storage::in_memory_readable_view::task_parameters> task_parameters;
  std::transform(
    task_ids.cbegin(),
    task_ids.cend(),
    std::back_inserter(task_parameters),
    [](auto id) -> gqe::storage::in_memory_readable_view::task_parameters { return {id}; });

  gqe::context_reference ctx_ref{task_manager_ctx.get(), query_ctx.get()};
  auto tasks = table->readable_view()->get_read_tasks(
    std::move(task_parameters), ctx_ref, stage_id, col_names, col_types);

  EXPECT_EQ(tasks.size(), 1);
  ASSERT_FALSE(tasks.empty());

  auto& task = tasks.at(0);
  task->execute();
  auto result = task->result();

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->column(0), test_table->get_column(0));
}

TEST_P(InMemoryReadTest, ReadSecondCol)
{
  constexpr int32_t stage_id             = 0;
  const std::vector<int32_t> task_ids    = {0};
  std::vector<std::string> col_names     = {"float"};
  std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::FLOAT32)};

  std::vector<gqe::storage::in_memory_readable_view::task_parameters> task_parameters;
  std::transform(
    task_ids.cbegin(),
    task_ids.cend(),
    std::back_inserter(task_parameters),
    [](auto id) -> gqe::storage::in_memory_readable_view::task_parameters { return {id}; });

  gqe::context_reference ctx_ref{task_manager_ctx.get(), query_ctx.get()};
  auto tasks = table->readable_view()->get_read_tasks(
    std::move(task_parameters), ctx_ref, stage_id, col_names, col_types);

  EXPECT_EQ(tasks.size(), 1);
  ASSERT_FALSE(tasks.empty());

  auto& task = tasks.at(0);
  task->execute();
  auto result = task->result();

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(result->num_columns(), 1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->column(0), test_table->get_column(1));
}

TEST_P(InMemoryReadTest, ReadAll)
{
  constexpr int32_t stage_id             = 0;
  const std::vector<int32_t> task_ids    = {0};
  std::vector<std::string> col_names     = {"int", "float"};
  std::vector<cudf::data_type> col_types = {cudf::data_type(cudf::type_id::INT32),
                                            cudf::data_type(cudf::type_id::FLOAT32)};

  std::vector<gqe::storage::in_memory_readable_view::task_parameters> task_parameters;
  std::transform(
    task_ids.cbegin(),
    task_ids.cend(),
    std::back_inserter(task_parameters),
    [](auto id) -> gqe::storage::in_memory_readable_view::task_parameters { return {id}; });

  gqe::context_reference ctx_ref{task_manager_ctx.get(), query_ctx.get()};
  auto tasks = table->readable_view()->get_read_tasks(
    std::move(task_parameters), ctx_ref, stage_id, col_names, col_types);

  EXPECT_EQ(tasks.size(), 1);
  ASSERT_FALSE(tasks.empty());

  auto& task = tasks.at(0);
  task->execute();
  auto result = task->result();

  ASSERT_TRUE(result.has_value());
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result, test_table->view());
}

INSTANTIATE_TEST_SUITE_P(
  ZeroCopyOnOff,
  InMemoryReadTest,
  testing::Values(test_parameters{true, gqe::compression_format::none},
                  test_parameters{false, gqe::compression_format::none},
                  test_parameters{false, gqe::compression_format::ans},
                  test_parameters{false, gqe::compression_format::lz4},
                  test_parameters{false, gqe::compression_format::snappy},
                  test_parameters{false, gqe::compression_format::gdeflate},
                  test_parameters{false, gqe::compression_format::deflate},
                  test_parameters{false, gqe::compression_format::cascaded},
                  test_parameters{false, gqe::compression_format::zstd},
                  test_parameters{false, gqe::compression_format::bitcomp},
                  test_parameters{false, gqe::compression_format::best_compression_ratio},
                  test_parameters{false, gqe::compression_format::best_decompression_speed},
                  test_parameters{false, gqe::compression_format::ans, true},
                  test_parameters{false, gqe::compression_format::none, true}));

// Test fixture for in_memory_read_task::execute
//
// The fixture creates four input tables. Each table has four columns, two integer columns, and two
// double columns. In the second column of the integer and double columns, every other element is a
// null value. The four tables have 100 rows each and the values over all columns go from 0 to 399.
//
// There are convenience methods to create row groups from one or more input tables, to create a
// range filter that can be used as a partial filter, to create comparison tables, and to create a
// read_task to test.
//
// Tests can change parameters of the read task by manipulating _task_manager_ctx and _query_ctx
// before creating the read task, e.g., to compress the row group columns.
class InMemoryReadTaskTest : public ::testing::Test {
 protected:
  InMemoryReadTaskTest()
    : _task_manager_ctx(std::make_unique<gqe::task_manager_context>()),
      _query_ctx(std::make_unique<gqe::query_context>(gqe::optimization_parameters{true})),
      _ctx_ref(gqe::context_reference{_task_manager_ctx.get(), _query_ctx.get()}),
      _task_id(0),
      _stage_id(0),
      _memory_kind(gqe::memory_kind::device{rmm::cuda_device_id(0)}),
      _memory_resource(std::make_unique<rmm::mr::cuda_memory_resource>())
  {
    // Explicitly enable pruning, in case it is disabled by default.
    _query_ctx->parameters.use_partition_pruning = true;
    // Create the four input tables, the corresponding column indexes, and the data types.
    for (size_t i = 0; i < NUM_TABLES; ++i) {
      auto table = create_input_table(NUM_ROWS, i * NUM_ROWS);
      _input_tables.push_back(table->view());
      _tables.push_back(std::move(table));
    }
    _column_indexes = {0, 1, 2, 3};
    _data_types     = {cudf::data_type(cudf::type_id::INT32),
                       cudf::data_type(cudf::type_id::INT32),
                       cudf::data_type(cudf::type_id::FLOAT32),
                       cudf::data_type(cudf::type_id::FLOAT32)};
  }

  // Create a typed input column with optional null values
  template <typename T>
  cudf::test::fixed_width_column_wrapper<T> create_fixed_width_column_wrapper(
    cudf::size_type num_rows, cudf::size_type offset, bool odds_are_null = false)
  {
    std::vector<T> values(num_rows);
    std::iota(values.begin(), values.end(), static_cast<T>(offset));
    if (odds_are_null) {
      std::vector<bool> is_null(num_rows);
      std::transform(values.begin(), values.end(), is_null.begin(), [](auto i) {
        return static_cast<int32_t>(i) % 2 == 0;
      });
      cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end(), is_null.begin());
      return col;
    } else {
      cudf::test::fixed_width_column_wrapper<T> col(values.begin(), values.end());
      return col;
    }
  }

  // Create a cudf::table consisting of 2 integer columns (one without nulls and one with nulls) and
  // 2 float columns (same).
  std::unique_ptr<cudf::table> create_input_table(cudf::size_type num_rows, cudf::size_type offset)
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(create_fixed_width_column_wrapper<int32_t>(num_rows, offset).release());
    columns.push_back(create_fixed_width_column_wrapper<int32_t>(num_rows, offset, true).release());
    columns.push_back(create_fixed_width_column_wrapper<double>(num_rows, offset).release());
    columns.push_back(create_fixed_width_column_wrapper<double>(num_rows, offset, true).release());
    return std::make_unique<cudf::table>(
      std::move(columns), cudf::get_default_stream(), _memory_resource.get());
  }

  // Create a range filter start <= column < end
  template <typename T>
  std::unique_ptr<gqe::expression> create_range_filter(cudf::size_type column_index,
                                                       T start_inclusive,
                                                       T end_exclusive)
  {
    return std::make_unique<gqe::logical_and_expression>(
      std::make_unique<gqe::less_equal_expression>(
        std::make_unique<gqe::literal_expression<T>>(start_inclusive),
        std::make_unique<gqe::column_reference_expression>(column_index)),
      std::make_unique<gqe::less_expression>(
        std::make_unique<gqe::column_reference_expression>(column_index),
        std::make_unique<gqe::literal_expression<T>>(end_exclusive)));
  }

  // Create row groups from input table_views
  std::vector<const gqe::storage::row_group*> create_row_groups(
    std::vector<cudf::table_view>::iterator begin, std::vector<cudf::table_view>::iterator end)
  {
    std::vector<const gqe::storage::row_group*> result;
    std::transform(
      begin, end, std::back_inserter(result), [&](const cudf::table_view& input_table) {
        auto const comp_format = _query_ctx->parameters.in_memory_table_compression_format;
        auto const nvcomp_data_format =
          _query_ctx->parameters.in_memory_table_compression_data_type;
        auto const chunk_size = _query_ctx->parameters.compression_chunk_size;
        std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
        std::transform(
          input_table.begin(),
          input_table.end(),
          std::back_inserter(columns),
          [comp_format, nvcomp_data_format, chunk_size](
            const cudf::column_view& column_view) -> std::unique_ptr<gqe::storage::column_base> {
            if (comp_format == gqe::compression_format::none) {
              return std::make_unique<gqe::storage::contiguous_column>(cudf::column(column_view));
            } else {
              return std::make_unique<gqe::storage::compressed_column>(
                cudf::column(column_view),
                comp_format,
                rmm::cuda_stream_default,
                rmm::mr::get_current_device_resource(),
                nvcomp_data_format,
                chunk_size);
            }
          });
        auto zone_map = std::make_unique<gqe::zone_map>(input_table, PARTITION_SIZE);
        auto row_group =
          std::make_unique<gqe::storage::row_group>(std::move(columns), std::move(zone_map));
        auto row_group_ptr = row_group.get();
        _row_groups.push_back(std::move(row_group));
        // The pointer cannot escape the function because it is stored in _row_groups
        return row_group_ptr;
      });
    return result;
  }

  // Convenience method to create an in_memory_read_task, so that the tests don't have to pass
  // parameters that do not affect the test result, e.g., the task_id.
  storage::in_memory_read_task create_read_task(
    std::vector<const storage::row_group*> row_groups,
    std::vector<cudf::size_type> column_indexes,
    std::vector<cudf::data_type> data_types,
    std::unique_ptr<gqe::expression> zone_map_filter = nullptr)
  {
    auto partial_filter = zone_map_filter ? zone_map_filter->clone() : nullptr;
    return storage::in_memory_read_task(_ctx_ref,
                                        _task_id,
                                        _stage_id,
                                        row_groups,
                                        column_indexes,
                                        data_types,
                                        _memory_kind,
                                        std::move(partial_filter));
  }

  // Number of rows per input table
  static constexpr cudf::size_type NUM_ROWS = 100;

  // Number of rows per zone map partition
  static constexpr cudf::size_type PARTITION_SIZE = 5;

  // Number of input tables
  static constexpr size_t NUM_TABLES = 4;

  // Parameters that are required by the constructor of in_memory_read_task but don't really change
  // across the tests. The only exception is _query_ctx.parameters, which can be manipulated to
  // enable/disable zero-copy reads and compression.
  const std::unique_ptr<gqe::task_manager_context> _task_manager_ctx;
  const std::unique_ptr<gqe::query_context> _query_ctx;
  const context_reference _ctx_ref;
  const int32_t _task_id;
  const int32_t _stage_id;
  const memory_kind::type _memory_kind;

  // The input tables need be accessible on the GPU, otherwise cudf::concatenate does not process
  // null counts correctly. The constructor sets this to CUDA device memory.
  std::unique_ptr<rmm::mr::device_memory_resource> _memory_resource;

  // Parameters that influence the input (i.e., input table views from which row groups are
  // constructed), and the result of in_memory_read_task::execute (i.e., the requested columns and
  // datatypes)
  std::vector<cudf::table_view> _input_tables;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;

  // Input tables created in the constructor need to be kept valid during the lifetime of the test
  std::vector<std::unique_ptr<cudf::table>> _tables;

  // Row groups that are created by create_row_groups need to be kept valid during the lifetime of
  // the test
  std::vector<std::unique_ptr<gqe::storage::row_group>> _row_groups;
};

TEST_F(InMemoryReadTaskTest, returnSingleRowGroupDirectly)
{
  // Create a task with a single (!) row group
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  storage::in_memory_read_task task = create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result contains the contents of the row group and is passed as a borrowed result
  ASSERT_TRUE(task.result().has_value());
  EXPECT_FALSE(*task.is_result_owned());
  cudf::table_view expected = _input_tables[0];
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected);
}

TEST_F(InMemoryReadTaskTest, copySingleRowGroupIfZeroCopyIsDisabled)
{
  // Disable zero-copying
  _query_ctx->parameters.read_zero_copy_enable = false;
  // Create a task with a single row group
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.begin() + 1);
  storage::in_memory_read_task task = create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result contains the contents of the row group and is passed as an owned result
  ASSERT_TRUE(task.result().has_value());
  EXPECT_TRUE(*task.is_result_owned());
  cudf::table_view expected = _input_tables[0];
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected);
}

TEST_F(InMemoryReadTaskTest, concatenateMultipleRowGroups)
{
  // Create a task with multiple row groups
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  storage::in_memory_read_task task = create_read_task(row_groups, _column_indexes, _data_types);

  // Execute the read task
  task.execute();

  // The result contains the contents of the row group and is passed as a borrowed result
  ASSERT_TRUE(task.result().has_value());
  EXPECT_TRUE(*task.is_result_owned());
  auto expected = cudf::concatenate(_input_tables);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, returnEmptyTableIfEntireInputIsPruned)
{
  // Create a zone map filter that exclude all inputs. If there are N input tables, column_0 > N *
  // NUM_ROWS should filter out all rows.
  std::unique_ptr<gqe::expression> partial_filter = std::make_unique<gqe::greater_expression>(
    std::make_unique<gqe::column_reference_expression>(0),
    std::make_unique<gqe::literal_expression<int32_t>>(_input_tables.size() * NUM_ROWS));
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with row groups (actual number should not matter), and pass the filter
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result is empty
  ASSERT_TRUE(task.result().has_value());
  EXPECT_EQ(task.result()->num_rows(), 0);
}

TEST_F(InMemoryReadTaskTest, returnSingleRowGroupAfterPruningDirectly)
{
  // Create a zone map filter 148 <= col0 < 157. This should return three partitions, from 145
  // (inclusive) to 160 (exclusive), i.e., 15 rows in total. The rows are contained in the second
  // input table row group.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter<int32_t>(0, 148, 157);
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as a borrowed
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_FALSE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 15;
  constexpr cudf::size_type start_offset_in_result = 145;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, concatenatePartitionsInMultipleRowGroups)
{
  // Create a zone map filter 198 <= col0 < 207. This should return three partitions, from 195
  // (inclusive) to 210 (exclusive), i.e., 15 rows in total. The first partition is contained in the
  // second row group and the other partitions in the third row group.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter<int32_t>(0, 198, 207);
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple row groups, and pass the filter. The filter will prune all but
  // two consecutive row groups.
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the two row groups and is passed as an
  // owned result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 15;
  constexpr cudf::size_type start_offset_in_result = 195;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, concatenateDiscontiguousPartitionsInSingleRowGroup)
{
  // Create a zone map filter 118 <= col0 < 127 OR 168.0 <= col2 <= 177.0 (the different columns
  // don't really matter). This should return six partitions: three from 115 (inclusive) to 130
  // (exclusive) and three from 165 (inclusive) to 180 (exclusive). Both are contained in the second
  // row group.
  std::unique_ptr<gqe::expression> partial_filter = std::make_unique<gqe::logical_or_expression>(
    create_range_filter<int32_t>(0, 118, 127), create_range_filter<double>(2, 165, 180));
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple row groups, and pass the filter. The filter will prune all but
  // one row group.
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying discontiguous partitions of the single row group and is
  // passed as an owned result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result       = 15;
  constexpr cudf::size_type start_offset_in_result_1 = 115;
  constexpr cudf::size_type start_offset_in_result_2 = 165;
  auto expected1 = create_input_table(num_rows_in_result, start_offset_in_result_1);
  auto expected2 = create_input_table(num_rows_in_result, start_offset_in_result_2);
  auto expected =
    cudf::concatenate(std::vector<cudf::table_view>{expected1->view(), expected2->view()});
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}

TEST_F(InMemoryReadTaskTest, concatenateSingleRowGroupIfCompressed)
{
  // Compress input dat
  _query_ctx->parameters.in_memory_table_compression_format = gqe::compression_format::ans;
  // Create a zone map filter 148 <= col0 < 157, which returns 3 partitions with 15 rows.
  std::unique_ptr<gqe::expression> partial_filter = create_range_filter(0, 148, 157);
  auto zone_map_filter = zone_map_expression_transformer::transform(*partial_filter);
  // Create a task with multiple (!) row groups, and pass the filter. The filter will prune all but
  // one row groups.
  std::vector<const storage::row_group*> row_groups =
    create_row_groups(_input_tables.begin(), _input_tables.end());
  storage::in_memory_read_task task =
    create_read_task(row_groups, _column_indexes, _data_types, std::move(zone_map_filter));

  // Execute the read task
  task.execute();

  // The result only contains the qualifying partitions of the row group and is passed as an owned
  // result
  ASSERT_TRUE(task.result().has_value());
  ASSERT_TRUE(*task.is_result_owned());
  constexpr cudf::size_type num_rows_in_result     = 15;
  constexpr cudf::size_type start_offset_in_result = 145;
  auto expected = create_input_table(num_rows_in_result, start_offset_in_result);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*task.result(), expected->view());
}
