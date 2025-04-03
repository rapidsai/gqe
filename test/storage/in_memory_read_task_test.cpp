/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <iterator>
#include <memory>

struct test_parameters {
  test_parameters(bool read_zero_copy_enable, gqe::compression_format comp_format)
    : read_zero_copy_enable(read_zero_copy_enable), comp_format(comp_format)
  {
  }

  bool read_zero_copy_enable;
  gqe::compression_format comp_format;
};

class InMemoryReadTest : public testing::TestWithParam<test_parameters> {
 public:
  InMemoryReadTest()

  {
    auto const params = GetParam();

    gqe::optimization_parameters opms(true);
    opms.read_zero_copy_enable              = params.read_zero_copy_enable;
    opms.in_memory_table_compression_format = params.comp_format;

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
    auto const comp_format        = query_ctx->parameters.in_memory_table_compression_format;
    auto const nvcomp_data_format = query_ctx->parameters.in_memory_table_compression_data_type;
    auto const chunk_size         = query_ctx->parameters.compression_chunk_size;

    std::vector<std::unique_ptr<gqe::storage::column_base>> columns;
    std::transform(test_columns.cbegin(),
                   test_columns.cend(),
                   std::back_inserter(columns),
                   [comp_format, nvcomp_data_format, chunk_size](
                     auto const& col) -> std::unique_ptr<gqe::storage::column_base> {
                     if (comp_format == gqe::compression_format::none) {
                       return std::make_unique<gqe::storage::contiguous_column>(cudf::column(*col));
                     } else {
                       return std::make_unique<gqe::storage::compressed_column>(
                         cudf::column(*col),
                         comp_format,
                         rmm::cuda_stream_default,
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
                  test_parameters{false, gqe::compression_format::best_decompression_speed}));