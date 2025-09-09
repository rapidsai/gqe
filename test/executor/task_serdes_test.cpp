/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "utilities.hpp"

#include <gqe/communicator.hpp>
#include <gqe/context_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <gtest/gtest.h>

#include <proto/task.pb.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

class SerdesTest : public ::testing::Test {
 protected:
  SerdesTest()
    : task_manager_ctx{gqe::multi_process_task_manager_context::default_init(MPI_COMM_WORLD)},
      query_ctx(gqe::optimization_parameters(true)),
      ctx_ref{task_manager_ctx.get(), &query_ctx}
  {
  }
  void TearDown() override { task_manager_ctx->finalize(); }

  std::unique_ptr<gqe::multi_process_task_manager_context> task_manager_ctx;
  gqe::query_context query_ctx;
  gqe::context_reference ctx_ref;
};

TEST_F(SerdesTest, FixedWidthColumns)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col_wrap_0({0, 1, 2, 3, 4, 5},
                                                             {1, 0, 1, 0, 1, 0});
  cudf::test::fixed_width_column_wrapper<float> col_wrap_1({0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(col_wrap_0.release());
  columns.push_back(col_wrap_1.release());
  auto table = std::make_unique<cudf::table>(std::move(columns));

  auto& col_0 = table->get_column(0);
  auto& col_1 = table->get_column(1);

  auto col_0_data_location      = reinterpret_cast<uint64_t>(col_0.view().data<std::byte>());
  auto col_0_null_mask_location = reinterpret_cast<uint64_t>(col_0.view().null_mask());
  auto col_1_data_location      = reinterpret_cast<uint64_t>(col_1.view().data<std::byte>());
  auto col_1_null_mask_location = reinterpret_cast<uint64_t>(col_1.view().null_mask());
  gqe::test::executed_task task(ctx_ref, 0, 0, std::move(table));

  auto metadata = task_manager_ctx->migration_service->to_proto(task);

  EXPECT_EQ(metadata.id().id(), "0");
  EXPECT_EQ(metadata.status(), proto::TaskStatus::finished);
  EXPECT_EQ(metadata.result().table().num_rows(), 6);
  EXPECT_EQ(metadata.result().table().columns_size(), 2);
  EXPECT_EQ(metadata.result().table().columns(0).type(), proto::DataType::INT64);
  EXPECT_EQ(metadata.result().table().columns(0).data_location(), col_0_data_location);
  EXPECT_EQ(metadata.result().table().columns(0).null_mask_location(), col_0_null_mask_location);
  EXPECT_EQ(metadata.result().table().columns(0).null_count(), 3);
  EXPECT_EQ(metadata.result().table().columns(1).type(), proto::DataType::FLOAT32);
  EXPECT_EQ(metadata.result().table().columns(1).data_location(), col_1_data_location);
  EXPECT_EQ(metadata.result().table().columns(1).null_mask_location(), col_1_null_mask_location);
  EXPECT_EQ(metadata.result().table().columns(1).null_count(), 0);
}

TEST_F(SerdesTest, StringColumn)
{
  std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
  cudf::test::strings_column_wrapper strings_column(strings.begin(), strings.end());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(strings_column.release());
  auto table            = std::make_unique<cudf::table>(std::move(columns));
  auto& string_col      = table->get_column(0);
  auto chars_location   = reinterpret_cast<uint64_t>(string_col.view().data<std::byte>());
  auto offsets_location = reinterpret_cast<uint64_t>(string_col.view().child(0).data<std::byte>());
  gqe::test::executed_task task(ctx_ref, 0, 0, std::move(table));

  auto metadata = task_manager_ctx->migration_service->to_proto(task);
  EXPECT_EQ(metadata.id().id(), "0");
  EXPECT_EQ(metadata.status(), proto::TaskStatus::finished);
  EXPECT_EQ(metadata.result().table().num_rows(), 7);
  EXPECT_EQ(metadata.result().table().columns_size(), 1);
  EXPECT_EQ(metadata.result().table().columns(0).type(), proto::DataType::STRING);
  EXPECT_EQ(metadata.result().table().columns(0).children_size(), 1);
  EXPECT_EQ(metadata.result().table().columns(0).data_bytes(), 22);
  EXPECT_EQ(metadata.result().table().columns(0).data_location(), chars_location);
  EXPECT_EQ(metadata.result().table().columns(0).children(0).data_location(), offsets_location);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  GQE_MPI_TRY(MPI_Init(NULL, NULL));

  int result = RUN_ALL_TESTS();

  GQE_MPI_TRY(MPI_Finalize());
  return result;
}