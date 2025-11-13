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

#include "../utility.hpp"

#include <gqe/compiler/Conversion/Passes.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/CudfTableMaterialize.hpp>
#include <gqe/compiler/Dialect/GPU/Pipelines/Passes.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>
#include <gqe/compiler/Dialect/RelAlg/Pipelines/Passes.hpp>
#include <gqe/compiler/Tools/DialectRegistry.hpp>
#include <gqe/compiler/Tools/GPUModuleSerialization.hpp>
#include <gqe/utility/cuda_driver.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/kernel_arguments_builder.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * Test fixture for CudfTableMaterialize tests
 */
class CudfTableMaterializeTest : public testing::Test {
 protected:
  void SetUp() override
  {
    // Create a registry.
    registry = std::make_unique<mlir::DialectRegistry>();

    // Register commonly used dialect translations.
    gqe::compiler::tools::registerLLVMIRTranslations(*registry);

    // Create the Context and load the relevant dialects into it.
    context = std::make_unique<mlir::MLIRContext>(*registry);

    // Load required dialects
    context->loadDialect<gqe::compiler::relalg::RelAlgDialect>();
    context->loadDialect<mlir::cf::ControlFlowDialect>();
    context->loadDialect<mlir::gpu::GPUDialect>();
    context->loadDialect<mlir::index::IndexDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::scf::SCFDialect>();

    // Create an op builder.
    builder = std::make_unique<mlir::OpBuilder>(context.get());

    // Create a pass manager.
    pm = std::make_unique<mlir::PassManager>(context.get());

    // Set the device and initialize CUDA context.
    cudaSetDevice(deviceId);

    // Get GPU architecture and configure NVVM lowering pipeline.
    auto gpuArchOrdinal        = gqe::utility::detectDeviceArchitecture(deviceId);
    gpuArch                    = (llvm::Twine("sm_") + llvm::Twine(gpuArchOrdinal)).str();
    pipelineOptions.targetArch = gpuArch;
  }

  // CUDA device ID.
  static constexpr int32_t deviceId = 0;

  std::unique_ptr<mlir::DialectRegistry> registry;
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::PassManager> pm;
  std::string gpuArch;
  gqe::compiler::gpu::GPUToNVVMPipelineOptions pipelineOptions;
};

/**
 * Helper function to copy column data from GPU to CPU and convert to int32 for comparison
 */
template <typename T>
std::vector<int32_t> copy_and_convert_column(const cudf::column_view& column, int scale_factor = 1)
{
  std::vector<T> raw_data(column.size());
  CUDF_EXPECTS(
    cudaMemcpy(raw_data.data(), column.data<T>(), column.size() * sizeof(T), cudaMemcpyDefault) ==
      cudaSuccess,
    "Failed to copy column data from GPU");

  std::vector<int32_t> converted_data;
  converted_data.reserve(raw_data.size());

  if constexpr (std::is_same_v<T, double>) {
    for (const auto& value : raw_data) {
      converted_data.push_back(static_cast<int32_t>(value * scale_factor));
    }
  } else {
    for (const auto& value : raw_data) {
      converted_data.push_back(static_cast<int32_t>(value));
    }
  }

  return converted_data;
}

/**
 * Generic helper to create a row hash for set comparison
 */
std::vector<int32_t> extract_table_data(const cudf::table_view& table)
{
  // Pre-allocate space for all data (row-major order)
  std::vector<int32_t> all_data(table.num_rows() * table.num_columns());

  for (int col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
    auto column = table.column(col_idx);
    std::vector<int32_t> column_data;

    switch (column.type().id()) {
      case cudf::type_id::INT32:
      case cudf::type_id::TIMESTAMP_DAYS:
        column_data = copy_and_convert_column<int32_t>(column);
        break;
      case cudf::type_id::FLOAT64:
        column_data = copy_and_convert_column<double>(column, 100);  // Scale for precision
        break;
      default: throw std::runtime_error("Unsupported column type for row set comparison");
    }

    // Store column data in row-major order
    for (cudf::size_type row = 0; row < table.num_rows(); ++row) {
      all_data[row * table.num_columns() + col_idx] = column_data[row];
    }
  }

  return all_data;
}

/**
 * Helper function to create row tuples for set comparison
 */
template <size_t N>
std::set<std::array<int32_t, N>> create_row_set(const std::vector<int32_t>& data,
                                                cudf::size_type num_rows)
{
  std::set<std::array<int32_t, N>> row_set;

  for (cudf::size_type row = 0; row < num_rows; ++row) {
    std::array<int32_t, N> row_array;
    for (size_t col = 0; col < N; ++col) {
      row_array[col] = data[row * N + col];
    }
    row_set.insert(row_array);
  }

  return row_set;
}

/**
 * Helper function to verify that two tables contain exactly the same rows (order-agnostic)
 * Uses set-based comparison to ensure each row appears exactly once in both tables
 */
bool verify_same_row_sets(const cudf::table_view& table1,
                          const cudf::table_view& table2,
                          const std::vector<std::string>& column_names = {})
{
  if (table1.num_rows() != table2.num_rows()) {
    llvm::outs() << "Row count mismatch: " << table1.num_rows() << " vs " << table2.num_rows()
                 << "\n";
    return false;
  }

  if (table1.num_columns() != table2.num_columns()) {
    llvm::outs() << "Column count mismatch: " << table1.num_columns() << " vs "
                 << table2.num_columns() << "\n";
    return false;
  }

  // Check if all column types are supported
  for (int col_idx = 0; col_idx < table1.num_columns(); ++col_idx) {
    auto type1 = table1.column(col_idx).type().id();
    auto type2 = table2.column(col_idx).type().id();

    if (type1 != type2) {
      llvm::outs() << "Column " << col_idx << " type mismatch: " << static_cast<int>(type1)
                   << " vs " << static_cast<int>(type2) << "\n";
      return false;
    }

    if (type1 != cudf::type_id::INT32 && type1 != cudf::type_id::FLOAT64 &&
        type1 != cudf::type_id::TIMESTAMP_DAYS) {
      llvm::outs() << "Unsupported column type for row set comparison: " << static_cast<int>(type1)
                   << "\n";
      return false;
    }
  }

  try {
    // Extract all data from both tables
    auto data1 = extract_table_data(table1);
    auto data2 = extract_table_data(table2);

    // Create row sets based on number of columns
    bool sets_equal = false;
    switch (table1.num_columns()) {
      case 1: {
        auto set1  = create_row_set<1>(data1, table1.num_rows());
        auto set2  = create_row_set<1>(data2, table2.num_rows());
        sets_equal = (set1 == set2);
        if (!sets_equal) {
          llvm::outs() << "Row sets are different!\n";
          llvm::outs() << "Set1 size: " << set1.size() << ", Set2 size: " << set2.size() << "\n";
        }
        break;
      }
      case 2: {
        auto set1  = create_row_set<2>(data1, table1.num_rows());
        auto set2  = create_row_set<2>(data2, table2.num_rows());
        sets_equal = (set1 == set2);
        if (!sets_equal) {
          llvm::outs() << "Row sets are different!\n";
          llvm::outs() << "Set1 size: " << set1.size() << ", Set2 size: " << set2.size() << "\n";
        }
        break;
      }
      case 3: {
        auto set1  = create_row_set<3>(data1, table1.num_rows());
        auto set2  = create_row_set<3>(data2, table2.num_rows());
        sets_equal = (set1 == set2);
        if (!sets_equal) {
          llvm::outs() << "Row sets are different!\n";
          llvm::outs() << "Set1 size: " << set1.size() << ", Set2 size: " << set2.size() << "\n";
        }
        break;
      }
      case 4: {
        auto set1  = create_row_set<4>(data1, table1.num_rows());
        auto set2  = create_row_set<4>(data2, table2.num_rows());
        sets_equal = (set1 == set2);
        if (!sets_equal) {
          llvm::outs() << "Row sets are different!\n";
          llvm::outs() << "Set1 size: " << set1.size() << ", Set2 size: " << set2.size() << "\n";
        }
        break;
      }
      default:
        // For tables with more than 4 columns, not supported
        llvm::outs() << "Tables with " << table1.num_columns()
                     << " columns not supported by generic row set comparison\n";
        return false;
    }

    return sets_equal;

  } catch (const std::exception& e) {
    llvm::outs() << "Error in row set comparison: " << e.what() << "\n";
    return false;
  }
}

/**
 * Build a simple MLIR module that reads from a cuDF table and writes to another cuDF table
 */
mlir::ModuleOp buildCudfTableMaterializeModule(mlir::OpBuilder& builder)
{
  auto wrapper_module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapper_module.getBody());

  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create input table scan
  llvm::SmallVector<mlir::Attribute, 2> input_columns = {
    gqe::compiler::relalg::ColumnDefAttr::get(builder.getContext(), "id", builder.getI32Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(builder.getContext(), "value", builder.getF64Type())};
  auto input_columns_attr = builder.getArrayAttr(input_columns);
  auto table_scan_op      = builder.create<gqe::compiler::relalg::CudfTableScanOp>(
    builder.getUnknownLoc(), builder.getStringAttr("input_table"), input_columns_attr);

  // Create output table materialize
  llvm::SmallVector<mlir::Attribute, 2> output_columns = {
    gqe::compiler::relalg::ColumnDefAttr::get(builder.getContext(), "id", builder.getI32Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(builder.getContext(), "value", builder.getF64Type())};
  auto output_columns_attr = builder.getArrayAttr(output_columns);
  builder.create<gqe::compiler::relalg::CudfTableMaterializeOp>(
    builder.getUnknownLoc(),
    table_scan_op.getResult(),
    builder.getStringAttr("output_table"),
    output_columns_attr);

  return wrapper_module;
}

/**
 * Build MLIR module for lineitem table materialize with proper column types
 */
mlir::ModuleOp buildLineitemTableMaterializeModule(mlir::OpBuilder& builder)
{
  auto wrapper_module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapper_module.getBody());

  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create lineitem table scan with proper column types
  llvm::SmallVector<mlir::Attribute, 4> input_columns = {
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_shipdate", builder.getI32Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_discount", builder.getF64Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_quantity", builder.getF64Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_extendedprice", builder.getF64Type())};
  auto input_columns_attr = builder.getArrayAttr(input_columns);
  auto table_scan_op      = builder.create<gqe::compiler::relalg::CudfTableScanOp>(
    builder.getUnknownLoc(), builder.getStringAttr("lineitem"), input_columns_attr);

  // Create output table materialize with same columns
  llvm::SmallVector<mlir::Attribute, 4> output_columns = {
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_shipdate", builder.getI32Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_discount", builder.getF64Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_quantity", builder.getF64Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_extendedprice", builder.getF64Type())};
  auto output_columns_attr = builder.getArrayAttr(output_columns);
  builder.create<gqe::compiler::relalg::CudfTableMaterializeOp>(
    builder.getUnknownLoc(),
    table_scan_op.getResult(),
    builder.getStringAttr("lineitem_output"),
    output_columns_attr);

  return wrapper_module;
}

/**
 * @brief Test 1: Basic compilation pipeline
 * Verifies that CudfTableMaterialize can be compiled from RelAlg → SCF → GPU binary
 */
TEST_F(CudfTableMaterializeTest, CompilationPipeline)
{
  // Build MLIR module with CudfTableMaterialize
  auto mlir_module = buildCudfTableMaterializeModule(*builder);

  // Test RelAlg → SCF conversion
  gqe::compiler::relalg::buildLowerToSCFPassPipeline(*pm);
  auto relalgResult = pm->run(mlir_module);
  ASSERT_TRUE(mlir::succeeded(relalgResult)) << "RelAlg to SCF conversion failed";

  // Test GPU compilation (using fixture's pipeline options)
  gqe::compiler::gpu::buildLowerToNvvmPassPipeline(*pm, pipelineOptions);
  auto gpuResult = pm->run(mlir_module);
  ASSERT_TRUE(mlir::succeeded(gpuResult)) << "GPU compilation failed";

  // Serialize to cubin
  constexpr auto targetFormat = mlir::gpu::CompilationTarget::Binary;
  auto kernelBinary           = gqe::compiler::tools::serializeModule(mlir_module, targetFormat);
  ASSERT_TRUE(kernelBinary.has_value()) << "Failed to serialize GPU kernel";
}

/**
 * @brief Test 2: Real GPU execution
 * Actually launches the compiled kernel and verifies data is written correctly
 */
TEST_F(CudfTableMaterializeTest, RealGPUExecution)
{
  // Create input cuDF table with user-defined row count
  constexpr size_t row_num   = 500;
  auto [id_data, value_data] = gqe_test::generateTestVectors(row_num);

  auto id_column = cudf::test::fixed_width_column_wrapper<int32_t>(id_data.begin(), id_data.end());
  auto value_column =
    cudf::test::fixed_width_column_wrapper<double>(value_data.begin(), value_data.end());

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(id_column.release());
  columns.push_back(value_column.release());
  auto input_table = std::make_unique<cudf::table>(std::move(columns));

  std::vector<std::string> simple_column_names = {"id", "value"};

  // Set up MLIR compilation (using test fixture)
  auto mlir_module = buildCudfTableMaterializeModule(*builder);

  // Full compilation pipeline (using test fixture)
  gqe::compiler::relalg::buildLowerToSCFPassPipeline(*pm);
  ASSERT_TRUE(mlir::succeeded(pm->run(mlir_module))) << "RelAlg to SCF conversion failed";

  gqe::compiler::gpu::buildLowerToNvvmPassPipeline(*pm, pipelineOptions);
  ASSERT_TRUE(mlir::succeeded(pm->run(mlir_module))) << "GPU compilation failed";

  constexpr auto targetFormat = mlir::gpu::CompilationTarget::Binary;
  auto kernelBinary           = gqe::compiler::tools::serializeModule(mlir_module, targetFormat);
  ASSERT_TRUE(kernelBinary.has_value()) << "Failed to serialize GPU kernel";

  // Set up kernel execution (like TPC-H Q6)
  gqe::utility::KernelArgsBuilder args;

  // Input table arguments
  args.append(input_table->num_rows());
  for (auto column : input_table->mutable_view()) {
    args.append(column);
  }

  // Output table arguments
  args.append(input_table->num_rows());  // output table capacity
  auto output_id_column = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, input_table->num_rows(), cudf::mask_state::UNALLOCATED);
  auto output_value_column = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64},
                                                           input_table->num_rows(),
                                                           cudf::mask_state::UNALLOCATED);
  args.append(output_id_column->mutable_view());
  args.append(output_value_column->mutable_view());

  // Atomic counter for CudfTableMaterialize
  rmm::device_scalar<int64_t> atomic_counter(0, rmm::cuda_stream_default);
  args.append(atomic_counter);

  // Launch kernel
  auto voidArgs = args.build();
  std::string kernelName("input_tableKernel");
  gqe::utility::KernelLauncher launcher(*kernelBinary);

  auto launchConfig = launcher.detectLaunchConfiguration(kernelName);
  launcher.launch(kernelName, launchConfig, voidArgs);
  rmm::cuda_stream_default.synchronize();

  // Verify results
  auto final_counter = atomic_counter.value(rmm::cuda_stream_default);
  ASSERT_EQ(final_counter, input_table->num_rows()) << "All rows should have been written";

  // Create output table and verify
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.push_back(std::move(output_id_column));
  output_columns.push_back(std::move(output_value_column));
  auto output_table = std::make_unique<cudf::table>(std::move(output_columns));

  ASSERT_EQ(output_table->num_rows(), input_table->num_rows())
    << "Output should have same number of rows as input";
  ASSERT_EQ(output_table->num_columns(), input_table->num_columns())
    << "Output should have same number of columns as input";

  // Verify that input and output contain exactly the same rows (order-agnostic)
  llvm::outs() << "Verifying data integrity for " << input_table->num_rows() << " rows...\n";
  ASSERT_TRUE(verify_same_row_sets(input_table->view(), output_table->view(), simple_column_names))
    << "Input and output tables should contain exactly the same rows";
}

/**
 * @brief Test 3: GPU execution with lineitem parquet files
 * Reads lineitem table from parquet files and executes GPU kernel
 */
TEST_F(CudfTableMaterializeTest, LineitemParquetExecution)
{
  const std::string data_path = gqe_test::get_tpch_data_path() + "/lineitem/";

  // Read lineitem table from parquet files (subset of columns for this test)
  std::vector<std::string> lineitem_columns = {
    "l_shipdate", "l_discount", "l_quantity", "l_extendedprice"};
  std::vector<cudf::data_type> lineitem_types = {cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
                                                 cudf::data_type(cudf::type_id::FLOAT64),
                                                 cudf::data_type(cudf::type_id::FLOAT64),
                                                 cudf::data_type(cudf::type_id::FLOAT64)};

  auto input_table = gqe_test::readTableFromParquet(data_path, lineitem_columns);
  input_table      = gqe_test::enforceDataTypes(std::move(input_table), lineitem_types);

  // Verify we read the lineitem data correctly
  ASSERT_GT(input_table->num_rows(), 0) << "Should have read rows from lineitem parquet";
  ASSERT_EQ(input_table->num_columns(), 4) << "Should have read 4 columns from lineitem parquet";

  // Set up MLIR compilation (using test fixture)
  auto mlir_module = buildLineitemTableMaterializeModule(*builder);

  // Full compilation pipeline (using test fixture)
  gqe::compiler::relalg::buildLowerToSCFPassPipeline(*pm);
  ASSERT_TRUE(mlir::succeeded(pm->run(mlir_module))) << "RelAlg to SCF conversion failed";

  gqe::compiler::gpu::buildLowerToNvvmPassPipeline(*pm, pipelineOptions);
  ASSERT_TRUE(mlir::succeeded(pm->run(mlir_module))) << "GPU compilation failed";

  constexpr auto targetFormat = mlir::gpu::CompilationTarget::Binary;
  auto kernelBinary           = gqe::compiler::tools::serializeModule(mlir_module, targetFormat);
  ASSERT_TRUE(kernelBinary.has_value()) << "Failed to serialize GPU kernel";

  // Set up kernel execution
  gqe::utility::KernelArgsBuilder args;

  // Input table arguments (from parquet)
  args.append(input_table->num_rows());
  for (auto column : input_table->mutable_view()) {
    args.append(column);
  }

  // Output table arguments (lineitem columns)
  // TODO: Use DECIMAL types here once they land in the main branch to avoid floating-point
  // comparison issues
  args.append(input_table->num_rows());  // output table capacity
  auto output_shipdate_column =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                  input_table->num_rows(),
                                  cudf::mask_state::UNALLOCATED);
  auto output_discount_column =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64},
                                  input_table->num_rows(),
                                  cudf::mask_state::UNALLOCATED);
  auto output_quantity_column =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64},
                                  input_table->num_rows(),
                                  cudf::mask_state::UNALLOCATED);
  auto output_extendedprice_column =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64},
                                  input_table->num_rows(),
                                  cudf::mask_state::UNALLOCATED);
  args.append(output_shipdate_column->mutable_view());
  args.append(output_discount_column->mutable_view());
  args.append(output_quantity_column->mutable_view());
  args.append(output_extendedprice_column->mutable_view());

  // Atomic counter for CudfTableMaterialize
  rmm::device_scalar<int64_t> atomic_counter(0, rmm::cuda_stream_default);
  args.append(atomic_counter);

  // Launch kernel
  auto voidArgs = args.build();
  std::string kernelName("lineitemKernel");
  gqe::utility::KernelLauncher launcher(*kernelBinary);

  auto launchConfig = launcher.detectLaunchConfiguration(kernelName);
  launcher.launch(kernelName, launchConfig, voidArgs);
  rmm::cuda_stream_default.synchronize();

  // Verify results
  auto final_counter = atomic_counter.value(rmm::cuda_stream_default);
  ASSERT_EQ(final_counter, input_table->num_rows()) << "All lineitem rows should have been written";

  // Create output table and verify (lineitem columns)
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  output_columns.push_back(std::move(output_shipdate_column));
  output_columns.push_back(std::move(output_discount_column));
  output_columns.push_back(std::move(output_quantity_column));
  output_columns.push_back(std::move(output_extendedprice_column));
  auto output_table = std::make_unique<cudf::table>(std::move(output_columns));

  ASSERT_EQ(output_table->num_rows(), input_table->num_rows())
    << "Output should have same number of rows as input";
  ASSERT_EQ(output_table->num_columns(), input_table->num_columns())
    << "Output should have same number of columns as input";

  // Verify that input and output contain exactly the same rows (order-agnostic)
  llvm::outs() << "Verifying lineitem data integrity for " << input_table->num_rows()
               << " rows...\n";
  ASSERT_TRUE(verify_same_row_sets(input_table->view(), output_table->view(), lineitem_columns))
    << "Input and output lineitem tables should contain exactly the same rows";
}

int main(int argc, char** argv)
{
  // Initialize LLVM commandline options.
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR CuDF table materialize integration test\n");

  // Initialize GoogleTest.
  ::testing::InitGoogleTest(&argc, argv);

  // Run all tests.
  return RUN_ALL_TESTS();
}
