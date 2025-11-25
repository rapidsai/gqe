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
#include <gqe/compiler/Dialect/GPU/Pipelines/Passes.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgTypes.hpp>
#include <gqe/compiler/Dialect/RelAlg/Pipelines/Passes.hpp>
#include <gqe/compiler/Tools/DialectRegistry.hpp>
#include <gqe/compiler/Tools/GPUModuleSerialization.hpp>
#include <gqe/utility/cuda_driver.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/kernel_arguments_builder.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/raw_ostream.h>

#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class MlirTpchTest : public testing::Test {
 protected:
  void SetUp() override
  {
    // Create a registry.
    registry = std::make_unique<mlir::DialectRegistry>();

    // Register commonly used dialect translations.
    gqe::compiler::tools::registerLLVMIRTranslations(*registry);

    // Create the Context and load the relevant dialects into in. The dialects
    // are the ones we explicitly use to generate our program.
    context = std::make_unique<mlir::MLIRContext>(*registry);

    // GPU dialect provides a host and device functions. Host and device
    // functions aren't separate dialects, which makes using the GPU dialect
    // slightly confusing.
    context->loadDialect<mlir::gpu::GPUDialect>();
    context->loadDialect<mlir::index::IndexDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::ub::UBDialect>();

    // Load GQE dialects.
    context->loadDialect<gqe::compiler::relalg::RelAlgDialect>();

    // Create an op builder.
    builder = std::make_unique<mlir::OpBuilder>(context.get());

    // Create a pass manager.
    pm = std::make_unique<mlir::PassManager>(context.get());

    // Initialize CUDA before querying GPU architecture.
    gqe::utility::safeCuInit();

    // Get GPU architecture and configure NVVM lowering pipeline.
    auto gpuArchOrdinal        = gqe::utility::detectDeviceArchitecture(deviceId);
    gpuArch                    = (llvm::Twine("sm_") + llvm::Twine(gpuArchOrdinal)).str();
    pipelineOptions.targetArch = gpuArch;

#ifdef DEBUG_MLIR_MODULE
    // Print detected GPU architecture
    llvm::outs() << "Detected arch: " << gpuArch << "\n";
#endif
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

namespace {

/// Return number of days since UNIX epoch 1970-01-01.
///
/// cuDF specifies day timestamps as 32-bit integers since the UNIX epoch.
///
/// Sources:
///  https://howardhinnant.github.io/date_algorithms.html#days_from_civil
///  https://github.com/rapidsai/cudf/blob/branch-25.08/cpp/include/cudf/types.hpp#L216
int32_t days_from_civil(int32_t year, uint32_t month, uint32_t day) noexcept
{
  year -= month <= 2;
  const int32_t era  = (year >= 0 ? year : year - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(year - era * 400);                        // [0, 399]
  const unsigned doy = (153 * (month > 2 ? month - 3 : month + 9) + 2) / 5 + day - 1;  // [0, 365]
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;  // [0, 146096]
  return era * 146097 + static_cast<int32_t>(doe) - 719468;
}

/// @brief Build TPC-H Query 6 in MLIR.
///
/// @param[in] builder The MLIR OpBuilder for building the query.
///
/// @return A `std::optional` containing an MLIR module with Q6.
std::optional<mlir::ModuleOp> buildTpchQ6(mlir::OpBuilder& builder)
{
  auto wrapperModule = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapperModule.getBody());
  mlir::OpBuilder::InsertionGuard insertGuard(builder);

  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // FROM lineitem
  auto tableName                                    = "lineitem";
  auto tableNameAttr                                = builder.getStringAttr(tableName);
  llvm::SmallVector<mlir::Attribute, 4> loadColumns = {
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_shipdate", builder.getI32Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_discount", builder.getF64Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_quantity", builder.getF64Type()),
    gqe::compiler::relalg::ColumnDefAttr::get(
      builder.getContext(), "l_extendedprice", builder.getF64Type())};
  auto loadColumnsAttr = builder.getArrayAttr(loadColumns);
  auto lineitemScanOp  = builder.create<gqe::compiler::relalg::CudfTableScanOp>(
    builder.getUnknownLoc(), tableNameAttr, loadColumnsAttr);

  auto filterOp =
    builder.create<gqe::compiler::relalg::FilterOp>(builder.getUnknownLoc(), lineitemScanOp);
  mlir::Block* q6FilterPredicate;

  // WHERE
  //   l_shipdate >= date '1994-01-01'
  //   AND l_shipdate < date '1994-01-01' + interval '1' year
  //   AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
  //   AND l_quantity < 24
  {
    mlir::OpBuilder::InsertionGuard filterPredicateGuard(builder);
    q6FilterPredicate = &filterOp.getPredicate().emplaceBlock();
    builder.setInsertionPointToStart(q6FilterPredicate);

    auto shipdateIU = gqe::compiler::relalg::IURefAttr::get(
      builder.getContext(), 0, builder.getStringAttr("l_shipdate"));

    auto discountIU = gqe::compiler::relalg::IURefAttr::get(
      builder.getContext(), 1, builder.getStringAttr("l_discount"));

    auto quantityIU = gqe::compiler::relalg::IURefAttr::get(
      builder.getContext(), 2, builder.getStringAttr("l_quantity"));

    // Set TPC-H Q6 constants
    mlir::Value shipdateLower = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), builder.getI32Type(), days_from_civil(1994, 1, 1));
    mlir::Value shipdateUpper = builder.create<mlir::arith::ConstantIntOp>(
      builder.getUnknownLoc(), builder.getI32Type(), days_from_civil(1995, 1, 1));

    // Debugging note: Computing `0.06 - 0.01` and `0.06 + 0.01` have rounding
    // errors that cause a massively incorrect Q6 final result. Strangely, the
    // error occurs only with `F64` type but not with `F32` type.
    //
    // Comparing to the precomputed literals results in the correct Q6 result.
    mlir::Value discountLower = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64FloatAttr(0.05));
    mlir::Value discountUpper = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64FloatAttr(0.07));

    mlir::Value quantityUpper = builder.create<mlir::arith::ConstantOp>(
      builder.getUnknownLoc(), builder.getF64FloatAttr(24.0));

    // Generate comparison ops
    // TODO: use branches instead of predication
    auto shipdateRef = builder.create<gqe::compiler::relalg::GetIUOp>(
      builder.getUnknownLoc(), builder.getI32Type(), shipdateIU);
    auto shipdateLowerPredicate = builder
                                    .create<mlir::arith::CmpIOp>(builder.getUnknownLoc(),
                                                                 mlir::arith::CmpIPredicate::sle,
                                                                 shipdateLower,
                                                                 shipdateRef.getResult())
                                    .getResult();
    auto shipdateUpperPredicate = builder
                                    .create<mlir::arith::CmpIOp>(builder.getUnknownLoc(),
                                                                 mlir::arith::CmpIPredicate::sgt,
                                                                 shipdateUpper,
                                                                 shipdateRef.getResult())
                                    .getResult();
    auto shipdatePredicate =
      builder
        .create<mlir::arith::AndIOp>(
          builder.getUnknownLoc(), shipdateLowerPredicate, shipdateUpperPredicate)
        .getResult();

    // SQL BETWEEN is inclusive, thus `<=` and `>=`
    auto discountRef = builder.create<gqe::compiler::relalg::GetIUOp>(
      builder.getUnknownLoc(), builder.getF64Type(), discountIU);
    auto discountLowerPredicate = builder
                                    .create<mlir::arith::CmpFOp>(builder.getUnknownLoc(),
                                                                 mlir::arith::CmpFPredicate::OLE,
                                                                 discountLower,
                                                                 discountRef.getResult())
                                    .getResult();
    auto discountUpperPredicate = builder
                                    .create<mlir::arith::CmpFOp>(builder.getUnknownLoc(),
                                                                 mlir::arith::CmpFPredicate::OGE,
                                                                 discountUpper,
                                                                 discountRef.getResult())
                                    .getResult();
    auto shipdateDiscountPredicate =
      builder
        .create<mlir::arith::AndIOp>(
          builder.getUnknownLoc(), shipdatePredicate, discountLowerPredicate)
        .getResult();
    auto discountPredicate =
      builder
        .create<mlir::arith::AndIOp>(
          builder.getUnknownLoc(), shipdateDiscountPredicate, discountUpperPredicate)
        .getResult();

    auto quantityRef = builder.create<gqe::compiler::relalg::GetIUOp>(
      builder.getUnknownLoc(), builder.getF64Type(), quantityIU);
    auto quantityLowerPredicate = builder
                                    .create<mlir::arith::CmpFOp>(builder.getUnknownLoc(),
                                                                 mlir::arith::CmpFPredicate::OLT,
                                                                 quantityRef.getResult(),
                                                                 quantityUpper)
                                    .getResult();
    auto quantityPredicate = builder.create<mlir::arith::AndIOp>(
      builder.getUnknownLoc(), discountPredicate, quantityLowerPredicate);

    builder.create<gqe::compiler::relalg::YieldOp>(builder.getUnknownLoc(),
                                                   mlir::ValueRange(quantityPredicate));
  }

  // l_extendedprice * l_discount
  auto mapOp = builder.create<gqe::compiler::relalg::MapOp>(builder.getUnknownLoc(), filterOp);
  mlir::Block* q6MapExpression;

  {
    mlir::OpBuilder::InsertionGuard mapExpressionGuard(builder);
    q6MapExpression = &mapOp.getExpression().emplaceBlock();
    builder.setInsertionPointToStart(q6MapExpression);

    auto discountIU = gqe::compiler::relalg::IURefAttr::get(
      builder.getContext(), 1, builder.getStringAttr("l_discount"));
    auto discountRef = builder.create<gqe::compiler::relalg::GetIUOp>(
      builder.getUnknownLoc(), builder.getF64Type(), discountIU);

    auto extendedPriceIU = gqe::compiler::relalg::IURefAttr::get(
      builder.getContext(), 3, builder.getStringAttr("l_extendedprice"));
    auto extendedPriceRef = builder.create<gqe::compiler::relalg::GetIUOp>(
      builder.getUnknownLoc(), builder.getF64Type(), extendedPriceIU);

    auto multiplyOp =
      builder.create<mlir::arith::MulFOp>(builder.getUnknownLoc(), extendedPriceRef, discountRef);

    builder.create<gqe::compiler::relalg::YieldOp>(builder.getUnknownLoc(),
                                                   mlir::ValueRange{multiplyOp});
  }

  // SUM(...)
  llvm::SmallVector<mlir::Attribute, 4> aggFns = {gqe::compiler::relalg::AggregateFnArgAttr::get(
    builder.getContext(),
    gqe::compiler::relalg::IURefAttr::get(builder.getContext(), 0, nullptr),
    gqe::compiler::relalg::AggregateFunction::Sum)};

  auto aggFnsAttr = builder.getArrayAttr(aggFns);
  builder.create<gqe::compiler::relalg::ScalarAggregateOp>(
    builder.getUnknownLoc(), mapOp, aggFnsAttr);

  // TODO: SELECT ... AS revenue, i.e., the sink op

  return std::make_optional(wrapperModule);
}

/**
 * @brief Convert the read columns into the specified types if necessary
 *
 * Note: Copied from the private namespace in src/storage/parquet.cpp.
 */
std::unique_ptr<cudf::table> enforceDataTypes(std::unique_ptr<cudf::table> input,
                                              std::vector<cudf::data_type> const& data_type)
{
  auto input_columns     = input->release();
  auto const num_columns = input_columns.size();

  std::vector<std::unique_ptr<cudf::column>> converted_columns;
  converted_columns.reserve(num_columns);
  for (std::size_t column_idx = 0; column_idx < num_columns; column_idx++) {
    auto const column_view   = input_columns[column_idx]->view();
    auto const expected_type = data_type[column_idx];

    if (column_view.type() == expected_type) {
      converted_columns.push_back(std::move(input_columns[column_idx]));
    } else {
      converted_columns.push_back(cudf::cast(column_view, expected_type));
    }
  }
  assert(converted_columns.size() == num_columns);

  return std::make_unique<cudf::table>(std::move(converted_columns));
}

std::unique_ptr<cudf::table> readTableFromParquet(llvm::StringRef dataPath,
                                                  std::vector<std::string>& columns)
{
  auto filePaths = gqe::utility::get_parquet_files(dataPath.str());

  auto readSource = cudf::io::source_info(std::move(filePaths));
  cudf::io::parquet_reader_options_builder builder(readSource);
  builder.columns(columns);
  auto table = cudf::io::read_parquet(builder);

  return std::move(table.tbl);
}

}  // namespace

/// @brief Build TPC-H Q6 using only the RelAlg dialect, and lower it to PTX.
///
/// This integration test checks that the RelAlgToSCF pass lowers Q6 to valid
/// SCF IR, and the custom `buildLowerToNvvmPassPipeline` lowers to valid NVVM
/// IR. For the purpose of this test, "valid NVVM IR" means that can be
/// serialized to PTX.
TEST_F(MlirTpchTest, Q6RelAlgToPTX)
{
  constexpr auto targetFormat = mlir::gpu::CompilationTarget::Assembly;

  auto result = buildTpchQ6(*builder);
  EXPECT_TRUE(result.has_value());

  auto wrapperModule = result.value();

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR module in RelAlg dialect.
  wrapperModule.print(llvm::outs(), mlir::OpPrintingFlags().printNameLocAsPrefix());
#endif

  // Add the RelAlgToSCF pass.
  gqe::compiler::relalg::buildLowerToSCFPassPipeline(*pm);

  // Run the RelAlgToSCF pass.
  EXPECT_TRUE(pm->run(wrapperModule).succeeded());
  pm->clear();

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR module lowered to SCF dialect.
  wrapperModule.print(llvm::outs(), mlir::OpPrintingFlags().printNameLocAsPrefix());
#endif

  // Add the lowering pass pipeline.
  gqe::compiler::gpu::buildLowerToNvvmPassPipeline(*pm, pipelineOptions);

  // Run the lowering passes.
  EXPECT_TRUE(pm->run(wrapperModule).succeeded());
  pm->clear();

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR module lowered to NVVM dialect.
  wrapperModule.print(llvm::outs(), mlir::OpPrintingFlags().printNameLocAsPrefix());
#endif

  // Serialize the module to PTX
  auto tpchQ6Kernel = gqe::compiler::tools::serializeModule(wrapperModule, targetFormat);
  EXPECT_TRUE(tpchQ6Kernel.has_value());

#ifdef DEBUG_MLIR_MODULE
  // Print the PTX kernel.
  llvm::outs() << tpchQ6Kernel << "\n";
#endif

  SUCCEED();
}

TEST_F(MlirTpchTest, Q6Execution)
{
  constexpr auto targetFormat = mlir::gpu::CompilationTarget::Binary;
  const auto dataPath         = gqe_test::get_tpch_data_path() + "/lineitem/";

  auto builderResult = buildTpchQ6(*builder);
  EXPECT_TRUE(builderResult.has_value());

  auto wrapperModule = builderResult.value();

  // Add the RelAlgToSCF pass.
  gqe::compiler::relalg::buildLowerToSCFPassPipeline(*pm);

  // Run the RelAlgToSCF pass.
  EXPECT_TRUE(pm->run(wrapperModule).succeeded());
  pm->clear();

  // Add the lowering pass pipeline.
  gqe::compiler::gpu::buildLowerToNvvmPassPipeline(*pm, pipelineOptions);

  // Run the lowering passes.
  EXPECT_TRUE(pm->run(wrapperModule).succeeded());
  pm->clear();

  // Serialize the module to cubin
  auto tpchQ6Kernel = gqe::compiler::tools::serializeModule(wrapperModule, targetFormat);
  EXPECT_TRUE(tpchQ6Kernel.has_value());

  // Load the `lineitem` table from Parquet.
  std::vector<std::string> lineitemColumns = {
    "l_shipdate", "l_discount", "l_quantity", "l_extendedprice"};
  std::vector<cudf::data_type> lineitemTypes = {cudf::data_type(cudf::type_id::TIMESTAMP_DAYS),
                                                cudf::data_type(cudf::type_id::FLOAT64),
                                                cudf::data_type(cudf::type_id::FLOAT64),
                                                cudf::data_type(cudf::type_id::FLOAT64)};

  auto lineitemTable = readTableFromParquet(dataPath, lineitemColumns);
  lineitemTable      = enforceDataTypes(std::move(lineitemTable), lineitemTypes);
  llvm::outs() << "rows: " << lineitemTable->num_rows() << "\n";

  // Build kernel arguments.
  //
  // FIXME: There should be a transform pass on an intermediate "KernelOutlining" dialect that
  // declares the kernels and builds their launch arguments.
  gqe::utility::KernelArgsBuilder args;
  args.append(lineitemTable->num_rows());

  for (auto column : lineitemTable->mutable_view()) {
    args.append(column);
  }

  rmm::device_scalar<double> resultValue(0, cudf::get_default_stream());
  args.append(resultValue);

  // Get the kernel and arguments.
  auto voidArgs = args.build();
  std::string kernelName("lineitemKernel");
  gqe::utility::KernelLauncher launcher(*tpchQ6Kernel);

  // Launch the kernel.
  auto launchConfig = launcher.detectLaunchConfiguration(kernelName);
  launcher.launch(kernelName, launchConfig, voidArgs);

  // Stream synchronize.
  cudf::get_default_stream().synchronize();

  auto queryResult = resultValue.value(cudf::get_default_stream());
  llvm::outs() << "TPC-H Q6 result: " << llvm::format("%0.4f", queryResult) << "\n";

  constexpr double expected = 123141078.2283;
  // The tight tolerance of 1e-06 used by gqe-python fails with atomic
  // floating-point reduction. Loosen the tolerance to 1e-05.
  constexpr double tolerance = 1e-05;
  ASSERT_NEAR(queryResult, expected, tolerance);

  SUCCEED();
}

int main(int argc, char** argv)
{
  // Initialize LLVM commandline options.
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR TPC-H Q6 integration test\n");

  // Initialize GoogleTest.
  ::testing::InitGoogleTest(&argc, argv);

  // Run all tests.
  return RUN_ALL_TESTS();
}
