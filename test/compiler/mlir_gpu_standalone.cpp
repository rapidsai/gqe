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

#include <gqe/compiler/Dialect/GPU/Pipelines/Passes.hpp>
#include <gqe/compiler/Tools/DialectRegistry.hpp>
#include <gqe/compiler/Tools/GPUModuleSerialization.hpp>
#include <gqe/utility/cuda_driver.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/kernel_arguments_builder.hpp>

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <optional>
#include <rmm/device_scalar.hpp>
#include <string>

/**
 * @brief Build a simple GPU kernel named `test` that takes no arguments.
 *
 * The kernel calls `printf` to print the string "test".
 */
mlir::ModuleOp buildSimpleModule(mlir::OpBuilder& builder, llvm::StringRef kernelName)
{
  // MLIR lowering passes expect the `GPUModuleOp` to be wrapped in a
  // `ModuleOp`. Didn't find any way to get around this and directly use the
  // `GPUModuleOp`.
  auto wrapper_module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapper_module.getBody());

  // The insertion guard captures the current op insertion point. When
  // destroyed, the insert guard resets the op insertion point to the initial
  // point.
  mlir::OpBuilder::InsertionGuard insertGuard(builder);

  // Create a GPU module. This is essentially a compilation unit for device
  // code.
  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  // Set the insertion point to the beginning of the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create a function with the signature "__global__ void()".
  auto testFnTy = builder.getFunctionType({}, {});
  auto testFn = builder.create<mlir::gpu::GPUFuncOp>(builder.getUnknownLoc(), kernelName, testFnTy);
  testFn->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

  // Set the insertion point to the beginning of the function.
  builder.setInsertionPointToStart(&testFn.getBody().front());

  // Initialize an MLIR string reference with our test string.
  mlir::StringRef strRef("test\n\0");

  // Call the `printf` function provided by CUDA. The MLIR `gpu` dialect
  // provides the function header.
  builder.create<mlir::gpu::PrintfOp>(
    builder.getUnknownLoc(), builder.getStringAttr(strRef), mlir::ValueRange{});

  // Call `return` to terminate the kernel function.
  builder.create<mlir::gpu::ReturnOp>(builder.getUnknownLoc());

  return wrapper_module;
}

/**
 * @brief Build a GPU kernel named `scalar_literal_arg` that takes a scalar
 * literal argument.
 */
mlir::ModuleOp buildScalarLiteralArgModule(mlir::OpBuilder& builder, llvm::StringRef kernelName)
{
  // MLIR lowering passes expect the `GPUModuleOp` to be wrapped in a
  // `ModuleOp`. Didn't find any way to get around this and directly use the
  // `GPUModuleOp`.
  auto wrapper_module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapper_module.getBody());

  // The insertion guard captures the current op insertion point. When
  // destroyed, the insert guard resets the op insertion point to the initial
  // point.
  mlir::OpBuilder::InsertionGuard insertGuard(builder);

  // Create a GPU module. This is essentially a compilation unit for device
  // code.
  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  // Set the insertion point to the beginning of the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create the literal type
  mlir::Type argTy = builder.getI32Type();

  // Create a function with the signature "__global__ void(int32_t)".
  auto testFnTy = builder.getFunctionType({argTy}, {});
  auto testFn = builder.create<mlir::gpu::GPUFuncOp>(builder.getUnknownLoc(), kernelName, testFnTy);
  testFn->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

  // Set the insertion point to the beginning of the function.
  builder.setInsertionPointToStart(&testFn.getBody().front());

  // Initialize an MLIR string reference with our test string.
  mlir::StringRef strRef("test: 0x%X\n\0");

  // Call the `printf` function provided by CUDA. The MLIR `gpu` dialect
  // provides the function header.
  builder.create<mlir::gpu::PrintfOp>(builder.getUnknownLoc(),
                                      builder.getStringAttr(strRef),
                                      mlir::ValueRange{testFn.getArgument(0)});

  // Call `return` to terminate the kernel function.
  builder.create<mlir::gpu::ReturnOp>(builder.getUnknownLoc());

  return wrapper_module;
}

/**
 * @brief Build a GPU kernel named `scalar_memref_arg` that takes a scalar
 * `memref` argument.
 */
mlir::ModuleOp buildScalarMemrefArgModule(mlir::OpBuilder& builder, llvm::StringRef kernelName)
{
  // MLIR lowering passes expect the `GPUModuleOp` to be wrapped in a
  // `ModuleOp`. Didn't find any way to get around this and directly use the
  // `GPUModuleOp`.
  auto wrapper_module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapper_module.getBody());

  // The insertion guard captures the current op insertion point. When
  // destroyed, the insert guard resets the op insertion point to the initial
  // point.
  mlir::OpBuilder::InsertionGuard insertGuard(builder);

  // Create a GPU module. This is essentially a compilation unit for device
  // code.
  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  // Set the insertion point to the beginning of the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create the memref type
  mlir::MemRefType argTy = mlir::MemRefType::get(
    mlir::SmallVector<int64_t, 0>(),
    builder.getI32Type(),
    mlir::MemRefLayoutAttrInterface(),
    mlir::gpu::AddressSpaceAttr::get(builder.getContext(), mlir::gpu::AddressSpace::Global));

  // Create a function with the signature "__global__ void(MemRef<int32_t>)".
  auto testFnTy = builder.getFunctionType({argTy}, {});
  auto testFn = builder.create<mlir::gpu::GPUFuncOp>(builder.getUnknownLoc(), kernelName, testFnTy);
  testFn->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

  // Set the insertion point to the beginning of the function.
  builder.setInsertionPointToStart(&testFn.getBody().front());

  // Load the memref argument
  mlir::memref::LoadOp scalarArg = builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(),
                                                                        testFn.getArgument(0),
                                                                        mlir::ValueRange(),
                                                                        /* non-temporal = */ false);

  // Initialize an MLIR string reference with our test string.
  mlir::StringRef strRef("test: 0x%X\n\0");

  // Call the `printf` function provided by CUDA. The MLIR `gpu` dialect
  // provides the function header.
  builder.create<mlir::gpu::PrintfOp>(
    builder.getUnknownLoc(), builder.getStringAttr(strRef), mlir::ValueRange{scalarArg});

  // Call `return` to terminate the kernel function.
  builder.create<mlir::gpu::ReturnOp>(builder.getUnknownLoc());

  return wrapper_module;
}

/**
 * @brief Build a GPU kernel named `memref_arg` that takes a dynamic `memref`
 * argument.
 */
mlir::ModuleOp buildDynamicMemrefArgModule(mlir::OpBuilder& builder, llvm::StringRef kernelName)
{
  // MLIR lowering passes expect the `GPUModuleOp` to be wrapped in a
  // `ModuleOp`. Didn't find any way to get around this and directly use the
  // `GPUModuleOp`.
  auto wrapper_module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(wrapper_module.getBody());

  // The insertion guard captures the current op insertion point. When
  // destroyed, the insert guard resets the op insertion point to the initial
  // point.
  mlir::OpBuilder::InsertionGuard insertGuard(builder);

  // Create a GPU module. This is essentially a compilation unit for device
  // code.
  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  // Set the insertion point to the beginning of the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create the memref type
  mlir::MemRefType argTy = mlir::MemRefType::get(
    mlir::ShapedType::kDynamic,
    builder.getI32Type(),
    mlir::MemRefLayoutAttrInterface(),
    mlir::gpu::AddressSpaceAttr::get(builder.getContext(), mlir::gpu::AddressSpace::Global));

  // Create a function with the signature "__global__ void(MemRef<?, int32_t>)".
  auto testFnTy = builder.getFunctionType({argTy}, {});
  auto testFn = builder.create<mlir::gpu::GPUFuncOp>(builder.getUnknownLoc(), kernelName, testFnTy);
  testFn->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());

  // Set the insertion point to the beginning of the function.
  builder.setInsertionPointToStart(&testFn.getBody().front());

  // Get the input column size.
  auto columnSize = builder.create<mlir::memref::DimOp>(
    builder.getUnknownLoc(), testFn.getArgument(0), /* dimension = */ 0);

  // Print the input column size.
  mlir::StringRef strRef("Column size: %d\n\0");
  builder.create<mlir::gpu::PrintfOp>(
    builder.getUnknownLoc(), builder.getStringAttr(strRef), mlir::ValueRange{columnSize});

  // Printf format string for column values.
  auto formatStr = builder.getStringAttr(" %2d: %d\n\0");

  // Load the memref argument
  auto lowerBound = builder.create<mlir::index::ConstantOp>(builder.getUnknownLoc(), 0);
  auto loopStride = builder.create<mlir::index::ConstantOp>(builder.getUnknownLoc(), 1);
  builder.create<mlir::scf::ForOp>(
    builder.getUnknownLoc(),
    lowerBound,
    columnSize,
    loopStride,
    mlir::ValueRange(),
    [&testFn, &formatStr](
      mlir::OpBuilder& builder, mlir::Location, mlir::Value loopOffset, mlir::ValueRange) {
      mlir::memref::LoadOp columnValue =
        builder.create<mlir::memref::LoadOp>(builder.getUnknownLoc(),
                                             testFn.getArgument(0),
                                             mlir::SmallVector<mlir::Value, 1>{loopOffset},
                                             /* non-temporal = */ false);

      builder.create<mlir::gpu::PrintfOp>(
        builder.getUnknownLoc(), formatStr, mlir::ValueRange{loopOffset, columnValue});

      builder.create<mlir::scf::YieldOp>(builder.getUnknownLoc(), mlir::ValueRange());
    });

  // Call `return` to terminate the kernel function.
  builder.create<mlir::gpu::ReturnOp>(builder.getUnknownLoc());

  return wrapper_module;
}

/**
 * @brief MLIR GPU standalone test fixture
 */
class MLIRGPUStandaloneTest : public testing::Test {
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

    context->loadDialect<mlir::index::IndexDialect>();
    context->loadDialect<mlir::memref::MemRefDialect>();
    context->loadDialect<mlir::scf::SCFDialect>();

    // GPU dialect provides a host and device functions. Host and device
    // functions aren't separate dialects, which makes using the GPU dialect
    // slightly confusing.
    context->loadDialect<mlir::gpu::GPUDialect>();

    // Create an op builder.
    builder = std::make_unique<mlir::OpBuilder>(context.get());

    GQE_CUDA_TRY(cudaSetDevice(deviceId));

    auto gpuArchOrdinal = gqe::utility::detectDeviceArchitecture(deviceId);
    gpuArch             = (llvm::Twine("sm_") + llvm::Twine(gpuArchOrdinal)).str();

#ifdef DEBUG_MLIR_MODULE
    // Print detected GPU architecture
    llvm::errs() << "Detected arch: " << gpuArch << "\n";
#endif

    // Create a pass manager.
    passManager = std::make_unique<mlir::PassManager>(builder->getContext());

    // Add the lowering pass pipeline.
    gqe::compiler::gpu::GPUToNVVMPipelineOptions options;
    options.targetArch = gpuArch;
    gqe::compiler::gpu::buildLowerToNvvmPassPipeline(*passManager, options);
  }

  // CUDA device ID.
  static constexpr int32_t deviceId = 0;

  // Specify either PTX or cubin.
  // static constexpr auto targetFormat =
  // mlir::gpu::CompilationTarget::Assembly;
  static constexpr auto targetFormat = mlir::gpu::CompilationTarget::Binary;

  std::unique_ptr<mlir::DialectRegistry> registry;
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::PassManager> passManager;
  std::string gpuArch;
};

// Test launching a simple kernel with no arguments.
TEST_F(MLIRGPUStandaloneTest, simpleLaunch)
{
  auto module_generation_start = std::chrono::high_resolution_clock::now();
  const std::string kernelName("simple_launch");

  // Generate the MLIR module.
  // auto host_main = generate_host_main(builder);
  auto testModule = buildSimpleModule(*builder, kernelName);

  auto module_generation_end          = std::chrono::high_resolution_clock::now();
  auto module_generation_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    module_generation_end - module_generation_start);
  llvm::errs() << "Module generation time: " << module_generation_elapsed_time.count() << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output before lowering to the MLIR NVVM dialect.
  testModule.dump();
#endif

  auto lowering_start = std::chrono::high_resolution_clock::now();

  // Run the lowering passes.
  EXPECT_TRUE(passManager->run(testModule).succeeded());

  auto lowering_end = std::chrono::high_resolution_clock::now();
  auto lowering_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(lowering_end - lowering_start);
  llvm::errs() << "Module lowering time: " << lowering_elapsed_time.count() << "ms\n";

  auto serialization_start = std::chrono::high_resolution_clock::now();

  // Serialize the module to a cubin (or PTX).
  auto testKernel = gqe::compiler::tools::serializeModule(testModule, targetFormat);
  EXPECT_TRUE(testKernel.has_value());

  auto serialization_end = std::chrono::high_resolution_clock::now();
  auto serialization_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(serialization_end - serialization_start);
  llvm::errs() << "Module serialization time: " << serialization_elapsed_time.count() << "ms\n";

  llvm::SmallVector<void*, 0> args;

  // Execute the generated module with CUDA Library API.
  const gqe::utility::LaunchConfiguration launchConfig = {1, 1, 0};
  gqe::utility::KernelLauncher launcher(*testKernel);
  launcher.launch(kernelName, launchConfig, args);

  cudf::get_default_stream().synchronize();
}

TEST_F(MLIRGPUStandaloneTest, scalarLiteralLaunch)
{
  const std::string kernelName("scalar_literal_arg");

  auto testModule = buildScalarLiteralArgModule(*builder, kernelName);

  // Run the lowering passes.
  EXPECT_TRUE(passManager->run(testModule).succeeded());

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output before lowering to the MLIR NVVM dialect.
  testModule.dump();
#endif

  auto testKernel = gqe::compiler::tools::serializeModule(testModule, targetFormat);
  EXPECT_TRUE(testKernel.has_value());

  // Allocate test data.
  llvm::SmallVector<int32_t, 1> data;
  data.push_back(0xDEADBEEF);

  // Prepare kernel arguments.
  gqe::utility::KernelArgsBuilder args;
  args.append(data[0]);
  auto voidArgs = args.build();

  // Launch the kernel.
  const gqe::utility::LaunchConfiguration launchConfig = {1, 1, 0};
  gqe::utility::KernelLauncher launcher(*testKernel);
  launcher.launch(kernelName, launchConfig, voidArgs);

  cudf::get_default_stream().synchronize();
}

TEST_F(MLIRGPUStandaloneTest, scalarMemrefLaunch)
{
  const std::string kernelName("scalar_memref_arg");

  auto testModule = buildScalarMemrefArgModule(*builder, kernelName);

  // Run the lowering passes.
  EXPECT_TRUE(passManager->run(testModule).succeeded());

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output before lowering to the MLIR NVVM dialect.
  testModule.dump();
#endif

  auto testKernel = gqe::compiler::tools::serializeModule(testModule, targetFormat);
  EXPECT_TRUE(testKernel.has_value());

  // Allocate test data.
  rmm::device_scalar<int32_t> gpuData(0xDEADBEEF, cudf::get_default_stream());

  // Prepare kernel arguments.
  gqe::utility::KernelArgsBuilder args;
  args.append(gpuData);
  auto voidArgs = args.build();

  // Launch the kernel.
  const gqe::utility::LaunchConfiguration launchConfig = {1, 1, 0};
  gqe::utility::KernelLauncher launcher(*testKernel);
  launcher.launch(kernelName, launchConfig, voidArgs);

  // Cleanup.
  cudf::get_default_stream().synchronize();
}

TEST_F(MLIRGPUStandaloneTest, dynamicMemrefLaunch)
{
  constexpr int32_t columnSize = 10;
  const std::string kernelName("dynamic_memref_arg");

  auto testModule = buildDynamicMemrefArgModule(*builder, kernelName);

  // Run the lowering passes.
  EXPECT_TRUE(passManager->run(testModule).succeeded());

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output before lowering to the MLIR NVVM dialect.
  testModule.dump();
#endif

  auto testKernel = gqe::compiler::tools::serializeModule(testModule, targetFormat);
  EXPECT_TRUE(testKernel.has_value());

  // Allocate test data.
  llvm::SmallVector<int32_t, columnSize> data;
  for (int32_t i = 0; i < columnSize; ++i) {
    data.push_back(10 + i);
  }

  // Prepare kernel arguments.
  auto gpuData = rmm::device_uvector<int32_t>(data.size(), cudf::get_default_stream());
  GQE_CUDA_TRY(
    cudaMemcpy(gpuData.data(), data.data(), sizeof(int32_t) * data.size(), cudaMemcpyDefault));
  auto column = cudf::column(std::move(gpuData), {}, 0);
  gqe::utility::KernelArgsBuilder args;
  args.append(column.mutable_view());
  auto voidArgs = args.build();

  // Launch the kernel.
  const gqe::utility::LaunchConfiguration launchConfig = {1, 1, 0};
  gqe::utility::KernelLauncher launcher(*testKernel);
  launcher.launch(kernelName, launchConfig, voidArgs);

  // Cleanup.
  cudf::get_default_stream().synchronize();
}
