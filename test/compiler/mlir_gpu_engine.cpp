/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gtest/gtest.h>

#include <kernel_launch.hpp>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Pipelines/Passes.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVM/NVVM/Target.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>

/**
 * @brief Generate a GPU kernel named `test`
 *
 * The kernel calls `printf` to print the string "test".
 */
void generate_gpu_test(mlir::OpBuilder& builder)
{
  // The insertion guard captures the current op insertion point. When
  // destroyed, the insert guard resets the op insertion point to the initial
  // point.
  mlir::OpBuilder::InsertionGuard insertGuard(builder);

  // Create a GPU module. This is essentially a compilation unit for device code.
  auto module = builder.create<mlir::gpu::GPUModuleOp>(builder.getUnknownLoc(),
                                                       builder.getStringAttr("gpu_kernels"));
  // Set the insertion point to the beginning of the module.
  builder.setInsertionPointToStart(&module.getBodyRegion().front());

  // Create a function "__global__ void test()".
  auto testFnTy = builder.getFunctionType({}, {});
  auto testFn   = builder.create<mlir::gpu::GPUFuncOp>(builder.getUnknownLoc(), "test", testFnTy);
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
}

/**
 * @brief Lower high-level MLIR dialects to MLIR's LLVM dialect.
 *
 * The lowering uses the pass pipeline provided by MLIR. The pipeline includes the cubin/PTX
 * serialization pass. The device code is wrapped in an MLIR module within the host code, instead a
 * bare `char` array.
 *
 * @param[in,out] module The MLIR module to lower.
 * @param[in] target_arch The target device architecture to generate (e.g., "sm_90").
 *
 * @return Success or failure.
 */
[[nodiscard]] mlir::LogicalResult convert_to_llvm(mlir::ModuleOp& module,
                                                  const llvm::StringRef target_arch)
{
  mlir::PassManager pm(module.getContext());

  // Configure the NVVM compilation options.
  //
  // See the function `lower_to_nvvm` in `mlir_gpu_standalone.cpp` for details.
  mlir::gpu::GPUToNVVMPipelineOptions nvvm_pipeline_options;
  nvvm_pipeline_options.optLevel  = 3;
  nvvm_pipeline_options.cubinChip = target_arch.str();
  // nvvm_pipeline_options.cubinFeatures = "+ptx80";

  // This pipeline includes several conversion passes. The pipeline splits the
  // host and device code, and separately lowers them. The pipeline includes
  // standard dialects, such as `Arith`, `Func`, and of course `GPU`.
  mlir::gpu::buildLowerToNVVMPassPipeline(pm, nvvm_pipeline_options);

  // Run the lowering passes.
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Failed to lower module to LLVM dialect\n";
    return mlir::LogicalResult::failure();
  }

  return mlir::LogicalResult::success();
}

/**
 * @brief Generate the `main` function.
 */
[[nodiscard]] mlir::ModuleOp generate_host_main(mlir::OpBuilder& builder, int device_id)
{
  // Create an MLIR module, effectively a top-level compiler basic block, and
  // set an insertion point at which we add ops.
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module->setAttr("gpu.container_module", builder.getUnitAttr());
  builder.setInsertionPointToEnd(module.getBody());

  // Generate the GPU kernel.
  generate_gpu_test(builder);

  // Add an `int main()` function.
  auto mainFnTy = builder.getFunctionType({}, builder.getI32Type());
  auto mainFn   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", mainFnTy);

  // Create a basic block for `main`.
  mlir::Block* mainEntryBlock = mainFn.addEntryBlock();
  builder.setInsertionPointToStart(mainEntryBlock);

  auto deviceIdVal = builder.create<mlir::arith::ConstantOp>(
    builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(device_id));
  builder.create<mlir::gpu::SetDefaultDeviceOp>(builder.getUnknownLoc(), deviceIdVal);

  // Add constant `index` types for the CUDA grid and block sizes.
  auto gridSizeX  = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  auto gridSizeY  = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  auto gridSizeZ  = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  auto blockSizeX = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  auto blockSizeY = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  auto blockSizeZ = builder.create<mlir::arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);

  // Add a constant `int` for the CUDA dynamic shared memory size.
  auto dynSharedMemSize =
    builder.create<mlir::arith::ConstantIntOp>(builder.getUnknownLoc(), 0, 32);

  // Add a call to the CUDA kernel launch function, with appropriate launch parameters.
  builder.create<mlir::gpu::LaunchFuncOp>(
    builder.getUnknownLoc(),
    mlir::SymbolRefAttr::get(builder.getContext(),
                             "gpu_kernels",
                             {mlir::SymbolRefAttr::get(builder.getContext(), "test")}),
    mlir::gpu::KernelDim3{gridSizeX, gridSizeY, gridSizeZ},
    mlir::gpu::KernelDim3{blockSizeX, blockSizeY, blockSizeZ},
    dynSharedMemSize,
    mlir::ValueRange{});

  // Create the return value of `main`.
  auto retVal = builder.create<mlir::arith::ConstantOp>(
    builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), retVal.getResult());

  return module;
}

/**
 * @brief Execute an MLIR module.
 *
 * Before execution, we lower the module from MLIR to LLVM IR.
 *
 * Execution is performed using the execution engine provided by MLIR. The
 * engine takes care of compiling LLVM IR to a binary, linking, and invokes the
 * `main` function as a new program.
 */
[[nodiscard]] mlir::LogicalResult execute(const mlir::ModuleOp& module)
{
  // Initialize LLVM.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Set the LLVM compiler flags to `-O3`.
  auto optPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);

  // Create an MLIR execution engine for compiling and running the generated
  // code.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer      = optPipeline;
  engineOptions.enableObjectDump = true;

  // Add the dynamic library paths that the execution engine needs to run CUDA
  // code. These libraries are taken from the `printf.mlir` integration test.
  auto lib_paths = std::array{llvm::StringRef{"/usr/local/lib/libmlir_cuda_runtime.so"},
                              llvm::StringRef{"/usr/local/lib/libmlir_runner_utils.so"}};
  engineOptions.sharedLibPaths = lib_paths;

  // Create the execution engine.
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "Failed to create engine.");
  std::unique_ptr<mlir::ExecutionEngine>& engine = maybeEngine.get();

  // Allocate an integer for the return value of `main`. Without a return value,
  // we trigger a segfault due to invalid memory access!
  int32_t retval = 0;

  // LLVM expects `main` to receive an argument array, with the return value at
  // position 0 followed by the actual arguments. In our case, it's only the
  // return value.
  llvm::SmallVector<void*> args{&retval};

  // Invoke `main`.
  llvm::Error err = engine->invokePacked("main", args);
  if (err) {
    llvm::errs() << "Failed to invoke code: ";
    llvm::errs() << err;
    llvm::errs() << "\n";

    return mlir::LogicalResult::failure();
  }

  return mlir::LogicalResult::success();
}

/**
 * @brief MLIR GPU engine test fixture
 */
class mlir_gpu_engine_test : public testing::Test {
 protected:
  void SetUp() override
  {
    // Create a registry.
    registry = std::make_unique<mlir::DialectRegistry>();

    // Register the relevant interfaces. This is mostly trial-and-error, but the
    // list is partially from `registerAllToLLVMIRTranslations` and
    // `registerAllGPUToLLVMIRTranslations`.
    mlir::registerBuiltinDialectTranslation(*registry);
    mlir::registerConvertComplexToLLVMInterface(*registry);
    mlir::registerConvertFuncToLLVMInterface(*registry);
    mlir::registerConvertMemRefToLLVMInterface(*registry);
    mlir::registerConvertNVVMToLLVMInterface(*registry);
    mlir::registerGPUDialectTranslation(*registry);
    mlir::registerLLVMDialectTranslation(*registry);
    mlir::registerNVVMDialectTranslation(*registry);
    mlir::arith::registerConvertArithToLLVMInterface(*registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(*registry);
    mlir::gpu::registerOffloadingLLVMTranslationInterfaceExternalModels(*registry);
    mlir::NVVM::registerNVVMTargetInterfaceExternalModels(*registry);
    mlir::ub::registerConvertUBToLLVMInterface(*registry);

    // FIXME: This test is currently broken. Call
    // `registerConvertVectorToLLVMInterface` when updating to 20.1.3 or later.

    // Create the Context and load the relevant dialects into in. The dialects
    // are the ones we explicitly use to generate our program.
    context = std::make_unique<mlir::MLIRContext>(*registry);

    // LLVM is the target dialect we lower host code to.
    context->loadDialect<mlir::LLVM::LLVMDialect>();
    // NVVM is the target dialect we lower device code to.
    context->loadDialect<mlir::NVVM::NVVMDialect>();
    // Func provides a convenient interface for function signature definitions and
    // function calls.
    context->loadDialect<mlir::func::FuncDialect>();
    //  Arith provides integer literals.
    context->loadDialect<mlir::arith::ArithDialect>();
    // GPU provides a host and device functions. Host and device functions aren't
    // separate dialects, which makes using the GPU dialect slightly confusing.
    context->loadDialect<mlir::gpu::GPUDialect>();
  }

  mlir::OpBuilder get_builder()
  {
    // Create an opbuilder, for building operations (e.g., ConstantOp, FuncOp).
    return mlir::OpBuilder(context.get());
  }

  std::unique_ptr<mlir::DialectRegistry> registry;
  std::unique_ptr<mlir::MLIRContext> context;
};

TEST_F(mlir_gpu_engine_test, execute_hello_world)
{
  constexpr int device_id = 0;

  // CUDA context is only necessary to dynamically detect device architecture.
  // The kernel launch occurs in a different process.
  cuda_init_and_context(device_id);

  auto gpu_arch     = detect_architecture_by_id(device_id);
  auto gpu_arch_str = llvm::Twine("sm_") + llvm::Twine(gpu_arch);

  // Print detected GPU architecture
  llvm::errs() << "Detected arch: " << gpu_arch_str << "\n";

  auto module_generation_start = std::chrono::high_resolution_clock::now();

  // Create an opbuilder, for building operations (e.g., ConstantOp, FuncOp).
  auto builder = get_builder();

  // Generate the MLIR module.
  auto host_main = generate_host_main(builder, device_id);

  auto module_generation_end          = std::chrono::high_resolution_clock::now();
  auto module_generation_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    module_generation_end - module_generation_start);
  llvm::errs() << "Module generation time: " << module_generation_elapsed_time.count() << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print the generate MLIR to screen.
  host_main.dump();
#endif

  auto lowering_start = std::chrono::high_resolution_clock::now();

  // Lower the higher-level MLIR dialects to MLIR LLVM dialect.
  EXPECT_TRUE(convert_to_llvm(host_main, gpu_arch_str.str()).succeeded());

  auto lowering_end = std::chrono::high_resolution_clock::now();
  auto lowering_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(lowering_end - lowering_start);
  llvm::errs() << "Module lowering time (includes serialization): " << lowering_elapsed_time.count()
               << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output after lowering to the MLIR LLVM dialect.
  host_main.dump();
#endif

  // Execute the generated module.
  EXPECT_TRUE(execute(host_main).succeeded());
}
