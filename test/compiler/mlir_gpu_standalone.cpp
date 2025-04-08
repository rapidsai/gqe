/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Dialect/GPU/IR/CompilationInterfaces.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVM/NVVM/Target.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>

/**
 * @brief Generate a GPU kernel named `test`
 *
 * The kernel calls `printf` to print the string "test".
 */
mlir::ModuleOp generate_gpu_test(mlir::OpBuilder& builder)
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

  return wrapper_module;
}

/**
 * @brief Serializes the MLIR module to something CUDA understands (i.e., cubin, PTX).
 *
 * Precondition: There can only be a single `GPUModuleOp`. However, this module
 * can contain multiple kernels.
 *
 * See:
 * https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp#L122
 * See:
 * https://github.com/llvm/llvm-project/blob/main/mlir/unittests/Target/LLVM/SerializeNVVMTarget.cpp
 *
 * @param[in] module The MLIR module to serialize. Must already be lowered to the MLIR `nvvm`
 dialect.
 * @param[in] target_format Whether to output a cubin or PTX code. For cubin,
 * the target attributes must be set correctly for the GPU model. For PTX, CUDA
 * will JIT compile the kernel itself.
 *
 * @return The serialized module.
*/
std::optional<llvm::SmallVector<char, 0>> serialize_module(
  mlir::ModuleOp& module, mlir::gpu::CompilationTarget target_format)
{
  mlir::gpu::TargetOptions options("", {}, "", "", target_format);

  for (auto gpuModule : module.getBody()->getOps<mlir::gpu::GPUModuleOp>()) {
    auto targetAttr = gpuModule.getTargetsAttr();
    auto serializer = llvm::dyn_cast<mlir::gpu::TargetAttrInterface>(targetAttr[0]);

    // This ultimately calls into `mlir::NVVM::NVPTXSerializer::moduleToObject`.
    //
    // Ideally, the MLIR library is built with `MLIR_ENABLE_NVPTXCOMPILER`
    // enabled to avoid executing `ptxas` as a separate process.
    //
    // See: https://github.com/llvm/llvm-project/blob/main/mlir/lib/Target/LLVM/NVVM/Target.cpp#L521
    std::optional<llvm::SmallVector<char, 0>> object =
      serializer.serializeToObject(gpuModule, options);

    assert(object != std::nullopt);
    assert(!object->empty());

    return object;
  }

  llvm::errs() << "Warning: No GPU module generated\n";

  return {};
}

/**
 * @brief Lower an MLIR GPU module to MLIR NVVM dialect
 *
 * Lowering is performed with a minimal pass pipeline.
 *
 * @param[in,out] module The MLIR module to lower.
 * @param[in] target_arch The target device architecture to generate (e.g., "sm_90").
 *
 * @return Success or failure.
 */
mlir::LogicalResult lower_to_nvvm(mlir::ModuleOp& module, const llvm::StringRef target_arch)
{
  // Create the MLIR pass manager.
  //
  // The passes are a minimal subset of
  // `mlir::gpu::buildLowerToNVVMPassPipeline`.
  //
  // Specifically, it excludes all host code passes. Only device code passes are
  // needed.
  //
  // see:
  // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp
  mlir::PassManager pm(module.getContext());

  mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
  // Compiler optimization level
  //
  // Corresponds to, e.g., `-O3`
  nvvmTargetOptions.optLevel = 3;
  // `chip` describes the target architecture.
  //
  // Example: `sm_90`
  //
  // See: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch
  nvvmTargetOptions.chip = target_arch.str();
  // `features` describes the PTX version 8.0 as `+ptx80`. The `+` symbol adds
  // this PTX attribute to the set, a `-` would remove the attribute from the
  // set.
  //
  // It corresponds to LLVM's llc `-mattr=+ptx80` flag.
  //
  // The PTX attribute is optional.
  //
  // See: https://llvm.org/docs/CommandGuide/llc.html
  // See: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTX.td
  // nvvmTargetOptions.features = "+ptx80";
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

  mlir::ConvertGpuOpsToNVVMOpsOptions opt;
  opt.useBarePtrCallConv = false;
  // Specify the index bit width
  //
  // Used by the MLIR Index dialect.
  //
  // `0` sets the default value, which is the machine's word length. Should be
  // 64-bit for NVIDIA GPUs (unverified).
  opt.indexBitwidth = 0;

  pm.addPass(mlir::createStripDebugInfoPass());
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps(opt));

  // Run the lowering passes.
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Failed to lower module to LLVM dialect\n";
    return mlir::LogicalResult::failure();
  }

  return mlir::LogicalResult::success();
}

/**
 * @brief MLIR GPU standalone test fixture
 */
class mlir_gpu_standalone_test : public testing::Test {
 protected:
  void SetUp() override
  {
    // Create a registry.
    registry = std::make_unique<mlir::DialectRegistry>();

    // Register the relevant interfaces. This is mostly trial-and-error, but the
    // list is partially from `registerAllToLLVMIRTranslations` and
    // `registerAllGPUToLLVMIRTranslations`.
    mlir::registerBuiltinDialectTranslation(*registry);
    mlir::arith::registerConvertArithToLLVMInterface(*registry);
    mlir::cf::registerConvertControlFlowToLLVMInterface(*registry);
    mlir::registerGPUDialectTranslation(*registry);
    mlir::registerLLVMDialectTranslation(*registry);
    mlir::registerConvertMemRefToLLVMInterface(*registry);
    // Make sure `registerNVVMDialectTranslation` is registered. Otherwise, the
    // `GpuFuncOp` is translated to a PTX `.func` instead of a `.entry`. The
    // `.entry` is a kernel, the `.func` is a non-kernel function.
    mlir::registerNVVMDialectTranslation(*registry);
    mlir::registerConvertNVVMToLLVMInterface(*registry);
    mlir::NVVM::registerNVVMTargetInterfaceExternalModels(*registry);

    // Create the Context and load the relevant dialects into in. The dialects
    // are the ones we explicitly use to generate our program.
    context = std::make_unique<mlir::MLIRContext>(*registry);

    // GPU dialect provides a host and device functions. Host and device
    // functions aren't separate dialects, which makes using the GPU dialect
    // slightly confusing.
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

// Launch kernel using context-dependent CUDA Module API
TEST_F(mlir_gpu_standalone_test, launch_ctx_dependent)
{
  constexpr int device_id = 0;

  // Specify either PTX or cubin.
  // constexpr auto target_format = mlir::gpu::CompilationTarget::Assembly;
  constexpr auto target_format = mlir::gpu::CompilationTarget::Binary;

  cuda_init_and_context(device_id);
  auto gpu_arch     = detect_architecture_by_id(device_id);
  auto gpu_arch_str = llvm::Twine("sm_") + llvm::Twine(gpu_arch);

  llvm::errs() << "Detected arch: " << gpu_arch_str << "\n";

  auto module_generation_start = std::chrono::high_resolution_clock::now();

  // Create an opbuilder, for building operations (e.g., ConstantOp, FuncOp).
  auto builder = get_builder();

  // Generate the MLIR module.
  // auto host_main = generate_host_main(builder);
  auto test_module = generate_gpu_test(builder);

  auto module_generation_end          = std::chrono::high_resolution_clock::now();
  auto module_generation_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    module_generation_end - module_generation_start);
  llvm::errs() << "Module generation time: " << module_generation_elapsed_time.count() << "ms\n";

  auto lowering_start = std::chrono::high_resolution_clock::now();

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output before lowering to the MLIR LLVM dialect.
  test_module.dump();
#endif

  // Lower the higher-level MLIR dialects to MLIR NVVM dialect.
  EXPECT_TRUE(lower_to_nvvm(test_module, gpu_arch_str.str()).succeeded());

  auto lowering_end = std::chrono::high_resolution_clock::now();
  auto lowering_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(lowering_end - lowering_start);
  llvm::errs() << "Module lowering time: " << lowering_elapsed_time.count() << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output after lowering to the MLIR NVVM dialect.
  test_module.dump();
#endif

  auto serialization_start = std::chrono::high_resolution_clock::now();

  // Serialize the module to a cubin (or PTX)
  auto test_kernel = serialize_module(test_module, target_format);
  EXPECT_TRUE(test_kernel.has_value());

  auto serialization_end = std::chrono::high_resolution_clock::now();
  auto serialization_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(serialization_end - serialization_start);
  llvm::errs() << "Module serialization time: " << serialization_elapsed_time.count() << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print PTX code.
  if (target_format == mlir::gpu::CompilationTarget::Assembly) {
    llvm::errs() << *test_kernel << "\n";
  }
#endif

  // Execute the generated module with CUDA Module API.
  launch_kernel_ctx_dependent(*test_kernel, "test");
}

// Launch kernel using context-independent CUDA Library API
TEST_F(mlir_gpu_standalone_test, launch_ctx_independent)
{
  constexpr int device_id = 0;

  // Specify either PTX or cubin.
  // constexpr auto target_format = mlir::gpu::CompilationTarget::Assembly;
  constexpr auto target_format = mlir::gpu::CompilationTarget::Binary;

  cuda_init_and_context(device_id);
  auto gpu_arch     = detect_architecture_by_id(device_id);
  auto gpu_arch_str = llvm::Twine("sm_") + llvm::Twine(gpu_arch);

  // Print detected GPU architecture
  llvm::errs() << "Detected arch: " << gpu_arch_str << "\n";

  auto module_generation_start = std::chrono::high_resolution_clock::now();

  // Create an opbuilder, for building operations (e.g., ConstantOp, FuncOp).
  auto builder = get_builder();

  // Generate the MLIR module.
  // auto host_main = generate_host_main(builder);
  auto test_module = generate_gpu_test(builder);

  auto module_generation_end          = std::chrono::high_resolution_clock::now();
  auto module_generation_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    module_generation_end - module_generation_start);
  llvm::errs() << "Module generation time: " << module_generation_elapsed_time.count() << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output before lowering to the MLIR NVVM dialect.
  test_module.dump();
#endif

  auto lowering_start = std::chrono::high_resolution_clock::now();

  // Lower the higher-level MLIR dialects to MLIR NVVM dialect.
  EXPECT_TRUE(lower_to_nvvm(test_module, gpu_arch_str.str()).succeeded());

  auto lowering_end = std::chrono::high_resolution_clock::now();
  auto lowering_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(lowering_end - lowering_start);
  llvm::errs() << "Module lowering time: " << lowering_elapsed_time.count() << "ms\n";

  auto serialization_start = std::chrono::high_resolution_clock::now();

  // Serialize the module to a cubin (or PTX)
  auto test_kernel = serialize_module(test_module, target_format);
  EXPECT_TRUE(test_kernel.has_value());

  auto serialization_end = std::chrono::high_resolution_clock::now();
  auto serialization_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(serialization_end - serialization_start);
  llvm::errs() << "Module serialization time: " << serialization_elapsed_time.count() << "ms\n";

  // Execute the generated module with CUDA Library API.
  launch_kernel_ctx_independent(*test_kernel, "test");
}
