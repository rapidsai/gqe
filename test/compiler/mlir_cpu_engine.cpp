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

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>

/**
 * @brief Generate a CPU function called `main`
 *
 * The function calls `printf` to print the string "test".
 */
[[nodiscard]] mlir::FailureOr<mlir::ModuleOp> generate_host_main(mlir::OpBuilder& builder)
{
  // Create an MLIR module, effectively a top-level compiler basic block, and
  // set an insertion point at which we add ops.
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  // Add an `int main()` function.
  auto mainFnTy = builder.getFunctionType({}, builder.getI32Type());
  auto mainFn   = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", mainFnTy);

  // Create a basic block for `main`.
  mlir::Block* mainEntryBlock = mainFn.addEntryBlock();
  builder.setInsertionPointToStart(mainEntryBlock);

  // Initialize an MLIR string reference with our test string.
  mlir::StringRef strRef("test\n\0");

  // Make the string a global variable. The MLIR Toy tutorial does this as well.
  // Not sure if it's really necessary; maybe a local variable would suffice.
  mlir::LLVM::GlobalOp global;
  {
    // The insertion guard captures the current op insertion point. When
    // destroyed, the insert guard resets the op insertion point to the initial
    // point.
    mlir::OpBuilder::InsertionGuard insertGuard(builder);

    // Set the insertion point to the beginning of the module.
    builder.setInsertionPointToStart(module.getBody());

    // Create a `char` array with length of our string.
    auto intTy      = builder.getI8Type();
    size_t str_size = strRef.size();
    auto type       = mlir::LLVM::LLVMArrayType::get(intTy, str_size);

    // Create the global string variable.
    global = builder.create<mlir::LLVM::GlobalOp>(builder.getUnknownLoc(),
                                                  type,
                                                  true,
                                                  mlir::LLVM::Linkage::Internal,
                                                  "testVar",
                                                  builder.getStringAttr(strRef),
                                                  0);
  }

  // Get the address of the global string at array offset 0, we want to create a
  // pointer.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(builder.getUnknownLoc(), global);
  mlir::Value cst0      = builder.create<mlir::LLVM::ConstantOp>(
    builder.getUnknownLoc(), builder.getI64Type(), builder.getIndexAttr(0));
  mlir::Value strGlobal =
    builder.create<mlir::LLVM::GEPOp>(builder.getUnknownLoc(),
                                      mlir::LLVM::LLVMPointerType::get(builder.getContext()),
                                      global.getType(),
                                      globalPtr,
                                      mlir::ArrayRef<mlir::Value>({cst0, cst0}));

  // Create the return type for `printf`.
  mlir::IntegerType llvmI32Ty = mlir::IntegerType::get(builder.getContext(), 32);
  // Create the argument type for `printf`.
  auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());

  // Create the function signature for `printf`. This is simplified, we don't
  // consider variadic arguments.
  auto printffn =
    mlir::LLVM::lookupOrCreateFn(builder, module, "printf", {llvmPtrTy}, llvmI32Ty, true);

  // Check if the function was successfully created.
  if (mlir::failed(printffn)) {
    llvm::errs() << "Failed to create printf function\n";
    return mlir::failure();  // Handle the failure appropriately
  }

  // Create the call to `printf`.
  builder.create<mlir::LLVM::CallOp>(
    builder.getUnknownLoc(), *printffn, mlir::ValueRange{strGlobal});

  // Create the return value of `main`.
  auto retVal = builder.create<mlir::arith::ConstantOp>(
    builder.getUnknownLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), retVal.getResult());

  return module;
}

/**
 * @brief Lower high-level MLIR dialects to MLIR's LLVM dialect.
 *
 * The lowering uses the pass pipeline provided by MLIR.
 */
[[nodiscard]] mlir::LogicalResult convert_to_llvm(mlir::ModuleOp& module)
{
  // Register the MLIR dialect translators for lowering Func and Arith to LLVM.
  mlir::registerBuiltinDialectTranslation(*module.getContext());
  mlir::registerLLVMDialectTranslation(*module.getContext());

  // Add the Arith and Func translation passes to the pass manager.
  // ReconcileUnrealizedCasts is necessary as well; not sure why, but
  // translation fails when its not included.
  mlir::PassManager pm(module.getContext());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // Run the lowering passes.
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "Failed to lower module to LLVM dialect\n";
    return mlir::LogicalResult::failure();
  }

  return mlir::LogicalResult::success();
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
  auto maybeEngine               = mlir::ExecutionEngine::create(module, engineOptions);
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
 * @brief MLIR CPU engine test fixture
 */
class mlir_cpu_engine_test : public testing::Test {
 protected:
  mlir_cpu_engine_test() : registry(), context(registry) {}

  void SetUp() override
  {
    // Create a registery and load the relevant dialects into it.

    // LLVM is the target dialect we lower to.
    context.loadDialect<mlir::LLVM::LLVMDialect>();
    // Func provides a convenient interface for function signature definitions and
    // function calls.
    context.loadDialect<mlir::func::FuncDialect>();
    //  Arith provides integer literals.
    context.loadDialect<mlir::arith::ArithDialect>();
  }

  mlir::OpBuilder get_builder()
  {
    // Create an opbuilder, for building operations (e.g., ConstantOp, FuncOp).
    return mlir::OpBuilder(&context);
  }

  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
};

TEST_F(mlir_cpu_engine_test, execute_hello_world)
{
  auto module_generation_start = std::chrono::high_resolution_clock::now();

  auto builder = get_builder();

  auto host_main_result = generate_host_main(builder);
  EXPECT_TRUE(mlir::LogicalResult(host_main_result).succeeded());
  auto host_main = *host_main_result;

  auto module_generation_end          = std::chrono::high_resolution_clock::now();
  auto module_generation_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    module_generation_end - module_generation_start);
  llvm::outs() << "Module generation time: " << module_generation_elapsed_time.count() << "ms\n";

  auto lowering_start = std::chrono::high_resolution_clock::now();

  EXPECT_TRUE(convert_to_llvm(host_main).succeeded());

  auto lowering_end = std::chrono::high_resolution_clock::now();
  auto lowering_elapsed_time =
    std::chrono::duration_cast<std::chrono::milliseconds>(lowering_end - lowering_start);
  llvm::outs() << "Module lowering time: " << lowering_elapsed_time.count() << "ms\n";

#ifdef DEBUG_MLIR_MODULE
  // Print the MLIR output after lowering to the MLIR LLVM dialect.
  host_main.dump();
#endif

  EXPECT_TRUE(execute(host_main).succeeded());
}
