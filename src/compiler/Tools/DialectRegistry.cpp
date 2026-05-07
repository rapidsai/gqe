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

#include <gqe/compiler/Tools/DialectRegistry.hpp>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Target/LLVM/NVVM/Target.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>

namespace gqe {
namespace compiler {
namespace tools {

void registerLLVMIRTranslations(mlir::DialectRegistry& registry)
{
  // Register the relevant upstream MLIR conversion interfaces for the LLVM
  // target backend.
  //
  // Built-in dialect.
  mlir::registerBuiltinDialectTranslation(registry);
  // `arith` dialect.
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  // `cf` dialect.
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  // `gpu` dialect.
  mlir::registerGPUDialectTranslation(registry);
  // mlir::gpu::registerConvertGpuToLLVMInterface(*registry);
  // mlir::NVVM::registerConvertGpuToNVVMInterface(*registry);
  // `index` dialect.
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  // `llvm` dialect.
  mlir::registerLLVMDialectTranslation(registry);
  // `memref` dialect.
  mlir::registerConvertMemRefToLLVMInterface(registry);
  // `nvvm` dialect.
  mlir::registerConvertNVVMToLLVMInterface(registry);
  // `ub` dialect.
  mlir::ub::registerConvertUBToLLVMInterface(registry);

  // Make sure `registerNVVMDialectTranslation` is registered. Otherwise, the
  // `GpuFuncOp` is translated to a PTX `.func` instead of a `.entry`. The
  // `.entry` is a kernel, the `.func` is a non-kernel function.
  mlir::registerNVVMDialectTranslation(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
}

}  // namespace tools
}  // namespace compiler
}  // namespace gqe
