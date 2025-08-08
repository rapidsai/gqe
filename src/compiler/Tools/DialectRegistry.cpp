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
