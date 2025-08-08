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

#include <gqe/compiler/Tools/GPUModuleSerialization.hpp>

#include <mlir/Dialect/GPU/IR/CompilationInterfaces.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <optional>

namespace gqe {
namespace compiler {
namespace tools {

std::optional<llvm::SmallVector<char, 0>> serializeModule(mlir::ModuleOp& module,
                                                          mlir::gpu::CompilationTarget targetFormat)
{
  mlir::gpu::TargetOptions options("", {}, "", "", targetFormat);

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

  return std::nullopt;
}

}  // namespace tools
}  // namespace compiler
}  // namespace gqe
