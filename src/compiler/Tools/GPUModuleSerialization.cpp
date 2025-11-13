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
