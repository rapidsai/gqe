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

#pragma once

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

#include <llvm/ADT/SmallVector.h>

#include <optional>

namespace gqe {
namespace compiler {
namespace tools {

/**
 * @brief Serializes the MLIR module to something CUDA understands (i.e., cubin, PTX).
 *
 * @pre The module must contain a single `GPUModuleOp`. However, this GPU module can contain
 * multiple kernels.
 *
 * This function directly returns the serialized module as a cubin or PTX code. In contrast,
 * `mlir::gpu::GpuModuleToBinaryPass` embeds the serialized module into the IR as a `gpu::BinaryOp`.
 *
 * References:
 * mlir/lib/Dialect/GPU/Transforms/ModuleToBinary.cpp
 * mlir/unittests/Target/LLVM/SerializeNVVMTarget.cpp
 *
 * @param[in] module The MLIR module to serialize. Must already be lowered to the MLIR `nvvm`
 * dialect.
 * @param[in] targetFormat Whether to output a cubin or PTX code. For cubin, the target attributes
 * must be set correctly for the GPU model. For PTX, CUDA will JIT compile the kernel itself.
 *
 * @return The serialized module.
 */
[[nodiscard]] std::optional<llvm::SmallVector<char, 0>> serializeModule(
  mlir::ModuleOp& module, mlir::gpu::CompilationTarget targetFormat);

}  // namespace tools
}  // namespace compiler
}  // namespace gqe
