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
