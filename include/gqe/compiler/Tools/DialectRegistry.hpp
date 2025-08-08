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

#include <mlir/IR/DialectRegistry.h>

namespace gqe {
namespace compiler {
namespace tools {

/**
 * @brief Registers dialect translations to LLVM IR used by GQE.
 *
 * References:
 * https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Tools/init_llvmir_translations.h
 *
 * @param registry The dialect registry.
 */
void registerLLVMIRTranslations(mlir::DialectRegistry& registry);

}  // namespace tools
}  // namespace compiler
}  // namespace gqe
