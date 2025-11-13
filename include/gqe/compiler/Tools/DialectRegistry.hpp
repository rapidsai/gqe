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
