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

#include <climits>
#include <cstdint>

namespace gqe {
namespace compiler {

/**
 * @brief Declare the type used as MLIR IndexType in GQE.
 *
 * References:
 * https://mlir.llvm.org/docs/Dialects/Builtin/#indextype
 * https://mlir.llvm.org/docs/Dialects/IndexOps
 */
using mlirIndexType = int32_t;

/**
 * @brief The bitwidth of the MLIR index type.
 */
constexpr int32_t mlirIndexBitwidth = sizeof(int32_t) * CHAR_BIT;

}  // namespace compiler
}  // namespace gqe
