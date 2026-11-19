/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>

#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

// LLVM 23's llvm::hash_combine() leaves its one-byte buffer uninitialized when it hashes nothing,
// which the generated computePropertiesHash of an op without properties does. The buffer is never
// read (the length is 0), but GCC warns about it when it inlines the call. This is fixed on LLVM
// main by https://github.com/llvm/llvm-project/pull/213578, which is not in any 23.x release.
// TODO: Remove once the minimum LLVM version includes that fix.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#define GET_OP_CLASSES
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.cpp.inc>
#pragma GCC diagnostic pop

namespace gqe {
namespace compiler {
namespace relalg {

mlir::LogicalResult MapOp::verifyRegions()
{
  // Check that the map expression yields at least one value, because the map
  // op must output a result row.
  if (getExpression().front().getTerminator()->getNumOperands() < 1) {
    return LogicalResult::failure();
  } else {
    return LogicalResult::success();
  }
}

}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
