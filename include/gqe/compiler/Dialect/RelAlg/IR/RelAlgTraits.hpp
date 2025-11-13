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

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgTypes.hpp>

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>

namespace gqe {
namespace compiler {
namespace relalg {
namespace OpTrait {

template <typename ConcreteType>
class TupleStreamConsumer : public mlir::OpTrait::TraitBase<ConcreteType, TupleStreamConsumer> {
 public:
  static mlir::LogicalResult verifyTrait(mlir::Operation* op)
  {
    bool isSuccess = llvm::any_of(
      op->getOperandTypes(), [](mlir::Type t) { return mlir::isa<relalg::TupleStreamType>(t); });
    return mlir::LogicalResult::success(isSuccess);
  }
};

}  // namespace OpTrait
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
