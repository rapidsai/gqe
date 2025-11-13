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

#include <gqe/compiler/Conversion/RelAlgToSCF/GetIU.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgTraits.hpp>

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

mlir::LogicalResult GetIUOpConversion::matchAndRewrite(
  relalg::GetIUOp IUOp, DeclarativeConversionPatternRewriter& rewriter) const
{
  int32_t iuIndex = IUOp.getIuRef().getIndex();

  mlir::OperandRange tupleStreams =
    IUOp->getParentWithTrait<relalg::OpTrait::TupleStreamConsumer>()->getOperands();

  mlir::Value iu = {};
  for (auto ts = tupleStreams.begin(); !iu && ts != tupleStreams.end(); ++ts) {
    iu = rewriter.getIUTuple(*ts).lookup(iuIndex).iu;
  }

  assert(iu && "expected a valid defining op for the IU");

  mlir::Type opReturnTy = IUOp.getResult().getType();
  assert(opReturnTy == iu.getType() && "expected that result type matches the type of the IU");

  rewriter.replaceOp(IUOp, iu);

  return mlir::success();
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
