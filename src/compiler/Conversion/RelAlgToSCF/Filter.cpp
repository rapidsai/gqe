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

#include <gqe/compiler/Conversion/RelAlgToSCF/Filter.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

mlir::LogicalResult FilterOpConversion::matchAndRewrite(
  relalg::FilterOp filterOp, DeclarativeConversionPatternRewriter& rewriter) const
{
  auto tupleStreamOp = rewriter.setInsertionPointToTupleStream(filterOp.getInput().getDefiningOp());
  auto threadActiveFlag = tupleStreamOp.getIsThreadActive();

  // The filter predicate evaluation occurs only when the thread is active,
  // because the predicate expression could contain `GetIUOp`s. As the
  // thread is participating in a warp, the thread's row index could be
  // ouf-of-bounds. In this case, GetIUOp would trigger an invalid memory
  // access.
  //
  // Since the predicate should return a Boolean, we typecheck the return
  // argument by specifying a Boolean as the Region's return type.
  auto executePredicateOp = rewriter.create<mlir::scf::IfOp>(rewriter.getUnknownLoc(),
                                                             mlir::TypeRange{rewriter.getI1Type()},
                                                             threadActiveFlag,
                                                             /* withElseRegion = */ true);
  auto elseBuilder        = executePredicateOp.getElseBodyBuilder();

  // If threadActiveFlag is `false`, the else block sets newThreadActiveFlag
  // to `false`.
  auto falseConstOp = elseBuilder.create<mlir::arith::ConstantIntOp>(
    elseBuilder.getUnknownLoc(), elseBuilder.getI1Type(), false);
  elseBuilder.create<mlir::scf::YieldOp>(elseBuilder.getUnknownLoc(),
                                         mlir::ValueRange{falseConstOp});

  // Push filter predicate into a the `then` body.
  executePredicateOp.getThenRegion().takeBody(filterOp.getPredicate());

  auto newThreadActiveFlag = rewriter.create<mlir::arith::AndIOp>(
    rewriter.getUnknownLoc(), executePredicateOp.getResult(0), threadActiveFlag);

  rewriter.appendIUTuple(filterOp.getInput());
  rewriter.create<relalg::ForwardTupleStreamOp>(rewriter.getUnknownLoc(), newThreadActiveFlag);

  rewriter.eraseOp(filterOp);

  return mlir::success();
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
