/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/compiler/Conversion/RelAlgToSCF/Map.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/SmallVectorExtras.h>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

mlir::LogicalResult MapOpConversion::matchAndRewrite(
  relalg::MapOp mapOp, DeclarativeConversionPatternRewriter& rewriter) const
{
  auto tupleStreamOp    = rewriter.setInsertionPointToTupleStream(mapOp.getInput().getDefiningOp());
  auto threadActiveFlag = tupleStreamOp.getIsThreadActive();

  // The map expression produces the IUs. Get the IUs' types.
  auto iuTys = mapOp.getExpression().front().getTerminator()->getOperandTypes();

  // The map expression evaluation occurs only when the thread is active,
  // because the expression could contain `GetIUOp`s.
  //
  // The expression determines the return value of the if-branch.
  auto executeExpressionOp = rewriter.create<mlir::scf::IfOp>(rewriter.getUnknownLoc(),
                                                              iuTys,
                                                              threadActiveFlag,
                                                              /* withElseRegion = */ true);

  // If threadActiveFlag is `false`, the else block poisons the IUs.
  auto elseBuilder = executeExpressionOp.getElseBodyBuilder();
  auto poisonIUs   = llvm::map_to_vector(iuTys, [&elseBuilder](mlir::Type iuTy) -> mlir::Value {
    return elseBuilder.create<mlir::ub::PoisonOp>(elseBuilder.getUnknownLoc(), iuTy);
  });
  elseBuilder.create<mlir::scf::YieldOp>(elseBuilder.getUnknownLoc(), poisonIUs);

  // Push map expression into a the `then` body.
  executeExpressionOp.getThenRegion().takeBody(mapOp.getExpression());

  // Add the produced IUs to the result tuple.
  for (auto iu : executeExpressionOp.getResults()) {
    rewriter.getIUTuple().pushBack(iu);
  }

  // Return the previous thread active flag, as the expression must always yield
  // a row.
  rewriter.create<relalg::ForwardTupleStreamOp>(rewriter.getUnknownLoc(), threadActiveFlag);

  rewriter.eraseOp(mapOp);

  return mlir::success();
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
