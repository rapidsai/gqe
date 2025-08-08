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

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/NamedIUTuple.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/IR/Builders.h>
#include <mlir/IR/Iterators.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Rewrite/PatternApplicator.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "declarative-to-imperative-conversion-pass"

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

mlir::LogicalResult applyDeclarativeToImperativeConversion(
  mlir::Operation* rootOp,
  const mlir::ConversionTarget& target,
  const mlir::FrozenRewritePatternSet& patterns)
{
  if (!rootOp) { return mlir::success(); }

  mlir::SmallVector<mlir::Operation*> toConvert;
  rootOp->walk<mlir::WalkOrder::PostOrder,
               mlir::ForwardDominanceIterator</* NoGraphRegions = */ false>>(
    [&](mlir::Operation* op) {
      auto legalityInfo = target.isLegal(op);

      if (!legalityInfo.has_value()) { toConvert.push_back(op); }

      if (legalityInfo && legalityInfo->isRecursivelyLegal) return mlir::WalkResult::skip();
      return mlir::WalkResult::advance();
    });

  mlir::MLIRContext* context = rootOp->getContext();
  relalg::NamedIUTupleCollection iuTupleCollection;
  DeclarativeConversionPatternRewriter rewriter(context, &iuTupleCollection);
  mlir::PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  for (mlir::Operation* op : toConvert) {
    if (rewriter.isErased(op)) { continue; }

    LLVM_DEBUG(llvm::dbgs() << "Visiting op: " << mlir::debugString(*op) << "\n");

    rewriter.setCurrentOp(op);

    if (applicator.matchAndRewrite(op, rewriter).failed()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to rewrite op " << mlir::debugString(*op) << "\n");
      return mlir::failure();
    }
  }

  rewriter.applyRewrites(toConvert);

  return mlir::success();
}

DeclarativeConversionPatternRewriter::DeclarativeConversionPatternRewriter(
  mlir::MLIRContext* ctx, relalg::NamedIUTupleCollection* iuTupleCollection)
  : mlir::PatternRewriter(ctx), currentOp(nullptr), iuTupleCollection(iuTupleCollection)
{
}

DeclarativeConversionPatternRewriter::~DeclarativeConversionPatternRewriter() {}

void DeclarativeConversionPatternRewriter::replaceOp(mlir::Operation* op,
                                                     mlir::ValueRange newValues)
{
  assert(op->getNumResults() == newValues.size() && "incorrect # of replacement values");

  for (auto [result, replacement] : llvm::zip_equal(op->getResults(), newValues)) {
    mapping.map(result, std::move(replacement));
  }

  eraseOp(op);
}

void DeclarativeConversionPatternRewriter::replaceOp(mlir::Operation* op, mlir::Operation* newOp)
{
  assert(op && newOp && "expected non-null op");
  replaceOp(op, newOp->getResults());
}

void DeclarativeConversionPatternRewriter::eraseOp(mlir::Operation* op) { erasedOps.insert(op); }

void DeclarativeConversionPatternRewriter::eraseBlock(mlir::Block* block)
{
  // Erase all ops in the block.
  for (mlir::Operation& op : *block) {
    eraseOp(&op);
  }

  // Keep operations in block live until rewrite is applied, as other operations
  // might still depend on their values.
  erasedBlocks.push_back(block);

  // Unlink block in parent.
  block->getParent()->getBlocks().remove(block);
}

relalg::ForwardTupleStreamOp DeclarativeConversionPatternRewriter::setInsertionPointToTupleStream(
  mlir::Operation* srcOp)
{
  auto it = tupleStreamContinuation.find(srcOp);
  assert(it != tupleStreamContinuation.end() && "expected that `srcOp` forwarded a tuple stream");

  auto tsOp = it->second;

  mlir::OpBuilder::InsertPoint insertPoint(tsOp->getBlock(), mlir::Block::iterator(tsOp));
  restoreInsertionPoint(insertPoint);

  return tsOp;
}

NamedIUTuple& DeclarativeConversionPatternRewriter::getIUTuple()
{
  return iuTupleCollection->getNamedIUTuple(currentOp->getResult(0));
}

NamedIUTuple& DeclarativeConversionPatternRewriter::getIUTuple(mlir::Value tupleStream)
{
  return iuTupleCollection->getNamedIUTuple(tupleStream);
}

void DeclarativeConversionPatternRewriter::appendIUTuple(mlir::Value otherTupleStream)
{
  // Before appending, create the IU tuple if it doens't already exist.
  std::ignore = iuTupleCollection->getNamedIUTuple(currentOp->getResult(0));
  iuTupleCollection->append(otherTupleStream, currentOp->getResult(0));
}

void DeclarativeConversionPatternRewriter::setCurrentOp(mlir::Operation* op) { currentOp = op; }

bool DeclarativeConversionPatternRewriter::isErased(mlir::Operation* op)
{
  return erasedOps.contains(op);
}

void DeclarativeConversionPatternRewriter::applyRewrites(
  mlir::ArrayRef<mlir::Operation*> conversionOrder)
{
  // Erase ops in reverse order of dependency chain, because MLIR asserts on
  // dangling dependencies.
  for (auto it = conversionOrder.rbegin(); it != conversionOrder.rend(); ++it) {
    auto op = *it;

    if (erasedOps.contains(op)) {
      for (auto result : op->getResults()) {
        auto replacement = mapping.lookupOrNull(result);
        if (replacement) {
          LLVM_DEBUG(llvm::dbgs() << "Replacing op: " << mlir::debugString(*op) << "\n");
          // Forward to the parent `replaceAllOpUsesWith`.
          mlir::PatternRewriter::replaceAllOpUsesWith(op, replacement);
        }
      }

      // Print spaces for alignment.
      LLVM_DEBUG(llvm::dbgs() << "Erasing   op: " << mlir::debugString(*op) << "\n");
      // Forward to the parent `eraseOp`.
      mlir::PatternRewriter::eraseOp(op);
    }
  }

  // Clean up ForwardTupleStreamOps. These are created during conversion, and
  // thus are not present in the `toConvert` list.
  for (auto op : forwardTupleStreamOps) {
    LLVM_DEBUG(llvm::dbgs() << "Cleaning  op: " << mlir::debugString(*op) << "\n");
    // Forward to the parent `eraseOp`.
    mlir::PatternRewriter::eraseOp(op);
  }

  for (auto block : erasedBlocks) {
    llvm::errs() << "Erasing block: " << block << "\n";
    // Forward to the parent `eraseBlock`.
    mlir::PatternRewriter::eraseBlock(block);
  }
}

mlir::LogicalResult DeclarativeConversionPattern::matchAndRewrite(
  mlir::Operation* op, mlir::PatternRewriter& rewriter) const
{
  auto& dialectRewriter = static_cast<DeclarativeConversionPatternRewriter&>(rewriter);

  return matchAndRewrite(op, dialectRewriter);
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
