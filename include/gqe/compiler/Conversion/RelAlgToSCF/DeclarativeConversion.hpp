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

#pragma once

#include <gqe/compiler/Conversion/RelAlgToSCF/NamedIUTuple.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <cassert>
#include <type_traits>
#include <utility>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

/**
 * @brief Apply the declarative to imperative conversion on an op graph.
 *
 * @param[in] rootOp The graph's root op.
 * @param[in] target Description of the legal and illegal target dialects.
 * @param[in] patterns The rewrite patterns to apply.
 *
 * @return True if the conversion succeeded.
 */
mlir::LogicalResult applyDeclarativeToImperativeConversion(
  mlir::Operation* rootOp,
  const mlir::ConversionTarget& target,
  const mlir::FrozenRewritePatternSet& patterns);

/**
 * @brief A rewriter that supports the declarative to imperative rewrite patterns.
 *
 * The declarative to imperative rewriter maintains the tuple stream state.
 *
 * A rewriter generally tracks the current insert location. The declarative to imperative rewriter
 * extends this to set the insert location based on the tuple stream. In lieu with the tuple stream,
 * the IU tuples are maintained as well.
 *
 * @copydoc
 */
class DeclarativeConversionPatternRewriter final : public mlir::PatternRewriter {
  friend mlir::LogicalResult applyDeclarativeToImperativeConversion(
    mlir::Operation* op,
    const mlir::ConversionTarget& target,
    const mlir::FrozenRewritePatternSet& patterns);

 public:
  explicit DeclarativeConversionPatternRewriter(mlir::MLIRContext* ctx,
                                                relalg::NamedIUTupleCollection* iuTupleCollection);

  ~DeclarativeConversionPatternRewriter() override;

  /**
   * @brief Replace the given operation with the new values.
   *
   * The number of op results and replacement values must match. The types may differ: the dialect
   * conversion driver will reconcile any surviving type mismatches at the end of the conversion
   * process with source materializations. The given operation is erased.
   *
   * @param[in] on The op to replace.
   * @param newValues The values to replace the op with.
   */
  void replaceOp(mlir::Operation* op, mlir::ValueRange newValues) override;

  /**
   * @brief Replace the given operation with the results of the new op.
   *
   * The number of op results must match. The types may differ: the dialect conversion driver will
   * reconcile any surviving type mismatches at the end of the conversion process with source
   * materializations. The original operationis erased.
   *
   * @param[in] op The op to replace.
   * @param[in] newOp The op to replace it with.
   */
  void replaceOp(mlir::Operation* op, mlir::Operation* newOp) override;

  /**
   * @brief Erase a dead operation.
   *
   * The uses of this operation *must* be made dead by the end of the conversion process.
   *
   * The erasing is lazy in order to keep references to the op alive during the rewrite pass. Only
   * after all ops have been rewritten is the erasure executed.
   *
   * @param[in] op The op to erase.
   */
  void eraseOp(mlir::Operation* op) override;

  /**
   * @brief Create an operation of specific op type at the current insertion point.
   *
   * Hides the `PatternRewriter::create()` method (inherited from `OpBuilder`). The purpose is to
   * catch when a ForwardTupleStreamOp is created, and save that op for restoration at a later point
   * in time.
   *
   * @param location The op location information.
   * @param args The arguments forwarded to the op builder method.
   *
   * @return The newly created op.
   */
  template <typename OpTy, typename... Args>
  OpTy create(mlir::Location location, Args&&... args)
  {
    assert(currentOp && "expected current op to be set by rewriter's owner");

    // Forward the call to the parent `create`.
    auto newOp = mlir::PatternRewriter::create<OpTy>(location, std::forward<Args>(args)...);

    // Save point at node after the new op.
    if constexpr (std::is_same<OpTy, gqe::compiler::relalg::ForwardTupleStreamOp>()) {
      auto tsOp = static_cast<gqe::compiler::relalg::ForwardTupleStreamOp>(newOp);

      auto entry           = std::make_pair(currentOp, tsOp);
      bool has_tuplestream = !tupleStreamContinuation.insert(std::move(entry)).second;
      assert(!has_tuplestream && "expected only a single tuplestream");

      forwardTupleStreamOps.push_back(newOp);
    }

    return newOp;
  }

  /**
   * @brief Erase all operations in a block.
   *
   * FIXME:  Not yet implemented.
   *
   * @param block The block to erase.
   */
  void eraseBlock(mlir::Block* block) override;

  /**
   * @brief Sets the insertion point to the node after the tuple stream forwarded by the source op.
   *
   * @param srcOp The source op.
   *
   * @return The `ForwardTupleStreamOp`.
   */
  [[nodiscard]] relalg::ForwardTupleStreamOp setInsertionPointToTupleStream(mlir::Operation* srcOp);

  /**
   * @brief Return the named IU tuple of the current operation.
   *
   * @return The named IU tuple.
   */
  [[nodiscard]] relalg::NamedIUTuple& getIUTuple();

  /**
   * @brief Return the named IU tuple of the tuple stream.
   *
   * @param tupleStream The tuple stream.
   *
   * @return The name IU tuple.
   */
  [[nodiscard]] relalg::NamedIUTuple& getIUTuple(mlir::Value tupleStream);

  /**
   * @brief Append the IU tuple of the other tuple stream to the tuple stream of
   * the current op.
   *
   * @param otherTupleStream The other tuple stream.
   */
  void appendIUTuple(mlir::Value otherTupleStream);

 private:
  void setCurrentOp(mlir::Operation* op);

  /**
   * @brief Check if the op has been erased.
   *
   * @param[in] op The op to test.
   *
   * @return True if the op has been erased.
   */
  [[nodiscard]] bool isErased(mlir::Operation* op);

  void applyRewrites(mlir::ArrayRef<mlir::Operation*> conversionOrder);

  mlir::Operation* currentOp;  ///< The Op currently being converted.

  relalg::NamedIUTupleCollection*
    iuTupleCollection;  ///< The IU tuple collection of the conversion pass.

  mlir::IRMapping mapping;  ///< Mapping of old values to new values.

  mlir::DenseMap<mlir::Operation*, relalg::ForwardTupleStreamOp>
    tupleStreamContinuation;  ///< Mapping of Op to the tuple stream continuation point for its
                              ///< parent Op.

  mlir::SmallVector<mlir::Block*> erasedBlocks;  ///< Erased blocks that are being kept alive.

  mlir::SmallPtrSet<mlir::Operation*, 0> erasedOps;  ///< Erased ops that are being kept alive.

  mlir::SmallVector<mlir::Operation*>
    forwardTupleStreamOps;  ///< Forward tuple streams to be cleaned up.
};

/**
 * @brief A declarative conversion pattern.
 *
 * An abstract base class for rewriting a declarative source dialect to
 * imperative target dialects.
 *
 * # Design notes
 *
 * The declarative conversion pattern inherits from `RewritePattern`, which is
 * generally the basis for MLIR transformations. It does not use
 * `ConversionPattern`, which has unrelated functionality. Differences include:
 *
 * - Declarative rewrites nest ops into each other by following the tuple
 *   stream. Thus, rewrites erase the op instead of replacing its result value
 *   (i.e., the tuple stream), and emit their result by creating an IU tuple.
 * - Declarative rewrites are one-shot. The rewriter does not support rewrite
 *   failures with rollback to the previous state.
 *
 * References:
 * https://mlir.llvm.org/docs/PatternRewriter/
 * https://mlir.llvm.org/docs/DialectConversion/
 * https://discourse.llvm.org/t/rfc-a-new-one-shot-dialect-conversion-driver/79083
 */
class DeclarativeConversionPattern : public mlir::RewritePattern {
 public:
  /**
   * @brief Wrapper around the RewritePattern method that passes a
   * DeclarativeConversionPatternRewriter.
   *
   * @param op The op to rewrite.
   * @param rewriter The declarative conversion pattern rewriter.
   */
  [[nodiscard]] virtual mlir::LogicalResult matchAndRewrite(
    mlir::Operation* op, DeclarativeConversionPatternRewriter& rewriter) const = 0;

  /**
   * @copydoc
   */
  [[nodiscard]] mlir::LogicalResult matchAndRewrite(mlir::Operation* op,
                                                    mlir::PatternRewriter& rewriter) const final;

 protected:
  /// Inherit the base constructors from `RewritePattern`.
  using RewritePattern::RewritePattern;

  /**
   * @brief Construct a conversion pattern with the given converter.
   *
   * The remaining arguments are forwarded to RewritePattern.
   *
   * @param args The arguments list.
   */
  template <typename... Args>
  DeclarativeConversionPattern(Args&&... args) : mlir::RewritePattern(std::forward<Args>(args)...)
  {
  }
};

/**
 * @brief An extension of DeclarativeConversionPattern that casts the op to the given type.
 */
template <typename SourceOp>
class OpDeclarativeConversionPattern : public DeclarativeConversionPattern {
 public:
  /**
   * @copydoc
   */
  OpDeclarativeConversionPattern(mlir::MLIRContext* context, mlir::PatternBenefit benefit = 1)
    : DeclarativeConversionPattern(SourceOp::getOperationName(), benefit, context)
  {
  }

  /**
   * @brief Wrapper around the DeclarativeConversionPattern method that passes
   * the derived op type.
   *
   * @param op The op to rewrite.
   * @param rewriter The declarative conversion pattern rewriter.
   *
   * @return True if the rewrite succeeded.
   */
  [[nodiscard]] mlir::LogicalResult matchAndRewrite(
    mlir::Operation* op, DeclarativeConversionPatternRewriter& rewriter) const final
  {
    return matchAndRewrite(mlir::cast<SourceOp>(op), rewriter);
  }

  /**
   * @brief Method that operates on the SourceOp type.
   *
   * Must be overridden by the derived pattern class.
   *
   * @param op The op to rewrite.
   * @param rewriter The declarative conversion pattern rewriter.
   *
   * @return True if the rewrite succeeded.
   */
  [[nodiscard]] virtual mlir::LogicalResult matchAndRewrite(
    SourceOp op, DeclarativeConversionPatternRewriter& rewriter) const = 0;

 private:
  // Fix compiler warning "DeclarativeConversionPattern<X>::matchAndRewrite
  // hides overloaded virtual function".
  //
  // GCC documentation:
  // > In cases where the different signatures are not an accident, the
  // simplest > solution is to add a using-declaration to the derived class to
  // un-hide > the base function, e.g. add using A::f; to B.
  //
  // References:
  // https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Transforms/DialectConversion.h#L692
  // https://gcc.gnu.org/onlinedocs/gcc-13.1.0/gcc/C_002b_002b-Dialect-Options.html#index-Woverloaded-virtual
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109740
  using DeclarativeConversionPattern::matchAndRewrite;
};

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
