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
