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

#include <gqe/compiler/Conversion/RelAlgToSCF/ScalarAggregate.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/NamedIUTuple.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/TypeRange.h>
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

mlir::LogicalResult ScalarAggregateOpConversion::matchAndRewrite(
  relalg::ScalarAggregateOp aggregateOp, DeclarativeConversionPatternRewriter& rewriter) const
{
  auto tupleStreamOp =
    rewriter.setInsertionPointToTupleStream(aggregateOp.getInput().getDefiningOp());

  // TODO: Programmatically extract number and type of results from
  // `aggregateOp.getFnArgs()`. Type is a function of input IU type and
  // `AggregateFunction`. For example, INT + SUM = BIGINT (check this!).
  assert(aggregateOp.getFnArgs().size() == 1 &&
         "multiple aggregate functions are not implemented yet");
  mlir::Type aggregateResultTy = rewriter.getF64Type();

  // TODO: Kernel arguments should be appended with a cleaner mechanism.
  mlir::Type memRefTy = mlir::MemRefType::get(
    /* shape = */ mlir::SmallVector<int64_t, 0>{},
    aggregateResultTy,
    mlir::MemRefLayoutAttrInterface(),
    mlir::gpu::AddressSpaceAttr::get(rewriter.getContext(), mlir::gpu::AddressSpace::Global));

  auto kernelOp = rewriter.getBlock()->getParent()->getParentOfType<mlir::gpu::GPUFuncOp>();
  assert(kernelOp && "expected a parent gpu.func op");

  auto signature = kernelOp.getFunctionType().getInputs();
  mlir::SmallVector<mlir::Type, 0> newSignature(signature);
  newSignature.push_back(memRefTy);
  auto kernelFnTy = rewriter.getFunctionType({newSignature}, {});
  kernelOp.setFunctionType(kernelFnTy);
  kernelOp.getBlocks().front().addArgument(memRefTy, rewriter.getUnknownLoc());

  auto aggregateArg = kernelOp.getArgument(signature.size());

  rewriter.create<mlir::scf::IfOp>(
    rewriter.getUnknownLoc(),
    tupleStreamOp.getIsThreadActive(),
    [&aggregateOp, &aggregateArg, &aggregateResultTy](mlir::OpBuilder& _rewriter, mlir::Location) {
      // `rewriter.create` must call into
      // `DeclarativeConversionPatternRewriter`, not `OpBuilder`.
      auto& rewriter = *static_cast<DeclarativeConversionPatternRewriter*>(&_rewriter);

      // Get the input IU tuple, which is created by the parent op.
      relalg::NamedIUTuple& iuTuple = rewriter.getIUTuple(aggregateOp.getInput());

      // Lookup IU of the current IU reference and generate the aggregation
      // step code.
      for (auto arg : aggregateOp.getFnArgs()) {
        auto typedArg   = mlir::cast<relalg::AggregateFnArgAttr>(arg);
        int32_t iuIndex = typedArg.getIuRef().getIndex();

        mlir::Value iu = iuTuple.lookup(iuIndex).iu;
        assert(iu && "expected a valid defining op for the IU");

        switch (typedArg.getAggFn()) {
          case relalg::AggregateFunction::Count: {
            // TODO: This code path should only be used when the IU is
            // nullable. In that case, increment the count when not null.
            mlir::Value oneOp = rewriter.create<mlir::arith::ConstantIntOp>(
              rewriter.getUnknownLoc(), rewriter.getI64Type(), 1);
            rewriter.create<mlir::memref::AtomicRMWOp>(rewriter.getUnknownLoc(),
                                                       mlir::arith::AtomicRMWKind::addi,
                                                       oneOp,
                                                       aggregateArg,
                                                       /* memref indices = */ mlir::ValueRange{});
            break;
          }
          case relalg::AggregateFunction::Sum: {
            auto iuCastOp = iu.getType() != aggregateResultTy
                              ? rewriter.create<mlir::arith::ExtFOp>(
                                  rewriter.getUnknownLoc(), aggregateResultTy, iu)
                              : iu;
            // The memref is a scalar value. Therefore, `indices` is an empty
            // `ValueRange`.
            rewriter.create<mlir::memref::AtomicRMWOp>(rewriter.getUnknownLoc(),
                                                       aggregateResultTy,
                                                       mlir::arith::AtomicRMWKind::addf,
                                                       iuCastOp,
                                                       aggregateArg,
                                                       /* memref indices = */ mlir::ValueRange{});
            break;
          }
        }
      }

      rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc());
    });

  rewriter.eraseOp(aggregateOp);

  return mlir::success();
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
