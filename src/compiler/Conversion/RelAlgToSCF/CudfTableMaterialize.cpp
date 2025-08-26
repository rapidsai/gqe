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

#include <gqe/compiler/Conversion/RelAlgToSCF/CudfTableMaterialize.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/NamedIUTuple.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

mlir::LogicalResult CudfTableMaterializeOpConversion::matchAndRewrite(
  relalg::CudfTableMaterializeOp materializeOp,
  DeclarativeConversionPatternRewriter& rewriter) const
{
  auto tupleStreamOp =
    rewriter.setInsertionPointToTupleStream(materializeOp.getInput().getDefiningOp());

  mlir::SmallVector<mlir::Type> memRefTypes;
  mlir::SmallVector<mlir::StringAttr> columnNames;

  // Create kernel argument types for each output column
  auto kernelOp = rewriter.getBlock()->getParent()->getParentOfType<mlir::gpu::GPUFuncOp>();
  assert(kernelOp && "expected a parent gpu.func op");

  auto signature = kernelOp.getFunctionType().getInputs();
  mlir::SmallVector<mlir::Type> newSignature(signature);

  // Prepare memref types for each output column
  for (mlir::Attribute attr : materializeOp.getOutputColumns()) {
    auto colDef   = llvm::cast<relalg::ColumnDefAttr>(attr);
    auto nameAttr = rewriter.getStringAttr(
      llvm::Twine(materializeOp.getTableName()).concat(".").concat(colDef.getColumnName()));

    columnNames.push_back(nameAttr);

    mlir::Type memRefTy = mlir::MemRefType::get(
      mlir::ShapedType::kDynamic,
      colDef.getColumnType(),
      mlir::MemRefLayoutAttrInterface(),
      mlir::gpu::AddressSpaceAttr::get(rewriter.getContext(), mlir::gpu::AddressSpace::Global));

    memRefTypes.push_back(memRefTy);
  }

  // Generate atomic row counter for output indexing
  mlir::Type atomicCounterTy = mlir::MemRefType::get(
    /* shape = */ mlir::SmallVector<int64_t, 0>{},
    rewriter.getIndexIntegerType(),  // Use index integer type for atomic operations
    mlir::MemRefLayoutAttrInterface(),
    mlir::gpu::AddressSpaceAttr::get(rewriter.getContext(), mlir::gpu::AddressSpace::Global));

  // Collect all new arguments first, then update signature once
  newSignature.push_back(rewriter.getIndexType());  // output table capacity
  for (auto memRefTy : memRefTypes) {
    newSignature.push_back(memRefTy);  // columns
  }
  newSignature.push_back(atomicCounterTy);  // counter

  // Single signature update
  auto kernelFnTy = rewriter.getFunctionType({newSignature}, {});
  kernelOp.setFunctionType(kernelFnTy);

  // Add arguments to kernel function
  auto& kernelBlock = kernelOp.getBlocks().front();
  kernelBlock.addArgument(rewriter.getIndexType(),
                          rewriter.getUnknownLoc());  // output table capacity
  for (auto memRefTy : memRefTypes) {
    kernelBlock.addArgument(memRefTy, rewriter.getUnknownLoc());
  }
  kernelBlock.addArgument(atomicCounterTy, rewriter.getUnknownLoc());

  // Get the newly added arguments for output columns
  auto tableCapacityArg = kernelOp.getArgument(signature.size());  // output table capacity
  mlir::SmallVector<mlir::Value> outputColumnArgs;
  for (size_t i = 0; i < memRefTypes.size(); ++i) {
    outputColumnArgs.push_back(kernelOp.getArgument(signature.size() + 1 + i));
  }
  auto atomicCounterArg = kernelOp.getArgument(signature.size() + 1 + memRefTypes.size());

  rewriter.create<mlir::scf::IfOp>(
    rewriter.getUnknownLoc(),
    tupleStreamOp.getIsThreadActive(),
    [&materializeOp, &outputColumnArgs, &atomicCounterArg, &tableCapacityArg, &columnNames](
      mlir::OpBuilder& _rewriter, mlir::Location) {
      // `rewriter.create` must call into
      // `DeclarativeConversionPatternRewriter`, not `OpBuilder`.
      auto& rewriter = *static_cast<DeclarativeConversionPatternRewriter*>(&_rewriter);

      // Get the input IU tuple, which is created by the parent op.
      relalg::NamedIUTuple& iuTuple = rewriter.getIUTuple(materializeOp.getInput());

      // Atomically allocate a row index for this thread
      mlir::Value oneOp = rewriter.create<mlir::arith::ConstantIntOp>(
        rewriter.getUnknownLoc(), rewriter.getIndexIntegerType(), 1);
      auto atomicRowIndexIntOp =
        rewriter.create<mlir::memref::AtomicRMWOp>(rewriter.getUnknownLoc(),
                                                   rewriter.getIndexIntegerType(),
                                                   mlir::arith::AtomicRMWKind::addi,
                                                   oneOp,
                                                   atomicCounterArg,
                                                   /* memref indices = */ mlir::ValueRange{});

      // Convert integer to index for memref indexing
      auto atomicRowIndexOp = rewriter.create<mlir::arith::IndexCastOp>(
        rewriter.getUnknownLoc(), rewriter.getIndexType(), atomicRowIndexIntOp.getResult());

      // Assert that we don't overflow the table capacity
      // The atomic operation returns the old value (our row index), which must be < capacity
      auto tableCapacityIntOp = rewriter.create<mlir::arith::IndexCastOp>(
        rewriter.getUnknownLoc(), rewriter.getIndexIntegerType(), tableCapacityArg);
      auto overflowCheckOp = rewriter.create<mlir::arith::CmpIOp>(rewriter.getUnknownLoc(),
                                                                  mlir::arith::CmpIPredicate::slt,
                                                                  atomicRowIndexIntOp.getResult(),
                                                                  tableCapacityIntOp.getResult());
      rewriter.create<mlir::cf::AssertOp>(
        rewriter.getUnknownLoc(),
        overflowCheckOp.getResult(),
        rewriter.getStringAttr("Table capacity overflow: too many rows to materialize"));

      // Store each column value to the output table
      for (const auto& [index, outputColumnArg, columnName] :
           llvm::enumerate(outputColumnArgs, columnNames)) {
        // Look up the IU value for this column
        // TODO: Handle column mapping using a MapOp outside of the CudfTableMaterializeOp
        // For now, assume the IU index matches the column index
        mlir::Value iuValue = iuTuple.lookup(index).iu;
        assert(iuValue && "expected a valid IU value for column");

        // Sanity check: Ensure IU value type matches column data type
        auto memrefType          = llvm::cast<mlir::MemRefType>(outputColumnArg.getType());
        auto expectedElementType = memrefType.getElementType();
        auto actualValueType     = iuValue.getType();
        assert(actualValueType == expectedElementType &&
               "IU value type must match column element type for materialization");

        // Store the value at the allocated row index
        rewriter.create<mlir::memref::StoreOp>(
          rewriter.getUnknownLoc(), iuValue, outputColumnArg, mlir::ValueRange{atomicRowIndexOp});
      }

      rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc());
    });

  rewriter.eraseOp(materializeOp);

  return mlir::success();
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
