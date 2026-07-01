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

#include <gqe/compiler/Conversion/RelAlgToSCF/CudfTableScan.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/NVGPU/IR/NVGPUDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

mlir::LogicalResult CudfTableScanOpConversion::matchAndRewrite(
  relalg::CudfTableScanOp scanOp, DeclarativeConversionPatternRewriter& rewriter) const
{
  mlir::SmallVector<mlir::StringRef, 2> columnNames;
  mlir::SmallVector<mlir::StringAttr, 2> columnAttrs;
  mlir::SmallVector<mlir::Type, 2> columnTys;
  mlir::SmallVector<mlir::Type, 3> signature;
  mlir::SmallVector<mlir::StringAttr, 3> argNames;

  signature.push_back(rewriter.getIndexType());
  argNames.push_back(rewriter.getStringAttr("scan_end_index"));

  for (mlir::Attribute attr : scanOp.getLoadColumns()) {
    auto colDef   = llvm::cast<relalg::ColumnDefAttr>(attr);
    auto nameAttr = rewriter.getStringAttr(
      llvm::Twine(scanOp.getTableName()).concat(".").concat(colDef.getColumnName()));

    argNames.push_back(nameAttr);
    columnNames.push_back(colDef.getColumnName());
    columnAttrs.push_back(nameAttr);
    columnTys.push_back(colDef.getColumnType());

    // TODO: Use ptr.ptr type in ReadOp kernel arguments and convert to memref
    // with `ptr.from_ptr`, instead of using memref directly. The ptr dialect will
    // be introduced in MLIR 21.0
    mlir::Type memRefTy = mlir::MemRefType::get(
      mlir::ShapedType::kDynamic,
      colDef.getColumnType(),
      mlir::MemRefLayoutAttrInterface(),
      mlir::gpu::AddressSpaceAttr::get(rewriter.getContext(), mlir::gpu::AddressSpace::Global));
    signature.push_back(memRefTy);
  }

  // Create a kernel "__global__ void tablenameKernel(...args...)".
  auto kernelFnTy = rewriter.getFunctionType({/* function signature */ signature}, {});
  auto kernelFn =
    rewriter.create<mlir::gpu::GPUFuncOp>(rewriter.getUnknownLoc(),
                                          mlir::Twine(scanOp.getTableName()).concat("Kernel").str(),
                                          kernelFnTy);
  kernelFn->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), rewriter.getUnitAttr());

  // Set kernel argument naming hints for debugging IR.
  for (auto [arg, name] : llvm::zip(kernelFn.getBody().front().getArguments(), argNames)) {
    arg.setLoc(mlir::NameLoc::get(name));
  }

  rewriter.setInsertionPointToStart(&kernelFn.getBody().front());

  // Get kernel dimensions.
  mlir::gpu::GridDimOp gridDimOp =
    rewriter.create<mlir::gpu::GridDimOp>(rewriter.getUnknownLoc(), mlir::gpu::Dimension::x);
  mlir::gpu::BlockDimOp blockDimOp =
    rewriter.create<mlir::gpu::BlockDimOp>(rewriter.getUnknownLoc(), mlir::gpu::Dimension::x);

  // `mlir::gpu::SubgroupSizeOp` and friends aren't supported by the NVVM
  // target. At the same time, `mlir::NVVM::WarpSizeOp` requires us to specify
  // a return type that must be "a LLVM dialect-compatible type", which
  // `index` is not. Directly using LLVM types is very low-level, we'd have to
  // cast it to arith.I32 and then to index with
  // `builtin.unrealized_conversion_cast`.
  //
  // However, NVGPU defines the constant `kWarpSize`. Use the warp size to
  // calculate the warp ID.
  mlir::Value warpSizeOp = rewriter.create<mlir::index::ConstantOp>(
    mlir::NameLoc::get(rewriter.getStringAttr("warp_size")), ::kWarpSize);

  // Get kernel IDs.
  mlir::gpu::ThreadIdOp threadIdOp =
    rewriter.create<mlir::gpu::ThreadIdOp>(rewriter.getUnknownLoc(), mlir::gpu::Dimension::x);
  mlir::gpu::BlockIdOp blockIdOp =
    rewriter.create<mlir::gpu::BlockIdOp>(rewriter.getUnknownLoc(), mlir::gpu::Dimension::x);
  mlir::Value warpIdOp = rewriter.create<mlir::index::DivUOp>(
    mlir::NameLoc::get(rewriter.getStringAttr("warp_id")), threadIdOp, warpSizeOp);
  mlir::gpu::LaneIdOp laneIdOp =
    rewriter.create<mlir::gpu::LaneIdOp>(mlir::NameLoc::get(rewriter.getStringAttr("lane_id")),
                                         /* upper_bound = */ nullptr);

  // Calculate loop bounds.
  //
  // Loop using a warp. The warp leader thread determines the loop counter.
  //
  // `warpStartIndex = (blockDim * blockIdx + warpId) * warpSize`
  auto blockStartIndexOp = rewriter.create<mlir::index::MulOp>(
    mlir::NameLoc::get(rewriter.getStringAttr("block_start_index")), blockDimOp, blockIdOp);
  auto warpStartWithinBlockOp = rewriter.create<mlir::index::MulOp>(
    mlir::NameLoc::get(rewriter.getStringAttr("warp_start_within_block")), warpSizeOp, warpIdOp);
  auto warpStartIndexOp = rewriter.create<mlir::index::AddOp>(
    mlir::NameLoc::get(rewriter.getStringAttr("warp_start_index")),
    blockStartIndexOp,
    warpStartWithinBlockOp);
  auto loopStrideOp = rewriter.create<mlir::index::MulOp>(
    mlir::NameLoc::get(rewriter.getStringAttr("loop_stride")), gridDimOp, blockDimOp);

  mlir::Value numInputRowsOp = kernelFn.getArgument(0);

  /* auto tableLoopOp =  */ rewriter.create<mlir::scf::ForOp>(
    rewriter.getUnknownLoc(),
    warpStartIndexOp,
    numInputRowsOp,
    loopStrideOp,
    /* initArgs */ mlir::ValueRange(),
    [&kernelFn, &columnNames, &columnAttrs, &columnTys, &laneIdOp, &numInputRowsOp](
      mlir::OpBuilder& _rewriter,
      mlir::Location,
      mlir::Value warpOffsetOp,
      mlir::ValueRange /* loop-carried variables */) -> void {
      // `rewriter.create` must call into
      // `DeclarativeConversionPatternRewriter`, not `OpBuilder`.
      auto& rewriter = *static_cast<DeclarativeConversionPatternRewriter*>(&_rewriter);

      // Calculate thread offset based on the warp offset.
      auto threadOffset = rewriter.create<mlir::index::AddOp>(
        mlir::NameLoc::get(rewriter.getStringAttr("thread_offset")), warpOffsetOp, laneIdOp);

      // Calculate whether thread offset is within loop bounds and the thread
      // has a value.
      [[maybe_unused]] auto threadActiveFlag = rewriter.create<mlir::index::CmpOp>(
        mlir::NameLoc::get(rewriter.getStringAttr("thread_active_flag")),
        mlir::index::IndexCmpPredicate::SLT,
        threadOffset,
        numInputRowsOp);

      // Generate IU loads.
      //
      // Load instruction execution occurs only when the thread is active,
      // because otherwise the row index is out-of-bounds.
      //
      // Create a separate if-branch per load, because this should (?) enable
      // "code sinking" optimization, which moves a load to where it's used.
      // If code sinking doesn't occur, the if + load should be (?) lowered to
      // a PTX predicated load instruction.
      for (auto [index, columnName, columnAttr, columnTy] :
           llvm::enumerate(columnNames, columnAttrs, columnTys)) {
        // Load the IU from memory if within the input bounds, otherwise
        // poison the IU to enforce loop bounds.
        auto optionalLoadOp = rewriter.create<mlir::scf::IfOp>(
          rewriter.getUnknownLoc(),
          threadActiveFlag,
          [&](mlir::OpBuilder& rewriter, mlir::Location) {
            auto loadOp = rewriter.create<mlir::memref::LoadOp>(rewriter.getUnknownLoc(),
                                                                columnTy,
                                                                kernelFn.getArgument(index + 1),
                                                                mlir::ValueRange{threadOffset},
                                                                /* non-temporal = */ false);
            rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc(), mlir::ValueRange{loadOp});
          },
          [&](mlir::OpBuilder& rewriter, mlir::Location) {
            auto poisonOp = rewriter.create<mlir::ub::PoisonOp>(rewriter.getUnknownLoc(), columnTy);
            rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc(),
                                                mlir::ValueRange{poisonOp});
          });

        // Add IU to the result tuple.
        rewriter.getIUTuple().pushBack(optionalLoadOp.getResult(0), columnAttr);
      }

      rewriter.create<gqe::compiler::relalg::ForwardTupleStreamOp>(rewriter.getUnknownLoc(),
                                                                   threadActiveFlag);

      rewriter.create<mlir::scf::YieldOp>(rewriter.getUnknownLoc());
    });

  // Terminate the kernel
  rewriter.create<mlir::gpu::ReturnOp>(rewriter.getUnknownLoc());

  rewriter.eraseOp(scanOp);

  return mlir::success();
}

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
