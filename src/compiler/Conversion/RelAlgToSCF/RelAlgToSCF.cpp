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

#include <gqe/compiler/Conversion/RelAlgToSCF/RelAlgToSCF.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/CudfTableScan.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/Filter.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/GetIU.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/Map.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/ScalarAggregate.hpp>
#include <gqe/compiler/Conversion/RelAlgToSCF/Yield.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <utility>

namespace gqe {
namespace compiler {
#define GEN_PASS_DEF_CONVERTRELALGTOSCFPASS
#include <gqe/compiler/Conversion/Passes.h.inc>
}  // namespace compiler
}  // namespace gqe

namespace gqe {
namespace compiler {
namespace relalg {

void populateRelAlgToSCFConversionPatterns(mlir::RewritePatternSet& patterns)
{
  using namespace detail;

  patterns.add<CudfTableScanOpConversion,
               FilterOpConversion,
               GetIUOpConversion,
               MapOpConversion,
               ScalarAggregateOpConversion,
               YieldOpConversion>(patterns.getContext());
}

}  // namespace relalg
}  // namespace compiler
}  // namespace gqe

namespace {

struct ConvertRelAlgToSCFPass
  : public gqe::compiler::impl::ConvertRelAlgToSCFPassBase<ConvertRelAlgToSCFPass> {
  ConvertRelAlgToSCFPass() = default;

  void runOnOperation() override;
};

}  // namespace

void ConvertRelAlgToSCFPass::runOnOperation()
{
  mlir::RewritePatternSet patterns(&getContext());
  gqe::compiler::relalg::populateRelAlgToSCFConversionPatterns(patterns);

  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<gqe::compiler::relalg::RelAlgDialect>();

  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::gpu::GPUDialect>();
  target.addLegalDialect<mlir::index::IndexDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::scf::SCFDialect>();
  target.addLegalDialect<mlir::ub::UBDialect>();

  (void)gqe::compiler::relalg::detail::applyDeclarativeToImperativeConversion(
    getOperation(), target, std::move(patterns));
}
