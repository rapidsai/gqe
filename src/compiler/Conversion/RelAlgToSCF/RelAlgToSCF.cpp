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

#include <gqe/compiler/Conversion/RelAlgToSCF/RelAlgToSCF.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/CudfTableMaterialize.hpp>
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

  patterns.add<CudfTableMaterializeOpConversion,
               CudfTableScanOpConversion,
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
