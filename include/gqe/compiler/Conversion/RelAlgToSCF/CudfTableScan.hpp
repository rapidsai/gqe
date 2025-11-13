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

#pragma once

#include <gqe/compiler/Conversion/RelAlgToSCF/DeclarativeConversion.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/Support/LLVM.h>

namespace gqe {
namespace compiler {
namespace relalg {
namespace detail {

struct CudfTableScanOpConversion : public OpDeclarativeConversionPattern<relalg::CudfTableScanOp> {
  /// Inherit constructors from OpDeclarativeConversionPattern.
  using OpDeclarativeConversionPattern<relalg::CudfTableScanOp>::OpDeclarativeConversionPattern;

  mlir::LogicalResult matchAndRewrite(
    relalg::CudfTableScanOp scanOp, DeclarativeConversionPatternRewriter& rewriter) const override;
};

}  // namespace detail
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
