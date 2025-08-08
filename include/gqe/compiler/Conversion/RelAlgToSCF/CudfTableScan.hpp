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
