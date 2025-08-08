/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;

#define GET_OP_CLASSES
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.cpp.inc>

namespace gqe {
namespace compiler {
namespace relalg {

mlir::LogicalResult MapOp::verifyRegions()
{
  // Check that the map expression yields at least one value, because the map
  // op must output a result row.
  if (getExpression().front().getTerminator()->getNumOperands() < 1) {
    return LogicalResult::failure();
  } else {
    return LogicalResult::success();
  }
}

}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
