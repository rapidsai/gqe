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

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.hpp>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

using namespace gqe::compiler::relalg;

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOpsDialect.cpp.inc>

//===----------------------------------------------------------------------===//
// RelAlg dialect.
//===----------------------------------------------------------------------===//

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOpsEnums.cpp.inc>

#define GET_ATTRDEF_CLASSES
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOpsAttrs.cpp.inc>

void RelAlgDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOps.cpp.inc>
    >();
  addTypes<
#define GET_TYPEDEF_LIST
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc>
    >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOpsAttrs.cpp.inc>
    >();
}
