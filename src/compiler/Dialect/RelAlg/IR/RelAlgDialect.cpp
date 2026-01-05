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

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgAttrs.hpp>
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
