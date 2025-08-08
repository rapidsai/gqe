//===- RelAlgDialect.td - Relational Algebra dialect ----*- tablegen -*----===//
//
// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier:
// LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//===----------------------------------------------------------------------===//

#ifndef RELALG_DIALECT
#define RELALG_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RelAlg dialect definition.
//===----------------------------------------------------------------------===//

def RelAlg_Dialect : Dialect {
  let name = "relalg";
  let summary = "A relational algebra dialect.";

  let description = [{
    RelAlg is a declarative dialect that represents relational algebra
    operators.
  }];

  let cppNamespace = "::gqe::compiler::relalg";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base RelAlg operation definition.
//===----------------------------------------------------------------------===//

class RelAlg_Op<string mnemonic, list<Trait> traits = []>
    : Op<RelAlg_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// RelAlg trait definitions.
//===----------------------------------------------------------------------===//

class RelAlg_Trait<string mnemonic> : NativeOpTrait<mnemonic> {
  let cppNamespace = "::gqe::compiler::relalg::OpTrait";
}

#endif // RELALG_DIALECT