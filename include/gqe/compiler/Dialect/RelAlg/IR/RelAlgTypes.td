//===- RelAlgTypes.td - Relational Algebra dialect types -*- tablegen -*---===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//===----------------------------------------------------------------------===//

#ifndef RELALG_TYPES
#define RELALG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "RelAlgDialect.td"

//===----------------------------------------------------------------------===//
// RelAlg type definitions
//===----------------------------------------------------------------------===//

class RelAlg_Type<string name, string typeMnemonic>
    : TypeDef<RelAlg_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def RelAlg_TupleStream : RelAlg_Type<"TupleStream", "tuplestream"> {
  let summary = "A tuple stream";

  let description = [{
    A tuple stream is an abstraction that represents the dependency chain of
    relational operators.
  }];
}

#endif // RELALG_TYPES