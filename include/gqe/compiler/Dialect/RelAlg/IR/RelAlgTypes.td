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
//===- RelAlgTypes.td - Relational Algebra dialect types -*- tablegen -*---===//

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
