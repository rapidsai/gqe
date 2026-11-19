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
//===- RelAlgDialect.td - Relational Algebra dialect ----*- tablegen -*----===//

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
