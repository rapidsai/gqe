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
//===- Passes.td - Conversion pass definition file --------*- tablegen -*-===//

#ifndef GQE_CONVERSION_PASSES
#define GQE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertRelAlgToSCFPass : Pass<"convert-relalg-to-scf"> {
  let summary = "Convert RelAlg dialect to structured control flow dialect.";

  let description = [{
    The conversion pass documentation can be found under "./docs/compiler/RelAlgToSCFDialectConversion.md".
  }];

  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::gpu::GPUDialect",
                           "mlir::index::IndexDialect",
                           "mlir::memref::MemRefDialect",
                           "::mlir::scf::SCFDialect", "mlir::ub::UBDialect"];
}

#endif // GQE_CONVERSION_PASSES
