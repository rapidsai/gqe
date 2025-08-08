//===- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
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