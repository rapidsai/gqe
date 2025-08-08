# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_mlir_dialect_library(GQECompilerGPUPipelines
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/gqe/compiler/Dialect/GPU

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransforms
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRGPUDialect
  MLIRGPUToNVVMTransforms
  MLIRIndexToLLVM
  MLIRMemRefTransforms
  MLIRMemRefToLLVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRUBToLLVM
)
