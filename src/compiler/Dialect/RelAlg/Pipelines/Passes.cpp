/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/compiler/Dialect/RelAlg/Pipelines/Passes.hpp>

#include <gqe/compiler/Conversion/RelAlgToSCF/RelAlgToSCF.hpp>

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

namespace gqe {
namespace compiler {
namespace relalg {
void buildLowerToSCFPassPipeline(mlir::OpPassManager& pm)
{
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(gqe::compiler::createConvertRelAlgToSCFPass());
}

void registerRelAlgToSCFPipeline()
{
  mlir::PassPipelineRegistration<>("relalg-lower-to-scf-pipeline",
                                   "The pipeline lowers the RelAlg dialect to SCF.",
                                   buildLowerToSCFPassPipeline);
}
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
