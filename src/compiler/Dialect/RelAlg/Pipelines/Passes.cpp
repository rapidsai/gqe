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
