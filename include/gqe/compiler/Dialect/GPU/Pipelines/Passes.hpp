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

#pragma once

#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassOptions.h>

#include <string>

namespace gqe {
namespace compiler {
namespace gpu {

/**
 * @brief Options for the custom GPU to NVVM pipeline for GQE.
 */
struct GPUToNVVMPipelineOptions : public mlir::PassPipelineOptions<GPUToNVVMPipelineOptions> {
  PassOptions::Option<std::string> targetArch{
    *this,
    "target-arch",
    llvm::cl::desc("The target architecture to serialize to cubin."),
    llvm::cl::init("sm_80")};
};

/**
 * @brief Build a custom pass pipeline for lowering the GPU dialect to NVVM for GQE.
 *
 * Rewriting is performed with a pass pipeline. The passes are a minimal subset of
 * `mlir::gpu::buildLowerToNVVMPassPipeline`.
 *
 * Specifically, only device code passes are required for GQE. This pipeline excludes host passes
 * and unneded passes such as kernel outlining.
 *
 * Customizing the passes allows a better understanding of the passes performed and adding dialect
 * passes not included in the default pipeline.
 *
 * TODO: Measure whether excluded the host passes reduces query compile time.
 *
 * References:
 * mlir/lib/Dialect/GPU/Pipelines/GPUToNVVMPipeline.cpp
 * mlir/test/lib/Dialect/LLVM/TestLowerToLLVM.cpp
 *
 * @param[in,out] pm The pipeline manager to which the passes are added.
 * @param[in] options The pipeline configuration options.
 */
void buildLowerToNvvmPassPipeline(mlir::OpPassManager& pm, const GPUToNVVMPipelineOptions& options);

/**
 * @brief Register the pipeline in the MLIR pass registry.
 */
void registerGqeGpuToNvvmPipeline();

}  // namespace gpu
}  // namespace compiler
}  // namespace gqe
