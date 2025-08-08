/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
