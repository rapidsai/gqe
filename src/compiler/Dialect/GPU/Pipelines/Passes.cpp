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

#include <gqe/compiler/Dialect/GPU/Pipelines/Passes.hpp>

#include <gqe/compiler/Types.hpp>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Pass/PassOptions.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include <climits>
#include <cstdint>

namespace gqe {
namespace compiler {
namespace gpu {

// Debugging hint: To debug passes and the pipeline order, insert
// `pm.addPass(mlir::createPrintIRPass())`.
void buildLowerToNvvmPassPipeline(mlir::OpPassManager& pm, const GPUToNVVMPipelineOptions& options)
{
  constexpr int32_t indexBitwidth = sizeof(mlirIndexType) * CHAR_BIT;

  mlir::ConvertIndexToLLVMPassOptions indexToLLVMOptions;
  indexToLLVMOptions.indexBitwidth = indexBitwidth;

  mlir::GpuNVVMAttachTargetOptions nvvmTargetOptions;
  // Compiler optimization level
  //
  // Corresponds to, e.g., `-O3`
  nvvmTargetOptions.optLevel = 3;
  // `chip` describes the target architecture.
  //
  // Example: `sm_90`
  //
  // See: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch
  nvvmTargetOptions.chip = options.targetArch;
  // `features` describes the PTX version 8.0 as `+ptx80`. The `+` symbol adds
  // this PTX attribute to the set, a `-` would remove the attribute from the
  // set.
  //
  // It corresponds to LLVM's llc `-mattr=+ptx80` flag.
  //
  // The PTX attribute is optional.
  //
  // See: https://llvm.org/docs/CommandGuide/llc.html
  // See: https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/NVPTX/NVPTX.td
  // nvvmTargetOptions.features = "+ptx80";

  mlir::ConvertGpuOpsToNVVMOpsOptions gpuToNvvmOptions;
  // A `MemRef` type used as kernel argument is converted to two pointer and
  // three dimension arguments. Setting the bare pointer calling convention
  // would result in a single pointer, but is not supported for
  // dynamically-sized memrefs.
  //
  // See: https://mlir.llvm.org/docs/TargetLLVMIR/#bare-pointer-calling-convention-for-ranked-memref
  gpuToNvvmOptions.useBarePtrCallConv = false;
  // Specify the index bit width
  //
  // Used by the MLIR Index dialect.
  //
  // `0` sets the default value, which is the machine's word length. Should be
  // 64-bit for NVIDIA GPUs (unverified).
  gpuToNvvmOptions.indexBitwidth = indexBitwidth;

  // From `buildCommonPassPipeline`.
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertNVVMToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());

  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
  pm.addPass(createConvertIndexToLLVMPass(indexToLLVMOptions));
  pm.addPass(mlir::createUBToLLVMConversionPass());

  // Strip debug info needs to come directly before `gpu-to-nvvm`, otherwise
  // module serialization fails.
  pm.addPass(mlir::createStripDebugInfoPass());

  // From `buildCommonPassPipeline`.
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createConvertGpuOpsToNVVMOps(gpuToNvvmOptions));

  // `arith-to-llvm` and `cf-to-llvm` need to run after `gpu-to-nvvm`, otherwise
  // there are unresolved casts. E.g.:
  // `builtin.unrealized_conversion_cast %block_id_x : index to i32`.
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  // `finalize-memref-to-llvm` needs to run after `gpu-to-nvvm`, otherwise GPU
  // memory spaces aren't convert to the LLVM integer memory space. The error
  // message is:
  //
  // `error: conversion of memref memory space #gpu.address_space<global> to
  // integer address space failed. Consider adding memory space conversions.`
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void registerGqeGpuToNvvmPipeline()
{
  mlir::PassPipelineRegistration<GPUToNVVMPipelineOptions>(
    "gqe-gpu-lower-to-nvvm-pipeline",
    "The pipeline performs the minimal amount of passes to lower device code to "
    "the main dialects to NVVM. The pipeline is adapted from "
    "`gpu-lower-to-nvvm-pipeline`. The main dialects are arith, gpu, index, "
    "memref, and scf.",
    buildLowerToNvvmPassPipeline);
}

}  // namespace gpu
}  // namespace compiler
}  // namespace gqe
