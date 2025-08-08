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

#pragma once

#include <mlir/Pass/PassManager.h>

namespace gqe {
namespace compiler {
namespace relalg {

/**
 * @brief Build a pass pipeline for lowering the RelAlg dialect to SCF.
 *
 * @param pm A pass manager.
 */
void buildLowerToSCFPassPipeline(mlir::OpPassManager& pm);

/**
 * @brief Register the pipeline in the MLIR pass registry.
 */
void registerRelAlgToSCFPipeline();
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
