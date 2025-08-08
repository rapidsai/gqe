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

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>

namespace gqe {
namespace compiler {
#define GEN_PASS_DECL_CONVERTRELALGTOSCFPASS
#include <gqe/compiler/Conversion/Passes.h.inc>

namespace relalg {
/**
 * @brief Collect a set of patterns to convert from the RelAlg dialect to SCF.
 */
void populateRelAlgToSCFConversionPatterns(mlir::RewritePatternSet& patterns);
}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
