/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cstdint>

namespace gqe {
namespace compiler {

/**
 * @brief Declare the type used as MLIR IndexType in GQE.
 *
 * References:
 * https://mlir.llvm.org/docs/Dialects/Builtin/#indextype
 * https://mlir.llvm.org/docs/Dialects/IndexOps
 */
using mlirIndexType = int32_t;

}  // namespace compiler
}  // namespace gqe
