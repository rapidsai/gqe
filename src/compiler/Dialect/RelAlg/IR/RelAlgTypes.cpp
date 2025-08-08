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

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgDialect.hpp>
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgTypes.hpp>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace gqe::compiler::relalg;

#define GET_TYPEDEF_CLASSES
#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgOpsTypes.cpp.inc>
