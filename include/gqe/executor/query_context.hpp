/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/executor/optimization_parameters.hpp>

#include <cstdint>
#include <memory>

namespace gqe {

/**
 * @brief Query context of the current query.
 *
 * The query context centralizes all important resources and parameters that
 * are relevant for query execution. This enables optimizations that rely on
 * context, as well as logging and debugging.
 *
 * Implementation note: Add a comment for each struct member documenting its purpose.
 * Each member must have a default setting.
 */
struct query_context {
  query_context() = delete;

  explicit query_context(optimization_parameters const* parameters) : parameters(parameters) {}

  gqe::optimization_parameters const* parameters;
};

}  // namespace gqe
