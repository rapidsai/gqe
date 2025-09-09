/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <type_traits>

namespace gqe {

// Forward declarations
class task_manager_context;
class query_context;

/**
 * @brief context_reference is as a trivially-copyable non-owning reference to various contexts
 * that might be used during execution.
 *
 */
struct context_reference {
  task_manager_context* _task_manager_context;
  query_context* _query_context;
};

static_assert(std::is_trivially_copyable_v<context_reference>,
              "context_reference has to be trivially copyable");
}  // namespace gqe