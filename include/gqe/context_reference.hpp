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
