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

#include <gqe/task_manager_context.hpp>

#include <utility>

namespace cxx_gqe {
/*
 * @brief Task manager context wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * task manager context.
 */
class task_manager_context {
 public:
  explicit task_manager_context(gqe::task_manager_context&& context) : _context(std::move(context))
  {
  }

  task_manager_context()                                       = delete;
  task_manager_context(task_manager_context const&)            = delete;
  task_manager_context& operator=(task_manager_context const&) = delete;

  /*
   * @brief Returns the C++ task manager context.
   *
   * This is a helper method used in the C++ bindings to convert from the wrapper to the actual
   * object.
   */
  inline gqe::task_manager_context& get() { return _context; }

 private:
  gqe::task_manager_context _context;
};

/*
 * @brief Returns a new task manager context wrapper.
 */
std::unique_ptr<task_manager_context> new_task_manager_context();
}  // namespace cxx_gqe
