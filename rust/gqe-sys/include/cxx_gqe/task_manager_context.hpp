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
