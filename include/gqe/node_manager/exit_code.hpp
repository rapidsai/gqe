/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>

namespace gqe::node_manager {

/**
 * @brief Exit statuses of the node manager process.
 *
 * Operators and supervising harnesses act on these values, so each one is fixed: never renumber an
 * enumerator, and give a new one an unused value. The underlying type keeps every value within the
 * 0-255 range that a process can report.
 *
 * The enumeration is unscoped, so a value converts to the `int` that `exit`, `_exit`, and `main`
 * take without a cast. It is nested in this namespace so that its enumerators are still named as
 * `exit_code::success` and do not spill into `gqe::node_manager`.
 */
namespace exit_code {

enum value : std::uint8_t {
  /// The node manager shut down cleanly, including after SIGINT or SIGTERM while serving.
  success = 0,
  /// Any failure not listed below, such as invalid arguments, a startup error, or a signal received
  /// before the server started.
  failure = 100,
  /// The shared memory became unrecoverable. The node manager must be restarted and every table
  /// reloaded.
  shared_memory_unrecoverable = 101,
  /// The task managers could not be brought back after a failure, so the node cannot serve. The
  /// node manager must be restarted and every table reloaded.
  task_managers_unavailable = 102,
};

namespace detail {

// Fails to compile if two enumerators share a value, because duplicate case labels are an error.
// Nothing calls it, because the check happens when the function is compiled.
// Every enumerator needs a case here. -Wswitch (an error under -Werror) reports a missing case only
// for a new value, not for one that duplicates an existing value, so check new cases in review.
// The switch is on a parameter because GCC does not warn when the condition is a constant.
constexpr bool values_are_distinct(value code)
{
  switch (code) {
    case success:
    case failure:
    case shared_memory_unrecoverable:
    case task_managers_unavailable: break;
  }
  return true;
}

}  // namespace detail

}  // namespace exit_code

}  // namespace gqe::node_manager
