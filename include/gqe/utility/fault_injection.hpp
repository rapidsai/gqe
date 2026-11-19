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

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

/**
 * @file fault_injection.hpp
 * @brief Trigger files for test-only fault injection.
 *
 * # Design
 *
 * A fault injection hook is code that fails or stalls a process on demand, so that a test can reach
 * a failure that is otherwise hard to cause deterministically. Hooks are compiled only into builds
 * configured with the CMake option `GQE_ENABLE_FAULT_INJECTION`, and each hook's call site is
 * wrapped in `#ifdef GQE_ENABLE_FAULT_INJECTION`. This header does not depend on that macro, so
 * that a translation unit compiled with it and one compiled without it agree on the definitions.
 *
 * A hook is armed by a trigger file. The environment variable named by the hook, by convention
 * `GQE_TEST_<NAME>_FILE`, holds the path of the file. A hook fires only while the file exists and
 * has a line for the calling rank. The variable alone would not be enough: spawned task managers
 * inherit the node manager's environment, so a variable would also fail the initial startup. With a
 * file, a test starts the node manager normally, creates the file to arm the hook, and removes it
 * to disarm it again.
 *
 * # Trigger File Format
 *
 * Each line is `<rank> [<delay_seconds>]`. The rank is a non-negative integer, and the optional
 * delay is a non-negative number of whole seconds that the hook waits before it acts. Blank lines
 * and lines that do not parse are ignored, including a line with anything but whitespace after the
 * delay. If several lines name the same rank, the first one applies.
 *
 * ## Thread Safety
 *
 * Every function reads the environment and the file on each call and keeps no state, so concurrent
 * calls are safe as long as nothing modifies the environment concurrently.
 */

namespace gqe::utility::fault_injection {

/**
 * @brief The parameters of a trigger line that applies to a rank.
 */
struct trigger {
  /// Time the hook waits before it acts. Zero if the line gives no delay.
  std::chrono::seconds delay{0};
};

/**
 * @brief Look up the trigger line for @p rank in the file named by an environment variable.
 *
 * @param variable Name of the environment variable that holds the trigger file's path.
 * @param rank Rank of the calling process.
 * @return The first matching line's parameters, or `std::nullopt` if the variable is unset or
 * empty, the file cannot be opened, or no line names @p rank.
 */
[[nodiscard]] inline std::optional<trigger> find_trigger(char const* variable, int rank)
{
  auto const* path = std::getenv(variable);
  if (path == nullptr || *path == '\0') { return std::nullopt; }

  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    int listed_rank = 0;
    if (!(fields >> listed_rank) || listed_rank != rank) { continue; }
    long long delay_seconds = 0;
    fields >> std::ws;
    if (!fields.eof()) {
      if (!(fields >> delay_seconds) || delay_seconds < 0) { continue; }
      // Nothing but whitespace may follow the delay.
      fields >> std::ws;
      if (!fields.eof()) { continue; }
    }
    return trigger{std::chrono::seconds(delay_seconds)};
  }
  return std::nullopt;
}

}  // namespace gqe::utility::fault_injection
