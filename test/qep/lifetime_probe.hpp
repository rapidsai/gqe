/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/qep/state.hpp>

#include <memory>
#include <utility>

namespace gqe {
namespace test {

/** @brief Test-only task-private state that reports when its last owner is destroyed. */
class lifetime_probe final : public qep::task_private_state {
 public:
  explicit lifetime_probe(bool& is_destroyed) : _is_destroyed{is_destroyed} {}
  ~lifetime_probe() override { _is_destroyed = true; }

 private:
  bool& _is_destroyed;
};

/** @brief Attach a destruction probe without changing the visible state's identity or metadata. */
[[nodiscard]] inline qep::shared_state track_lifetime(qep::shared_state visible, bool& is_destroyed)
{
  auto probe = qep::make_shared_state(
    qep::state_kind::task_private{std::make_unique<lifetime_probe>(is_destroyed)});
  return qep::make_anchored_state(std::move(probe), std::move(visible));
}

}  // namespace test
}  // namespace gqe
