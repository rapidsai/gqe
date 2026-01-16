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

#include <gqe/utility/linux.hpp>

#include <gqe/types.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

namespace gqe {
namespace {

/**
 * @brief Test that set_thread_affinity works.
 */
TEST(LinuxUtilityTest, SetThreadAffinity)
{
  // Use CPU 0, which should always exist
  cpu_set affinity(0);

  std::atomic<bool> success{false};
  std::thread worker([&]() {
    try {
      utility::set_thread_affinity(affinity);
      success = true;
    } catch (...) {
      success = false;
    }
  });
  worker.join();

  EXPECT_TRUE(success.load());
}

}  // namespace
}  // namespace gqe
