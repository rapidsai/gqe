/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "gqe/executor_next/utilities/object_registry.hpp"

#include "../test_utilities.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <barrier>
#include <numeric>
#include <thread>
#include <vector>

using namespace gqe::executor_next;
using namespace gqe_test;

TEST(registered_object, get)
{
  int constexpr value = 5;
  int obj             = value;
  detail::registered_object registered_int(obj);
  ASSERT_EQ(registered_int.get(), value);

  int constexpr set_value = 7;
  registered_int.get()    = set_value;
  ASSERT_EQ(registered_int.get(), set_value);
}

TEST(registered_object_guard, get)
{
  int constexpr value = 5;
  int obj             = value;
  registered_object_guard guard(std::make_shared<detail::registered_object<int>>(obj));
  ASSERT_TRUE(guard);
  ASSERT_EQ(guard.get(), value);

  int constexpr set_value = 7;
  guard.get()             = set_value;
  ASSERT_EQ(guard.get(), set_value);
}

TEST(registered_object_guard, empty)
{
  registered_object_guard<int> guard(nullptr);
  ASSERT_FALSE(guard);

  EXPECT_THROW({ [[maybe_unused]] int value = guard.get(); }, std::runtime_error);
}

TEST(object_registry, register_and_find)
{
  int const key           = 3;
  std::string const value = "my string";

  object_registry<int, std::string> registry;

  registry.register_object(key, value);

  EXPECT_THROW({ registry.register_object(key, value); }, std::runtime_error);

  {
    auto guard = registry.find(key);
    ASSERT_TRUE(guard);
    ASSERT_EQ(guard.get(), value);
  }

  auto empty_guard = registry.find(-1);
  ASSERT_FALSE(empty_guard);

  registry.unregister_object(key);

  EXPECT_THROW({ registry.unregister_object(key); }, std::runtime_error);
}

TEST(object_registry, contains)
{
  int const key           = 3;
  std::string const value = "my string";

  object_registry<int, std::string> registry;
  registry.register_object(key, value);
  ASSERT_TRUE(registry.contains(key));
}

TEST(object_registry, register_and_find_move_only)
{
  int const key                           = 3;
  std::string const value                 = "my string";
  std::unique_ptr<std::string> string_ptr = std::make_unique<std::string>(value);

  object_registry<int, std::unique_ptr<std::string>> registry;

  registry.register_object(key, std::move(string_ptr));

  auto guard = registry.find(key);
  ASSERT_TRUE(guard);
  ASSERT_EQ(*(guard.get()), value);
}

void register_and_find_parallel_test(size_t nobjects, size_t iters)
{
  nobjects = std::max(nobjects, size_t(1));  // must be at least 1

  std::vector<int> keys(nobjects);
  std::iota(keys.begin(), keys.end(), 0);

  std::vector<std::string> values(nobjects);
  for (size_t i = 0; i < nobjects; ++i) {
    values[i] = std::string("string ") + std::to_string(i);
  }

  object_registry<int, std::string> registry;

  std::barrier start_barrier(nobjects);
  std::vector<std::jthread> threads;

  for (size_t i = 0; i < nobjects; ++i) {
    threads.emplace_back([&, idx = i]() {
      int const key          = keys[idx];
      std::string const& val = values[idx];

      start_barrier.arrive_and_wait();  // sync start.

      for (size_t iter = 0; iter < iters; ++iter) {
        spin_while([&]() { return registry.contains(key); });
        registry.register_object(key, val);
        {
          auto guard = registry.find(key);
          EXPECT_TRUE(guard);
          EXPECT_EQ(guard.get(), val);
        }
        registry.unregister_object(keys[idx]);
      }
    });
  }
}

TEST(object_registry, register_and_find_parallel) { register_and_find_parallel_test(2, 100); }

TEST(object_registry, register_and_find_parallel_stress)
{
  register_and_find_parallel_test(std::thread::hardware_concurrency(), 100);
}

void register_and_unregister_parallel_test(size_t nobjects, size_t iters)
{
  // Note: this test is also run locally with -fsanitize=thread

  nobjects = std::max(nobjects, size_t(1));  // must be at least 1

  std::vector<int> keys(nobjects);
  std::iota(keys.begin(), keys.end(), 0);

  std::vector<std::string> values(nobjects);
  for (size_t i = 0; i < nobjects; ++i) {
    values[i] = std::string("string ") + std::to_string(i);
  }

  object_registry<int, std::string> registry;

  std::barrier start_barrier(nobjects * 2);
  std::vector<std::jthread> register_threads;
  std::vector<std::jthread> unregister_threads;

  for (size_t i = 0; i < nobjects; ++i) {
    register_threads.emplace_back([&, idx = i]() {
      int const key          = keys[idx];
      std::string const& val = values[idx];

      start_barrier.arrive_and_wait();  // sync start.

      for (size_t iter = 0; iter < iters; ++iter) {
        spin_while([&]() { return registry.contains(key); });
        registry.register_object(key, val);
        {
          auto guard = registry.find(key);
          // NOTE: We are racing with unregister, so we first
          // check if the guard exists.
          if (guard) { EXPECT_EQ(guard.get(), val); }
        }
      }
    });
  }

  for (size_t i = 0; i < nobjects; ++i) {
    unregister_threads.emplace_back([&, idx = i]() {
      int const key = keys[idx];

      start_barrier.arrive_and_wait();  // sync start.

      for (size_t iter = 0; iter < iters; ++iter) {
        spin_while([&]() { return !registry.contains(key); });
        registry.unregister_object(key);
      }
    });
  }
}

TEST(object_registry, register_and_unregister_parallel)
{
  register_and_unregister_parallel_test(2, 100);
}

TEST(object_registry, register_and_unregister_parallel_stress)
{
  register_and_unregister_parallel_test(std::thread::hardware_concurrency() / 2, 100);
}
