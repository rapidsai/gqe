/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/memory_resource/memory_utilities.hpp>
#include <gqe/query_context.hpp>
#include <gqe/task_manager_context.hpp>

#include <gtest/gtest.h>

namespace gqe {
namespace test {

/**
 * @brief Base test fixture class from which all libcudf tests should inherit.
 *
 * Example:
 * ```
 * class MyTestFixture : public gqe::test::BaseFixture {};
 * ```
 */
class BaseFixture : public ::testing::Test {
  std::unique_ptr<task_manager_context> _task_manager_ctx;
  std::unique_ptr<query_context> _query_ctx;

 public:
  /**
   * @brief Initialize contexts with custom optimization parameters.
   *
   * Call this in your test's constructor or SetUp() to customize parameters.
   * If not called, default parameters will be used on first access.
   */
  void initialize_contexts(optimization_parameters params)
  {
    _task_manager_ctx = std::make_unique<task_manager_context>(
      params, gqe::memory_resource::create_static_memory_pool());
    _query_ctx = std::make_unique<query_context>(params);
  }

  /**
   * @brief Get the task manager context.
   * If it is not initialized, it will be created with default optimization parameters.
   */
  task_manager_context* get_task_manager_ctx()
  {
    if (!_task_manager_ctx) { initialize_contexts(optimization_parameters(false)); }
    return _task_manager_ctx.get();
  }

  /**
   * @brief Get the query ctx object.
   * If it is not initialized, it will be created with default optimization parameters.
   */
  query_context* get_query_ctx()
  {
    if (!_query_ctx) { initialize_contexts(optimization_parameters(false)); }
    return _query_ctx.get();
  }
};

/**
 * @brief Base test fixture that takes a parameter.
 *
 * Example:
 * ```
 * class MyIntTestFixture : public gqe::test::BaseFixtureWithParam<int> {};
 * ```
 */
template <typename T>
class BaseFixtureWithParam : public ::testing::TestWithParam<T> {
  std::unique_ptr<task_manager_context> _task_manager_ctx;
  std::unique_ptr<query_context> _query_ctx;

 public:
  /**
   * @brief Initialize contexts with custom optimization parameters.
   *
   * Call this in your test's constructor or SetUp() to customize parameters.
   * If not called, default parameters will be used on first access.
   */
  void initialize_contexts(optimization_parameters params)
  {
    _task_manager_ctx = std::make_unique<task_manager_context>(
      params, gqe::memory_resource::create_static_memory_pool());
    _query_ctx = std::make_unique<query_context>(params);
  }

  /**
   * @brief Get the task manager context.
   * If it is not initialized, it will be created with default optimization parameters.
   */
  task_manager_context* get_task_manager_ctx()
  {
    if (!_task_manager_ctx) { initialize_contexts(optimization_parameters(false)); }
    return _task_manager_ctx.get();
  }

  /**
   * @brief Get the query ctx object.
   * If it is not initialized, it will be created with default optimization parameters.
   */
  query_context* get_query_ctx()
  {
    if (!_query_ctx) { initialize_contexts(optimization_parameters(false)); }
    return _query_ctx.get();
  }
};

}  // namespace test
}  // namespace gqe
