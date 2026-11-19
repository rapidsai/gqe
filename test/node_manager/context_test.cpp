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

#include <gqe/node_manager/context.hpp>

#include <gqe/executor/optimization_parameters.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace {

/// Removes the node manager's shared memory object on construction and on scope exit, even if an
/// assertion fails mid-test.
struct shm_name_guard {
  shm_name_guard() { boost::interprocess::shared_memory_object::remove(name); }

  ~shm_name_guard() { boost::interprocess::shared_memory_object::remove(name); }

  shm_name_guard(shm_name_guard const&)            = delete;
  shm_name_guard& operator=(shm_name_guard const&) = delete;

  static constexpr char const* name = gqe::node_manager::shared_memory_name;
};

/// Return default parameters whose shared memory segment is @p bytes.
gqe::optimization_parameters parameters_with_segment_bytes(std::size_t bytes)
{
  auto params                    = gqe::make_optimization_parameters(true);
  params.max_task_manager_memory = bytes;
  return params;
}

/// Return the message of the `std::runtime_error` constructing a context with @p params throws, or
/// an empty string if it throws none.
std::string construction_error(gqe::optimization_parameters const& params)
{
  try {
    gqe::node_manager::context const context(params);
  } catch (std::runtime_error const& error) {
    return error.what();
  }
  return {};
}

}  // namespace

TEST(NodeManagerContextTest, ExistingSegmentNamesTheStaleObject)
{
  shm_name_guard const guard;
  boost::interprocess::shared_memory_object const stale(
    boost::interprocess::create_only, shm_name_guard::name, boost::interprocess::read_write);

  EXPECT_THAT(construction_error(parameters_with_segment_bytes(std::size_t{8} << 20)),
              testing::HasSubstr("Is another node manager running?"));
}

TEST(NodeManagerContextTest, OtherFailuresOmitTheHint)
{
  shm_name_guard const guard;

  auto const message = construction_error(parameters_with_segment_bytes(1));
  EXPECT_THAT(message,
              testing::StartsWith("Failed to create shared memory segment 'gqe_shared_memory'"));
  EXPECT_THAT(message, testing::Not(testing::HasSubstr("Is another node manager running?")));
  EXPECT_FALSE(boost::interprocess::shared_memory_object::remove(shm_name_guard::name));
}
