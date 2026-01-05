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

#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <unistd.h>  // sysconf

template <typename T>
class MemoryResourceTest : public testing::Test {
 public:
  MemoryResourceTest() : mr(std::make_unique<T>()) {};

  std::unique_ptr<rmm::mr::device_memory_resource> mr;
};

// Specialization for boost shared memory: create and clean up the shared segment
template <>
class MemoryResourceTest<gqe::memory_resource::boost_shared_memory_resource>
  : public testing::Test {
 public:
  MemoryResourceTest()
  {
    auto segment = boost::interprocess::managed_shared_memory(
      boost::interprocess::create_only, "gqe_shared_memory", 1024 * 1024);  // 1MB
    mr = std::make_unique<gqe::memory_resource::boost_shared_memory_resource>();
  }

  ~MemoryResourceTest() { boost::interprocess::shared_memory_object::remove("gqe_shared_memory"); }

  std::unique_ptr<rmm::mr::device_memory_resource> mr;
};

using MemoryResourceTypes = ::testing::Types<gqe::memory_resource::numa_memory_resource,
                                             gqe::memory_resource::system_memory_resource,
                                             gqe::memory_resource::pinned_memory_resource,
                                             gqe::memory_resource::boost_shared_memory_resource>;
TYPED_TEST_SUITE(MemoryResourceTest, MemoryResourceTypes);

TYPED_TEST(MemoryResourceTest, AllocateZero)
{
  constexpr auto allocation_size = 0;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
  EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateKB)
{
  constexpr auto allocation_size = 1024;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateNonaligned)
{
  constexpr auto allocation_size = 13669;  // this is a prime number

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateLargerThanPagesize)
{
  const auto allocation_size = ::sysconf(_SC_PAGESIZE) + 3;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateMultipleOfPageize)
{
  const auto allocation_size = ::sysconf(_SC_PAGESIZE) * 2;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(ptr, allocation_size));
}
