/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/memory_resource/boost_shared_memory_resource.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <unistd.h>  // sysconf

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

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