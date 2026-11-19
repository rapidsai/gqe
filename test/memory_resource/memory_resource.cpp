/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/numa_pool_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/shared_memory_resource.hpp>
#include <gqe/memory_resource/shared_numa_pool_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/storage/segment_layout.hpp>
#include <gqe/types.hpp>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <boost/interprocess/shared_memory_object.hpp>
#include <cuda/stream>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <unistd.h>  // sysconf

#include <concepts>
#include <memory>

static_assert(!std::default_initializable<gqe::memory_resource::numa_memory_resource>);
static_assert(!std::default_initializable<gqe::memory_resource::numa_pool_memory_resource>);
static_assert(!std::default_initializable<gqe::memory_resource::shared_numa_pool_resource>);

namespace {

template <typename T>
[[nodiscard]] std::unique_ptr<T> make_memory_resource()
{
  return std::make_unique<T>();
}

template <>
[[nodiscard]] std::unique_ptr<gqe::memory_resource::numa_memory_resource> make_memory_resource()
{
  auto const kind = gqe::memory_kind::numa{};
  return std::make_unique<gqe::memory_resource::numa_memory_resource>(kind.numa_node_set,
                                                                      kind.page_kind);
}

// TODO: Remove this compile-time guard once POSIX file-handle support is added.
#if defined(GQE_ENABLE_FABRIC_SHARED_POOL_TESTS)
template <>
[[nodiscard]] std::unique_ptr<gqe::memory_resource::numa_pool_memory_resource>
make_memory_resource()
{
  auto const kind = gqe::memory_kind::numa_pool{};
  return std::make_unique<gqe::memory_resource::numa_pool_memory_resource>(kind.numa_node_id);
}

template <>
[[nodiscard]] std::unique_ptr<gqe::memory_resource::shared_numa_pool_resource>
make_memory_resource()
{
  auto const kind = gqe::memory_kind::shared_numa_pool{};
  return std::make_unique<gqe::memory_resource::shared_numa_pool_resource>(kind.numa_node_id);
}
#endif

}  // namespace

template <typename T>
class MemoryResourceTest : public testing::Test {
 public:
  MemoryResourceTest() : mr(make_memory_resource<T>()) {};

  std::unique_ptr<T> mr;
};

// Specialization for shared memory: create and clean up the shared segment
template <>
class MemoryResourceTest<gqe::memory_resource::shared_memory_resource> : public testing::Test {
 public:
  MemoryResourceTest()
  {
    boost::interprocess::shared_memory_object::remove("gqe_shared_memory");
    {
      gqe::shared_segment const created_segment(
        gqe::create_only,
        "gqe_shared_memory",
        1024 * 1024,  // 1MB
        gqe::segment_layout_policy_type::single_arena_unplaced);
    }
    segment = std::make_unique<gqe::shared_segment>(gqe::open_only, "gqe_shared_memory");
    mr      = std::make_unique<gqe::memory_resource::shared_memory_resource>(segment.get());
  }

  ~MemoryResourceTest() { boost::interprocess::shared_memory_object::remove("gqe_shared_memory"); }

  std::unique_ptr<gqe::shared_segment> segment;
  std::unique_ptr<gqe::memory_resource::shared_memory_resource> mr;
};

using MemoryResourceTypes = ::testing::Types<gqe::memory_resource::numa_memory_resource,
                                             gqe::memory_resource::system_memory_resource,
                                             gqe::memory_resource::pinned_memory_resource,
                                             gqe::memory_resource::shared_memory_resource
// TODO: Remove this compile-time guard once POSIX file-handle support is added.
#if defined(GQE_ENABLE_FABRIC_SHARED_POOL_TESTS)
                                             ,
                                             gqe::memory_resource::numa_pool_memory_resource,
                                             gqe::memory_resource::shared_numa_pool_resource
#endif
                                             >;
TYPED_TEST_SUITE(MemoryResourceTest, MemoryResourceTypes);

TYPED_TEST(MemoryResourceTest, AllocateZero)
{
  constexpr auto allocation_size = 0;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(cudf::get_default_stream(), allocation_size));
  EXPECT_NO_THROW(this->mr->deallocate(cudf::get_default_stream(), ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateKB)
{
  constexpr auto allocation_size = 1024;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(cudf::get_default_stream(), allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(cudf::get_default_stream(), ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateNonaligned)
{
  constexpr auto allocation_size = 13669;  // this is a prime number

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(cudf::get_default_stream(), allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(cudf::get_default_stream(), ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateLargerThanPagesize)
{
  const auto allocation_size = ::sysconf(_SC_PAGESIZE) + 3;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(cudf::get_default_stream(), allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(cudf::get_default_stream(), ptr, allocation_size));
}

TYPED_TEST(MemoryResourceTest, AllocateMultipleOfPageize)
{
  const auto allocation_size = ::sysconf(_SC_PAGESIZE) * 2;

  void* ptr = nullptr;
  EXPECT_NO_THROW(ptr = this->mr->allocate(cudf::get_default_stream(), allocation_size));
  ASSERT_NE(nullptr, ptr);
  EXPECT_TRUE(rmm::is_pointer_aligned(ptr));
  EXPECT_NO_THROW(this->mr->deallocate(cudf::get_default_stream(), ptr, allocation_size));
}

#if defined(GQE_ENABLE_FABRIC_SHARED_POOL_TESTS)
TEST(SharedNumaPoolResourceTest, ExportedPool)
{
  auto mr     = make_memory_resource<gqe::memory_resource::shared_numa_pool_resource>();
  std::ignore = mr->export_pool();

  void* ptr = mr->allocate(cudf::get_default_stream(), 1024);
  ASSERT_NE(nullptr, ptr);

  EXPECT_NO_THROW(mr->deallocate(cudf::get_default_stream(), ptr, 1024));
  EXPECT_NO_THROW(mr->finalize());
}

TEST(SharedNumaPoolResourceTest, ImportedPool)
{
  auto owner = make_memory_resource<gqe::memory_resource::shared_numa_pool_resource>();

  auto handle = owner->export_pool();

  void* owner_ptr = owner->allocate(cudf::get_default_stream(), 1024);
  ASSERT_NE(nullptr, owner_ptr);

  auto export_data = owner->export_pointer(owner_ptr);

  auto imported     = gqe::memory_resource::shared_numa_pool_resource::import_pool(handle);
  auto imported_ptr = imported.import_pointer(export_data);
  ASSERT_NE(nullptr, imported_ptr.get());
  imported_ptr.reset();

  EXPECT_NO_THROW(owner->deallocate(cudf::get_default_stream(), owner_ptr, 1024));
  EXPECT_NO_THROW(owner->finalize());
}
#else
// TODO: Remove this compile-time guard once POSIX file-handle support is added.
#endif
// The plan executor reports a query's peak from a counter frame pushed when the query starts. This
// pins the RMM semantics it relies on: the frame's peak excludes earlier allocations, and copies of
// the adaptor, including a type-erased one, share the counters.
TEST(QueryMemoryStatisticsTest, PushedFramePeakExcludesEarlierAllocations)
{
  using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
  rmm::mr::statistics_resource_adaptor statistics{
    any_device_resource{rmm::mr::cuda_memory_resource{}}};
  any_device_resource mr{statistics};
  auto const stream        = cudf::get_default_stream();
  constexpr auto alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

  void* retained = mr.allocate(stream, 1000, alignment);
  statistics.push_counters();
  EXPECT_EQ(statistics.get_bytes_counter().peak, 0);

  void* first  = mr.allocate(stream, 4096, alignment);
  void* second = mr.allocate(stream, 512, alignment);
  mr.deallocate(stream, first, 4096, alignment);
  void* third = mr.allocate(stream, 256, alignment);
  EXPECT_EQ(statistics.get_bytes_counter().peak, 4096 + 512);

  // Freeing an allocation that predates the frame does not lower the frame's peak.
  mr.deallocate(stream, retained, 1000, alignment);
  mr.deallocate(stream, second, 512, alignment);
  mr.deallocate(stream, third, 256, alignment);
  EXPECT_EQ(statistics.get_bytes_counter().peak, 4096 + 512);

  // Popping folds the frame into the outer counters, which saw 1000 bytes before it.
  statistics.pop_counters();
  EXPECT_EQ(statistics.get_bytes_counter().value, 0);
  EXPECT_EQ(statistics.get_bytes_counter().peak, 1000 + 4096 + 512);
}
