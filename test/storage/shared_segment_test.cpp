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

#include <gqe/storage/shared_segment.hpp>

#include <gqe/storage/segment_layout.hpp>
#include <gqe/utility/linux.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/containers/allocation_type.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>

// Include before numaif.h so its macros do not replace the kernel's enum names.
#include <linux/mempolicy.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <new>
#include <ranges>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace gqe::detail {

// Test access is limited to initializing a mapping with an explicit layout.
class shared_segment_test_access {
 public:
  static shared_segment_manager* initialize(void* base,
                                            std::size_t bytes,
                                            std::size_t header,
                                            segment_layout const& layout,
                                            std::span<std::size_t const> offsets)
  {
    return shared_segment_manager::initialize(
      base,
      bytes,
      header,
      segment_layout_policy_type::per_node_arenas_bound_overflow,
      layout,
      offsets);
  }
};

}  // namespace gqe::detail

namespace {

constexpr std::size_t segment_bytes = std::size_t{8} << 20;

constexpr auto policy = gqe::segment_layout_policy_type::single_arena_unplaced;

using int_vector =
  boost::interprocess::vector<int,
                              boost::interprocess::allocator<int, gqe::shared_segment_manager>>;

// Removes a shared memory object name on construction and on scope exit, even if an assertion fails
// mid-test.
struct shm_name_guard {
  explicit shm_name_guard(char const* name_) : name(name_)
  {
    boost::interprocess::shared_memory_object::remove(name);
  }

  ~shm_name_guard() { boost::interprocess::shared_memory_object::remove(name); }

  shm_name_guard(shm_name_guard const&)            = delete;
  shm_name_guard& operator=(shm_name_guard const&) = delete;

  char const* name;
};

// Sets the process umask and restores the previous one on scope exit.
struct umask_guard {
  explicit umask_guard(::mode_t mask) : previous(::umask(mask)) {}

  ~umask_guard() { ::umask(previous); }

  umask_guard(umask_guard const&)            = delete;
  umask_guard& operator=(umask_guard const&) = delete;

  ::mode_t previous;
};

// Records how many times it, and the probes it was built from, were copied and moved.
struct move_probe {
  move_probe() = default;

  move_probe(move_probe const& other) : copies(other.copies + 1), moves(other.moves) {}

  move_probe(move_probe&& other) noexcept : copies(other.copies), moves(other.moves + 1)
  {
    other.is_moved_from = true;
  }

  move_probe& operator=(move_probe const&) = delete;
  move_probe& operator=(move_probe&&)      = delete;

  int copies         = 0;
  int moves          = 0;
  bool is_moved_from = false;
};

// Keeps the copy and move counts of the probes it was constructed from, and counts its
// destructions.
struct probe_holder {
  probe_holder(move_probe from_lvalue, move_probe from_rvalue)
    : lvalue_copies(from_lvalue.copies),
      lvalue_moves(from_lvalue.moves),
      rvalue_copies(from_rvalue.copies),
      rvalue_moves(from_rvalue.moves)
  {
  }

  ~probe_holder() { ++destructions; }

  inline static int destructions = 0;

  int lvalue_copies;
  int lvalue_moves;
  int rvalue_copies;
  int rvalue_moves;
};

// Counts its constructions.
struct counted {
  counted() { ++constructions; }

  inline static int constructions = 0;
};

// Throws from its constructor.
struct throwing_constructor {
  throwing_constructor() { throw std::runtime_error("throwing_constructor failed"); }
};

// Constructs a second named object from its own constructor.
struct nesting_constructor {
  explicit nesting_constructor(gqe::shared_segment_manager* manager)
    : inner(manager->construct<int>("inner")(7))
  {
  }

  int* inner;
};

bool is_in_segment(gqe::shared_segment const& segment, void const* ptr)
{
  auto const start = reinterpret_cast<std::uintptr_t>(segment.get_address());
  auto const end   = start + segment.get_size();
  auto const value = reinterpret_cast<std::uintptr_t>(ptr);
  return value >= start && value < end;
}

// Create a shared memory object of @p bytes that holds no segment.
boost::interprocess::shared_memory_object make_raw_object(char const* name, std::size_t bytes)
{
  boost::interprocess::shared_memory_object object(
    boost::interprocess::create_only, name, boost::interprocess::read_write);
  object.truncate(static_cast<boost::interprocess::offset_t>(bytes));
  return object;
}

// Return the code of the `std::system_error` @p action throws, or an empty code if it throws none.
template <class Action>
std::error_code system_error_of(Action action)
{
  try {
    action();
  } catch (std::system_error const& error) {
    return error.code();
  }
  return {};
}

// Return the message of the `std::runtime_error` @p action throws, or an empty string if it throws
// none.
template <class Action>
std::string runtime_error_of(Action action)
{
  try {
    action();
  } catch (std::runtime_error const& error) {
    return error.what();
  }
  return {};
}

}  // namespace

TEST(SharedSegmentTest, CreateReportsOperatingSystemErrors)
{
  constexpr char name[] = "gqe_shared_segment_test_os_errors";
  shm_name_guard const guard(name);

  EXPECT_EQ(system_error_of([&] {
              [[maybe_unused]] gqe::shared_segment const opened(
                gqe::open_only, name, gqe::shared_segment::registration_policy::none);
            }),
            std::errc::no_such_file_or_directory);

  gqe::shared_segment const segment(gqe::create_only, name, segment_bytes, policy);
  EXPECT_EQ(system_error_of([&] {
              [[maybe_unused]] gqe::shared_segment const duplicate(
                gqe::create_only, name, segment_bytes, policy);
            }),
            std::errc::file_exists);
}

TEST(SharedSegmentTest, FailedPlanLeavesNoObject)
{
  constexpr char name[] = "gqe_shared_segment_test_failed_plan";
  shm_name_guard const guard(name);

  EXPECT_THROW(gqe::shared_segment(gqe::create_only, name, 1, policy), std::runtime_error);
  EXPECT_NO_THROW(gqe::shared_segment(gqe::create_only, name, segment_bytes, policy));
}

TEST(SharedSegmentTest, OpenRejectsFailedCreation)
{
  constexpr char name[] = "gqe_shared_segment_test_failed_creation";
  shm_name_guard const guard(name);
  auto const object = make_raw_object(name, 1);

  EXPECT_THAT(runtime_error_of([&] {
                [[maybe_unused]] gqe::shared_segment const opened(
                  gqe::open_only, name, gqe::shared_segment::registration_policy::none);
              }),
              testing::HasSubstr("failed to be created"));
}

// The opener reports the same error whether it sees the one-byte object before or after mapping it,
// so the outcome does not depend on when the truncation lands.
TEST(SharedSegmentTest, OpenWaitingForCreatorSeesFailure)
{
  constexpr char name[] = "gqe_shared_segment_test_waiting_open";
  shm_name_guard const guard(name);
  auto object = make_raw_object(name, segment_bytes);

  std::string message;
  {
    std::jthread const opener([&] {
      message = runtime_error_of([&] {
        [[maybe_unused]] gqe::shared_segment const opened(
          gqe::open_only, name, gqe::shared_segment::registration_policy::none);
      });
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    object.truncate(1);
  }

  EXPECT_THAT(message, testing::HasSubstr("failed to be created"));
}

TEST(SharedSegmentTest, OpenRejectsResizedSegment)
{
  constexpr char name[] = "gqe_shared_segment_test_resized";
  shm_name_guard const guard(name);
  gqe::shared_segment const creator(gqe::create_only, name, segment_bytes, policy);

  boost::interprocess::shared_memory_object object(
    boost::interprocess::open_only, name, boost::interprocess::read_write);
  object.truncate(static_cast<boost::interprocess::offset_t>(2 * segment_bytes));

  EXPECT_THAT(runtime_error_of([&] {
                [[maybe_unused]] gqe::shared_segment const opened(
                  gqe::open_only, name, gqe::shared_segment::registration_policy::none);
              }),
              testing::HasSubstr("records a size of"));
}

TEST(SharedSegmentTest, CreateAppliesPermissionsDespiteUmask)
{
  constexpr char name[] = "gqe_shared_segment_test_permissions";
  shm_name_guard const guard(name);
  boost::interprocess::permissions permissions;
  permissions.set_permissions(0660);
  {
    umask_guard const mask(077);
    gqe::shared_segment const segment(gqe::create_only, name, segment_bytes, policy, permissions);
  }

  boost::interprocess::shared_memory_object object(
    boost::interprocess::open_only, name, boost::interprocess::read_only);
  struct stat status{};
  ASSERT_EQ(::fstat(object.get_mapping_handle().handle, &status), 0);
  EXPECT_EQ(status.st_mode & ::mode_t{0777}, ::mode_t{0660});
}

TEST(SharedSegmentTest, RefusesWhatNoArenaHolds)
{
  constexpr char name[] = "gqe_shared_segment_test_refusals";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);
  using oversized = std::array<std::byte, 2 * segment_bytes>;

  EXPECT_THROW(segment.allocate_aligned(segment.get_size(), 8), std::bad_alloc);
  EXPECT_EQ(segment.get_segment_manager()->allocate_aligned(segment.get_size(), 8, std::nothrow),
            nullptr);
  EXPECT_THROW(segment.construct<oversized>(boost::interprocess::anonymous_instance)(),
               std::bad_alloc);

  EXPECT_NO_THROW(segment.deallocate(nullptr));
  EXPECT_NO_THROW(segment.destroy_ptr<int>(nullptr));
  EXPECT_TRUE(segment.all_memory_deallocated());
}

TEST(SharedSegmentTest, HandlesResolveAcrossMappings)
{
  constexpr char name[] = "gqe_shared_segment_test_handles";
  shm_name_guard const guard(name);
  gqe::shared_segment creator(gqe::create_only, name, segment_bytes, policy);
  gqe::shared_segment opener(gqe::open_only, name, gqe::shared_segment::registration_policy::none);
  constexpr std::uint64_t written = 0x0123'4567'89AB'CDEFULL;

  ASSERT_NE(creator.get_address(), opener.get_address());
  EXPECT_EQ(creator.get_size(), segment_bytes);
  EXPECT_EQ(opener.get_size(), segment_bytes);

  auto* const value = static_cast<std::uint64_t*>(
    creator.allocate_aligned(sizeof(std::uint64_t), alignof(std::uint64_t)));
  *value            = written;
  auto const handle = creator.get_handle_from_address(value);

  EXPECT_GT(handle, 0);
  EXPECT_EQ(creator.get_address_from_handle(handle), value);
  EXPECT_EQ(*static_cast<std::uint64_t*>(opener.get_address_from_handle(handle)), written);

  creator.deallocate(value);
}

TEST(SharedSegmentTest, VectorGrowsThroughAllocationCommand)
{
  constexpr char name[] = "gqe_shared_segment_test_vector";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);
  constexpr int count = 100'000;

  auto* const values = segment.construct<int_vector>(boost::interprocess::anonymous_instance)();
  std::ranges::copy(std::views::iota(0, count), std::back_inserter(*values));

  ASSERT_EQ(values->size(), static_cast<std::size_t>(count));
  EXPECT_TRUE(std::ranges::equal(*values, std::views::iota(0, count)));

  segment.destroy_ptr(values);
  EXPECT_TRUE(segment.all_memory_deallocated());
}

TEST(SharedSegmentTest, AllocationCommandRefusesExpandOnly)
{
  constexpr char name[] = "gqe_shared_segment_test_expand_only";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);
  auto* const manager = segment.get_segment_manager();

  auto* const block = static_cast<int*>(segment.allocate_aligned(64 * sizeof(int), alignof(int)));
  auto const free_with_block = segment.get_free_memory();
  // One element more than the block and the free space after it, so the expansion is attempted and
  // cannot be served.
  auto const limit = (manager->size(block) + free_with_block) / sizeof(int) + 1;

  auto received = limit;
  int* reuse    = block;
  EXPECT_EQ(manager->allocation_command(
              boost::interprocess::expand_fwd | boost::interprocess::nothrow_allocation,
              limit,
              received,
              reuse),
            nullptr);
  EXPECT_EQ(segment.get_free_memory(), free_with_block);

  received = limit;
  reuse    = block;
  EXPECT_THROW(manager->allocation_command(boost::interprocess::expand_fwd, limit, received, reuse),
               std::bad_alloc);
  EXPECT_EQ(segment.get_free_memory(), free_with_block);

  segment.deallocate(block);
  EXPECT_TRUE(segment.all_memory_deallocated());
}

TEST(SharedSegmentTest, ConstructForwardsArgumentsOnce)
{
  constexpr char name[] = "gqe_shared_segment_test_forwarding";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);
  move_probe lvalue_source;
  move_probe rvalue_source;
  probe_holder::destructions = 0;

  auto* const holder = segment.construct<probe_holder>(boost::interprocess::anonymous_instance)(
    lvalue_source, std::move(rvalue_source));

  EXPECT_TRUE(is_in_segment(segment, holder));
  EXPECT_EQ(holder->lvalue_copies, 1);
  EXPECT_EQ(holder->lvalue_moves, 0);
  EXPECT_EQ(holder->rvalue_copies, 0);
  EXPECT_EQ(holder->rvalue_moves, 1);
  EXPECT_FALSE(lvalue_source.is_moved_from);

  segment.destroy_ptr(holder);
  EXPECT_EQ(probe_holder::destructions, 1);
  EXPECT_TRUE(segment.all_memory_deallocated());
}

TEST(SharedSegmentTest, NamedObjectsAreFoundAndForgotten)
{
  constexpr char name[] = "gqe_shared_segment_test_named";
  shm_name_guard const guard(name);
  gqe::shared_segment creator(gqe::create_only, name, segment_bytes, policy);
  gqe::shared_segment opener(gqe::open_only, name, gqe::shared_segment::registration_policy::none);

  auto* const answer = creator.construct<int>("answer")(42);

  auto const found = creator.find<int>("answer");
  EXPECT_EQ(found.first, answer);
  EXPECT_EQ(found.second, std::size_t{1});

  auto const found_by_opener = opener.find<int>("answer");
  ASSERT_NE(found_by_opener.first, nullptr);
  EXPECT_EQ(opener.get_handle_from_address(found_by_opener.first),
            creator.get_handle_from_address(answer));
  EXPECT_EQ(*found_by_opener.first, 42);

  creator.destroy_ptr(answer);
  EXPECT_EQ(creator.find<int>("answer").first, nullptr);
  EXPECT_EQ(opener.find<int>("answer").first, nullptr);
  EXPECT_TRUE(creator.all_memory_deallocated());
}

TEST(SharedSegmentTest, DuplicateNameIsRejectedBeforeConstruction)
{
  constexpr char name[] = "gqe_shared_segment_test_duplicate_name";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);
  counted::constructions = 0;

  auto* const first = segment.construct<counted>("counted")();
  EXPECT_THAT(runtime_error_of([&] { segment.construct<counted>("counted")(); }),
              testing::HasSubstr("already registered"));
  EXPECT_EQ(counted::constructions, 1);

  segment.destroy_ptr(first);
  EXPECT_TRUE(segment.all_memory_deallocated());
}

TEST(SharedSegmentTest, ThrowingNamedConstructorLeavesNoTrace)
{
  constexpr char name[] = "gqe_shared_segment_test_throwing_constructor";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);

  EXPECT_THAT(runtime_error_of([&] { segment.construct<throwing_constructor>("object")(); }),
              testing::HasSubstr("throwing_constructor failed"));
  EXPECT_EQ(segment.find<throwing_constructor>("object").first, nullptr);
  EXPECT_TRUE(segment.all_memory_deallocated());

  auto* const replacement = segment.construct<int>("object")(1);
  EXPECT_EQ(segment.find<int>("object").first, replacement);

  segment.destroy_ptr(replacement);
  EXPECT_TRUE(segment.all_memory_deallocated());
}

// A name lock that is not recursive deadlocks here.
TEST(SharedSegmentTest, NamedConstructorMayConstructNamedObject)
{
  constexpr char name[] = "gqe_shared_segment_test_nested_names";
  shm_name_guard const guard(name);
  gqe::shared_segment segment(gqe::create_only, name, segment_bytes, policy);

  auto* const outer =
    segment.construct<nesting_constructor>("outer")(segment.get_segment_manager());

  EXPECT_EQ(segment.find<nesting_constructor>("outer").first, outer);
  EXPECT_EQ(segment.find<int>("inner").first, outer->inner);

  segment.destroy_ptr(outer->inner);
  segment.destroy_ptr(outer);
  EXPECT_TRUE(segment.all_memory_deallocated());
}

namespace {

// Two real allocators share a mapping; one fallback row avoids any CPU/node lookup.
class MultiArenaTest : public testing::Test {
 protected:
  static constexpr std::size_t arena_bytes = std::size_t{2} << 20;
  shm_name_guard guard{"gqe_shared_segment_test_multi_arena"};
  boost::interprocess::shared_memory_object object;
  boost::interprocess::mapped_region region;
  gqe::shared_segment_manager* manager = nullptr;
  std::array<std::size_t, 2> arena_offsets{};

  void initialize(std::vector<int> order = {0, 1})
  {
    auto const page   = gqe::page_kind{gqe::page_kind::system_default}.size();
    auto const header = gqe::shared_segment_manager::header_bytes(page, 2, 1);
    gqe::segment_layout layout{
      .arenas                 = {{gqe::cpu_set{}, gqe::arena_placement::none, arena_bytes},
                                 {gqe::cpu_set{}, gqe::arena_placement::none, arena_bytes}},
      .fallback_order_by_node = {std::move(order)},
      .pages                  = gqe::page_kind::system_default};
    object        = make_raw_object(guard.name, header + 2 * arena_bytes);
    region        = boost::interprocess::mapped_region(object, boost::interprocess::read_write);
    arena_offsets = {header, header + arena_bytes};
    manager       = gqe::detail::shared_segment_test_access::initialize(
      region.get_address(), region.get_size(), header, layout, arena_offsets);
  }

  void expect_in_arena(void const* pointer, std::size_t arena) const
  {
    auto const begin =
      reinterpret_cast<std::uintptr_t>(region.get_address()) + arena_offsets[arena];
    auto const address = reinterpret_cast<std::uintptr_t>(pointer);
    EXPECT_GE(address, begin);
    EXPECT_LT(address, begin + arena_bytes);
  }

  void TearDown() override
  {
    if (manager != nullptr) { EXPECT_TRUE(manager->all_memory_deallocated()); }
  }
};

}  // namespace

TEST_F(MultiArenaTest, OverflowAndDeallocationFollowOwningArena)
{
  initialize();
  // Boost temporarily over-allocates for over-aligned blocks. Fill with ordinary alignment
  // and check 64-byte alignment on the smaller overflow block.
  auto* first  = manager->allocate_aligned(3 * arena_bytes / 4, 8);
  auto* second = manager->allocate_aligned(arena_bytes / 4, 64);
  expect_in_arena(first, 0);
  expect_in_arena(second, 1);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second) % 64, 0);
  manager->deallocate(first);
  auto* reused = manager->allocate_aligned(3 * arena_bytes / 4, 8);
  expect_in_arena(reused, 0);
  manager->deallocate(second);
  manager->deallocate(reused);
}

TEST_F(MultiArenaTest, FallbackOrderChoosesPreferredArena)
{
  initialize({1, 0});
  auto* first  = manager->allocate_aligned(3 * arena_bytes / 4, 8);
  auto* second = manager->allocate_aligned(arena_bytes / 4, 64);
  expect_in_arena(first, 1);
  expect_in_arena(second, 0);
  manager->deallocate(first);
  manager->deallocate(second);
}

TEST_F(MultiArenaTest, AggregateFreeSpaceCannotServeOversizedAllocation)
{
  initialize();
  auto const request = arena_bytes + 4096;
  ASSERT_GT(manager->get_free_memory(), request);
  EXPECT_EQ(manager->allocate_aligned(request, 8, std::nothrow), nullptr);
  EXPECT_THROW(manager->allocate_aligned(request, 8), std::bad_alloc);
}

TEST_F(MultiArenaTest, AllocationCommandCanMoveToAnotherArena)
{
  initialize();
  auto* original = static_cast<int*>(manager->allocate_aligned(64 * sizeof(int), alignof(int)));
  auto* barrier  = manager->allocate_aligned(3 * arena_bytes / 4, 8);
  expect_in_arena(original, 0);
  expect_in_arena(barrier, 0);
  auto const requested = arena_bytes / (2 * sizeof(int));
  auto granted         = requested;
  int* reuse           = original;
  auto* replacement    = manager->allocation_command(
    boost::interprocess::allocate_new | boost::interprocess::expand_fwd, requested, granted, reuse);
  ASSERT_NE(replacement, nullptr);
  expect_in_arena(replacement, 1);
  EXPECT_EQ(reuse, nullptr);
  EXPECT_GE(granted, requested);
  manager->deallocate(original);
  manager->deallocate(barrier);
  manager->deallocate(replacement);
}

TEST_F(MultiArenaTest, NamedObjectsAndHandlesResolveAcrossMappings)
{
  initialize();
  // Leave room for the name index, but not for the large named object in arena 0.
  auto* filler  = manager->allocate_aligned(3 * arena_bytes / 4, 8);
  using payload = std::array<int, arena_bytes / (2 * sizeof(int))>;
  auto* value   = manager->construct<payload>("payload")();
  (*value)[0]   = 42;
  expect_in_arena(value, 1);
  gqe::shared_segment other(
    gqe::open_only, guard.name, gqe::shared_segment::registration_policy::none);
  ASSERT_NE(other.get_address(), region.get_address());
  auto* attached   = other.get_segment_manager();
  auto const found = attached->find<payload>("payload");
  ASSERT_NE(found.first, nullptr);
  EXPECT_EQ((*found.first)[0], 42);
  auto const handle = manager->get_handle_from_address(value);
  EXPECT_EQ(attached->get_handle_from_address(found.first), handle);
  EXPECT_EQ(attached->get_address_from_handle(handle), found.first);
  attached->destroy_ptr(found.first);
  EXPECT_EQ(manager->find<payload>("payload").first, nullptr);
  manager->deallocate(filler);
}

namespace {

using numa_mask = std::unique_ptr<bitmask, decltype(&numa_bitmask_free)>;

// Probe permissions separately; a subsequent segment failure is a test failure, not a skip.
bool can_bind_memory(int node)
{
  auto const page = gqe::page_kind{gqe::page_kind::system_default}.size();
  numa_mask mask(numa_allocate_nodemask(), numa_bitmask_free);
  auto* address = ::mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (address == MAP_FAILED) { throw std::system_error(errno, std::generic_category()); }
  numa_bitmask_setbit(mask.get(), node);
  auto const result = ::mbind(address, page, MPOL_BIND, mask->maskp, mask->size + 1, 0);
  auto const error  = errno;
  ::munmap(address, page);
  if (result == 0) { return true; }
  if (error == EPERM || error == ENOSYS) { return false; }
  throw std::system_error(error, std::generic_category(), "NUMA binding probe");
}

std::vector<int> host_arena_nodes()
{
  std::vector<int> nodes;
  numa_mask cpus(numa_allocate_cpumask(), numa_bitmask_free);
  for (int node = 0; node <= numa_max_node(); ++node) {
    if (numa_node_to_cpus(node, cpus.get()) == 0 && numa_bitmask_weight(cpus.get()) > 0) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

}  // namespace

class SharedSegmentNumaTest : public testing::Test {
 protected:
  std::vector<int> nodes;

  void SetUp() override
  {
    if (numa_available() < 0) { GTEST_SKIP() << "NUMA policy is unavailable"; }
    nodes = host_arena_nodes();
    ASSERT_FALSE(nodes.empty());
    errno = 0;
    numa_mask allowed(numa_get_mems_allowed(), numa_bitmask_free);
    if (errno == EPERM || errno == ENOSYS) {
      GTEST_SKIP() << "Reading allowed NUMA memories is denied or unavailable";
    }
    ASSERT_EQ(errno, 0) << std::strerror(errno);
    ASSERT_NE(allowed, nullptr);
    numa_mask binding(numa_allocate_nodemask(), numa_bitmask_free);
    ASSERT_NE(binding, nullptr);
    int mode          = MPOL_DEFAULT;
    auto const result = ::get_mempolicy(&mode, binding->maskp, binding->size + 1, nullptr, 0);
    if (result != 0 && (errno == EPERM || errno == ENOSYS)) {
      GTEST_SKIP() << "Reading NUMA memory policy is denied or unavailable";
    }
    ASSERT_EQ(result, 0) << std::strerror(errno);
    auto const is_bound = (mode & ~MPOL_MODE_FLAGS) == MPOL_BIND;
    if (is_bound && (mode & MPOL_F_RELATIVE_NODES)) {
      GTEST_SKIP() << "Relative membind policy is unsupported";
    }
    std::erase_if(nodes, [&](int node) {
      return !numa_bitmask_isbitset(allowed.get(), node) ||
             (is_bound && !numa_bitmask_isbitset(binding.get(), node));
    });
    if (nodes.empty()) { GTEST_SKIP() << "No eligible CPU-bearing NUMA nodes"; }
    for (auto node : nodes) {
      if (!can_bind_memory(node)) { GTEST_SKIP() << "NUMA binding is denied or unavailable"; }
    }
  }
};

TEST_F(SharedSegmentNumaTest, BoundPolicyCreatesAndAttaches)
{
  constexpr char name[] = "gqe_shared_segment_test_bound";
  shm_name_guard guard(name);
  constexpr auto bound = gqe::segment_layout_policy_type::per_node_arenas_bound_overflow;
  gqe::shared_segment creator(gqe::create_only, name, segment_bytes, bound);
  gqe::shared_segment opener(gqe::open_only, name, gqe::shared_segment::registration_policy::none);
  auto const layout = gqe::plan_segment_layout(bound, segment_bytes);
  ASSERT_EQ(layout.arenas.size(), nodes.size());
  auto const page = gqe::page_kind{layout.pages}.size();
  auto offset     = gqe::shared_segment_manager::header_bytes(
    page, nodes.size(), layout.fallback_order_by_node.size());
  numa_mask binding(numa_allocate_nodemask(), numa_bitmask_free);
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    int mode      = -1;
    auto* address = static_cast<std::byte*>(creator.get_address()) + offset;
    ASSERT_EQ(::get_mempolicy(&mode, binding->maskp, binding->size, address, MPOL_F_ADDR), 0)
      << std::strerror(errno);
    EXPECT_EQ(mode, MPOL_BIND);
    EXPECT_EQ(numa_bitmask_weight(binding.get()), 1);
    EXPECT_TRUE(numa_bitmask_isbitset(binding.get(), nodes[i]));
    offset += layout.arenas[i].bytes;
  }
  auto* value      = creator.construct<int>("value")(42);
  auto const found = opener.find<int>("value");
  ASSERT_NE(found.first, nullptr);
  EXPECT_EQ(*found.first, 42);
  creator.destroy_ptr(value);
  EXPECT_TRUE(creator.all_memory_deallocated());
}

TEST_F(SharedSegmentNumaTest, PagesResideOnEachArenasBoundNode)
{
  if (nodes.size() < 2) {
    GTEST_SKIP() << "Cross-node placement requires two eligible CPU-bearing nodes";
  }
  constexpr char name[] = "gqe_shared_segment_test_bound_pages";
  shm_name_guard guard(name);
  constexpr auto bound = gqe::segment_layout_policy_type::per_node_arenas_bound_overflow;
  gqe::shared_segment creator(gqe::create_only, name, segment_bytes, bound);
  auto const layout = gqe::plan_segment_layout(bound, segment_bytes);
  ASSERT_EQ(layout.arenas.size(), nodes.size());
  auto const page    = gqe::page_kind{layout.pages}.size();
  std::size_t offset = gqe::shared_segment_manager::header_bytes(
    page, layout.arenas.size(), layout.fallback_order_by_node.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    // Query an already-populated page in each arena without allocating on a particular CPU.
    int actual    = -1;
    auto* address = static_cast<std::byte*>(creator.get_address()) + offset;
    ASSERT_EQ(::get_mempolicy(&actual, nullptr, 0, address, MPOL_F_ADDR | MPOL_F_NODE), 0)
      << std::strerror(errno);
    EXPECT_EQ(actual, nodes[i]);
    offset += layout.arenas[i].bytes;
  }
}

TEST_F(SharedSegmentNumaTest, MembindRestrictsArenas)
{
  if (nodes.size() < 2) {
    GTEST_SKIP() << "Membind exclusion requires two eligible CPU-bearing nodes";
  }
  // Memory policy is per-thread; exiting this thread restores the test process's state.
  std::jthread worker([&] {
    numa_mask binding(numa_allocate_nodemask(), numa_bitmask_free);
    ASSERT_NE(binding, nullptr);
    auto const selected = nodes.back();
    numa_bitmask_setbit(binding.get(), selected);
    auto const result = ::set_mempolicy(MPOL_BIND, binding->maskp, binding->size);
    if (result != 0 && (errno == EPERM || errno == ENOSYS)) {
      GTEST_SKIP() << "Setting NUMA memory policy is denied or unavailable";
    }
    ASSERT_EQ(result, 0) << std::strerror(errno);
    EXPECT_EQ(gqe::utility::cpu_bearing_numa_nodes(), gqe::cpu_set{selected});
    auto const layout = gqe::plan_segment_layout(
      gqe::segment_layout_policy_type::per_node_arenas_bound_overflow, segment_bytes);
    ASSERT_EQ(layout.arenas.size(), 1u);
    EXPECT_EQ(layout.arenas.front().nodes, gqe::cpu_set{selected});
  });
}

TEST_F(SharedSegmentNumaTest, CpuAffinityDoesNotRestrictMemoryNodes)
{
  if (nodes.size() < 2) {
    GTEST_SKIP() << "Affinity regression requires two eligible CPU-bearing nodes";
  }
  std::jthread worker([&] {
    numa_mask affinity(numa_allocate_cpumask(), numa_bitmask_free);
    ASSERT_NE(affinity, nullptr);
    ASSERT_EQ(numa_sched_getaffinity(0, affinity.get()), 0) << std::strerror(errno);
    int selected = -1;
    for (unsigned int cpu = 0; cpu < affinity->size; ++cpu) {
      if (numa_bitmask_isbitset(affinity.get(), cpu) &&
          std::ranges::find(nodes, numa_node_of_cpu(cpu)) != nodes.end()) {
        selected = cpu;
        break;
      }
    }
    if (selected < 0) { GTEST_SKIP() << "No permitted CPU on an eligible memory node"; }
    numa_bitmask_clearall(affinity.get());
    numa_bitmask_setbit(affinity.get(), selected);
    ASSERT_EQ(numa_sched_setaffinity(0, affinity.get()), 0) << std::strerror(errno);
    numa_mask actual(numa_allocate_cpumask(), numa_bitmask_free);
    ASSERT_NE(actual, nullptr);
    ASSERT_EQ(numa_sched_getaffinity(0, actual.get()), 0) << std::strerror(errno);
    ASSERT_TRUE(numa_bitmask_equal(affinity.get(), actual.get()));
    gqe::cpu_set expected;
    for (auto node : nodes) {
      expected.add(node);
    }
    EXPECT_EQ(gqe::utility::cpu_bearing_numa_nodes(), expected);
  });
}
