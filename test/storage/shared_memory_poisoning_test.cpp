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

// Tests for unrecoverable shared memory: a process that dies while holding a lock in the segment
// must make later operations fail with `gqe::shared_memory_unrecoverable_error` or set the sticky
// poisoned flag, never block or terminate the process.
//
// Each test poisons a lock by forking a child that takes it and then kills itself with SIGKILL.
// The child touches only the mutex, never CUDA or the logger.

#include <gqe/memory_resource/shared_memory_resource.hpp>
#include <gqe/memory_resource/shared_resource_adaptor.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/shared_memory.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <gtest/gtest.h>

#include <boost/container/vector.hpp>
#include <boost/interprocess/exceptions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using boost::interprocess::interprocess_mutex;

/// @brief Return the allocator lock of the segment's only arena.
interprocess_mutex& segment_allocator_mutex(gqe::shared_segment& segment)
{
  return segment.get_segment_manager()->arena_lock_for_testing(0);
}

/**
 * @brief Fork a child that locks @p mutex and dies with SIGKILL while holding it.
 *
 * The child runs after CUDA is initialized in the parent, so it must touch nothing but the mutex,
 * which lives in a `MAP_SHARED` mapping.
 */
void kill_child_holding(interprocess_mutex& mutex)
{
  pid_t const pid = fork();
  ASSERT_NE(pid, -1) << "fork failed";
  if (pid == 0) {
    try {
      mutex.lock();
    } catch (...) {
      _exit(1);
    }
    kill(getpid(), SIGKILL);
    _exit(2);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFSIGNALED(status)) << "child did not die from a signal, status " << status;
  ASSERT_EQ(WTERMSIG(status), SIGKILL);
}

class SharedMemoryPoisoningTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    // Unique per process, so concurrent runs from different build trees do not share a segment.
    segment_name = "gqe_shared_memory_poisoning_test_" + std::to_string(getpid());
    gqe::detail::reset_shared_memory_poisoned_for_testing();
    boost::interprocess::shared_memory_object::remove(segment_name.c_str());
    segment =
      std::make_unique<gqe::shared_segment>(gqe::create_only,
                                            segment_name,
                                            32 << 20,
                                            gqe::segment_layout_policy_type::single_arena_unplaced);
    resource = std::make_shared<gqe::memory_resource::shared_memory_resource>(segment.get());
  }

  void TearDown() override
  {
    // Teardown after poisoning must neither block nor throw. The resource unregisters the segment
    // from CUDA, and the segment is unmapped and unlinked, none of which takes a segment lock.
    gqe::detail::reset_shared_memory_poisoned_for_testing();
    gqe::detach_shared_memory_poison_flag(*segment);
    resource.reset();
    segment.reset();
    boost::interprocess::shared_memory_object::remove(segment_name.c_str());
  }

  std::string segment_name;
  std::unique_ptr<gqe::shared_segment> segment;
  std::shared_ptr<gqe::memory_resource::shared_memory_resource> resource;
};

TEST_F(SharedMemoryPoisoningTest, KilledHolderMakesMutexNotRecoverable)
{
  // Table storage mutexes have this type, and so do the arena allocator locks.
  auto* mutex = segment->construct<interprocess_mutex>(boost::interprocess::anonymous_instance)();
  ASSERT_NO_FATAL_FAILURE(kill_child_holding(*mutex));

  // The first lock sees the dead owner, the next ones the unrecoverable mutex. Both must throw.
  for (int attempt = 0; attempt < 2; ++attempt) {
    try {
      mutex->lock();
      FAIL() << "locking a mutex whose holder died must throw";
    } catch (boost::interprocess::lock_exception const& error) {
      EXPECT_EQ(error.get_error_code(), boost::interprocess::not_recoverable);
    }
  }
  EXPECT_FALSE(gqe::is_shared_memory_poisoned());
}

TEST_F(SharedMemoryPoisoningTest, LockHelperThrowsTypedErrorAndPoisons)
{
  auto* mutex = segment->construct<interprocess_mutex>(boost::interprocess::anonymous_instance)();
  ASSERT_NO_FATAL_FAILURE(kill_child_holding(*mutex));

  EXPECT_THROW(
    { [[maybe_unused]] auto guard = gqe::lock_shared_memory_mutex(*mutex, "test operation"); },
    gqe::shared_memory_unrecoverable_error);
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
  EXPECT_NE(gqe::shared_memory_poison_cause().find("test operation"), std::string::npos);
}

TEST_F(SharedMemoryPoisoningTest, AllocateOnPoisonedAllocatorThrowsTypedError)
{
  ASSERT_NO_FATAL_FAILURE(kill_child_holding(segment_allocator_mutex(*segment)));

  EXPECT_THROW(static_cast<void>(resource->allocate_sync(256)),
               gqe::shared_memory_unrecoverable_error);
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
}

TEST_F(SharedMemoryPoisoningTest, DeallocateOnPoisonedAllocatorPoisonsInsteadOfTerminating)
{
  void* block = resource->allocate_sync(256);
  ASSERT_NE(block, nullptr);
  ASSERT_NO_FATAL_FAILURE(kill_child_holding(segment_allocator_mutex(*segment)));

  // Deallocation is noexcept, so a throw here would terminate the test process.
  resource->deallocate_sync(block, 256);
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
  // The segment records the cause where the lock fails, before the resource catches the error.
  EXPECT_NE(gqe::shared_memory_poison_cause().find("release shared memory to its arena"),
            std::string::npos);
}

TEST_F(SharedMemoryPoisoningTest, CleanupOfDetachedRowGroupsPoisonsInsteadOfThrowing)
{
  auto shared_mr_ref = gqe::memory_resource::shared_resource_adaptor{resource};
  gqe::shared_memory_allocator alloc(segment->get_segment_manager());
  auto stream = cudf::get_default_stream();

  auto* table_storage = segment->construct<gqe::storage::shared_in_memory_table_storage>(
    boost::interprocess::anonymous_instance)(alloc);
  auto const table_handle = segment->get_handle_from_address(table_storage);
  std::vector<std::string> const col_names{"a"};
  std::vector<cudf::data_type> const col_types{cudf::data_type{cudf::type_id::INT32}};
  gqe::storage::shared_in_memory_table table(segment.get(), col_names, col_types, table_handle);

  cudf::test::fixed_width_column_wrapper<int32_t> a({1, 2, 3});
  std::vector<cudf::column_view> const input{a};
  table.replace_contents(cudf::table_view{input}, shared_mr_ref, stream);
  auto const revision_before_clear = table.storage_revision();

  // Only the allocator lock is poisoned, so clearing detaches the row group under the healthy table
  // mutex and then fails to free it. The cleanup is noexcept and must record the failure.
  ASSERT_NO_FATAL_FAILURE(kill_child_holding(segment_allocator_mutex(*segment)));
  table.clear_row_groups(stream, shared_mr_ref);

  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
  EXPECT_GT(table.storage_revision(), revision_before_clear);
  EXPECT_EQ(table.num_row_groups(), 0u);
}

TEST_F(SharedMemoryPoisoningTest, ContainerDestructorOnPoisonedAllocatorPoisonsInsteadOfTerminating)
{
  // Containers in the segment free their buffer from a noexcept destructor.
  using vector_type = boost::container::vector<int, gqe::shared_memory_container_allocator<int>>;
  auto* vector      = segment->construct<vector_type>(boost::interprocess::anonymous_instance)(
    gqe::shared_memory_allocator(segment->get_segment_manager()));
  vector->assign({1, 2, 3});

  ASSERT_NO_FATAL_FAILURE(kill_child_holding(segment_allocator_mutex(*segment)));
  EXPECT_THROW(vector->resize(1 << 20), gqe::shared_memory_unrecoverable_error);
  gqe::detail::reset_shared_memory_poisoned_for_testing();

  // The vector object itself is leaked: freeing it would need the poisoned lock.
  vector->~vector_type();
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
  EXPECT_NE(gqe::shared_memory_poison_cause().find("release shared memory to its arena"),
            std::string::npos);
}

TEST_F(SharedMemoryPoisoningTest, DestroyNamedObjectOnPoisonedAllocatorThrowsInsteadOfTerminating)
{
  // Erasing a name from the segment's name map frees the name from the string's noexcept
  // destructor. Names longer than the string's inline buffer are allocated from the arena. A second
  // name keeps the map from being emptied and destroyed.
  std::string const name(64, 'a');
  std::string const other_name(64, 'b');
  auto* object = segment->construct<int>(name.c_str())(1);
  ASSERT_NE(segment->construct<int>(other_name.c_str())(2), nullptr);

  ASSERT_NO_FATAL_FAILURE(kill_child_holding(segment_allocator_mutex(*segment)));
  EXPECT_THROW(segment->destroy_ptr(object), gqe::shared_memory_unrecoverable_error);
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
}

TEST_F(SharedMemoryPoisoningTest, PoisoningIsSharedThroughTheSegment)
{
  gqe::attach_shared_memory_poison_flag(*segment);
  auto* const flag = &segment->get_segment_manager()->poisoned_flag();
  EXPECT_EQ(flag->load(), 0u);

  // This process records the poisoning in the segment.
  gqe::mark_shared_memory_poisoned("this process");
  EXPECT_EQ(flag->load(), 1u);

  // Another process recorded the poisoning: only the segment's flag is set.
  gqe::detail::reset_shared_memory_poisoned_for_testing();
  EXPECT_FALSE(gqe::is_shared_memory_poisoned());
  flag->store(1);
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
  EXPECT_NE(gqe::shared_memory_poison_cause().find("another process"), std::string::npos);

  // After detaching, only this process's own flag counts.
  gqe::detach_shared_memory_poison_flag(*segment);
  EXPECT_FALSE(gqe::is_shared_memory_poisoned());
}

TEST_F(SharedMemoryPoisoningTest, AttachSharesEarlierPoisoning)
{
  gqe::mark_shared_memory_poisoned("before attach");
  gqe::attach_shared_memory_poison_flag(*segment);
  auto* const flag = &segment->get_segment_manager()->poisoned_flag();
  EXPECT_EQ(flag->load(), 1u);
}

TEST_F(SharedMemoryPoisoningTest, AttachTakesNoLockInTheSegment)
{
  // A task manager can open the segment after one of its locks became unrecoverable. Attaching
  // must still share later poisoning with the other processes.
  ASSERT_NO_FATAL_FAILURE(kill_child_holding(segment_allocator_mutex(*segment)));
  gqe::attach_shared_memory_poison_flag(*segment);
  gqe::mark_shared_memory_poisoned("after attach");
  EXPECT_EQ(segment->get_segment_manager()->poisoned_flag().load(), 1u);
}

TEST_F(SharedMemoryPoisoningTest, RecordIgnoresOtherErrors)
{
  EXPECT_FALSE(gqe::record_if_shared_memory_unrecoverable(
    std::make_exception_ptr(std::runtime_error("unrelated")), "test operation"));
  EXPECT_FALSE(gqe::record_if_shared_memory_unrecoverable(nullptr, "test operation"));
  EXPECT_FALSE(gqe::is_shared_memory_poisoned());

  EXPECT_TRUE(gqe::record_if_shared_memory_unrecoverable(
    std::make_exception_ptr(boost::interprocess::lock_exception()), "test operation"));
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
}

TEST_F(SharedMemoryPoisoningTest, FirstCauseIsKept)
{
  gqe::mark_shared_memory_poisoned("first cause");
  gqe::mark_shared_memory_poisoned("second cause");
  EXPECT_EQ(gqe::shared_memory_poison_cause(), "first cause");
}

}  // namespace
