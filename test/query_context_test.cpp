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

#include <gqe/query_context.hpp>

#include <gqe/utility/error.hpp>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

void CUDART_CB mark_complete(void* state)
{
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  static_cast<std::atomic<bool>*>(state)->store(true, std::memory_order_release);
}

struct synchronization_gate {
  std::atomic<bool> entered{false};
  std::atomic<bool> is_released{false};
};

void CUDART_CB wait_for_release(void* state)
{
  auto* const gate = static_cast<synchronization_gate*>(state);
  gate->entered.store(true, std::memory_order_release);
  while (!gate->is_released.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

}  // namespace

static_assert(!std::default_initializable<gqe::query_stream>);
static_assert(!std::constructible_from<gqe::query_stream, cuda::stream_ref, std::uint64_t>);
static_assert(!std::constructible_from<gqe::query_stream, cuda::stream_ref>);

TEST(QueryStreamTest, SupportsRuntimeOwnedDefaultStream)
{
  auto const stream = gqe::query_stream::default_stream();

  EXPECT_EQ(stream.ref(), cudf::get_default_stream());
  EXPECT_EQ(stream.id(), std::nullopt);
}

TEST(QueryContextTest, CreatesDistinctQueryStreams)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  context.query_id = gqe::utility::uuid::generate();

  auto const first  = context.create_stream("pipeline=1/aggregate");
  auto const second = context.create_stream("pipeline=1/aggregate");

  EXPECT_EQ(first.id(), 0);
  EXPECT_EQ(second.id(), 1);
  EXPECT_NE(first.ref().get(), second.ref().get());
}

TEST(QueryContextTest, RejectsEmptyStreamName)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};

  EXPECT_THROW((void)context.create_stream(""), std::invalid_argument);
}

TEST(QueryContextTest, MovePreservesOwnedStreamsAndIds)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const first = context.create_stream("pipeline=1/read");

  auto moved        = std::move(context);
  auto const second = moved.create_stream("pipeline=2/write");

  EXPECT_EQ(first.id(), 0);
  EXPECT_EQ(second.id(), 1);
  EXPECT_NE(first.ref().get(), second.ref().get());
}

TEST(QueryContextTest, SynchronizesAllQueryOwnedStreams)
{
  // Declared before the context so callback storage outlives its streams on exception paths.
  std::atomic<bool> first_completed{false};
  std::atomic<bool> second_completed{false};
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const first  = context.create_stream("pipeline=1/read");
  auto const second = context.create_stream("pipeline=2/aggregate");

  GQE_CUDA_TRY(cudaLaunchHostFunc(first.ref().get(), mark_complete, &first_completed));
  GQE_CUDA_TRY(cudaLaunchHostFunc(second.ref().get(), mark_complete, &second_completed));

  context.close_and_synchronize_streams();

  EXPECT_TRUE(first_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(second_completed.load(std::memory_order_acquire));
}

TEST(QueryContextTest, SynchronizationPermanentlyClosesStreamCreation)
{
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  (void)context.create_stream("pipeline=1/read");

  context.close_and_synchronize_streams();

  EXPECT_THROW((void)context.create_stream("pipeline=2/write"), std::logic_error);
  EXPECT_NO_THROW(context.close_and_synchronize_streams());
}

TEST(QueryContextTest, ConcurrentStreamCreationFailsAfterSynchronizationBegins)
{
  // Declared before the context so callback storage outlives its stream on exception paths.
  synchronization_gate gate;
  gqe::query_context context{gqe::make_optimization_parameters(true)};
  auto const stream = context.create_stream("pipeline=1/read");
  GQE_CUDA_TRY(cudaLaunchHostFunc(stream.ref().get(), wait_for_release, &gate));

  std::atomic<bool> synchronization_failed{false};
  std::thread synchronizer{[&] {
    try {
      context.close_and_synchronize_streams();
    } catch (...) {
      synchronization_failed.store(true, std::memory_order_release);
    }
  }};

  while (!gate.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  std::atomic<bool> creation_completed{false};
  std::atomic<bool> creation_was_rejected{false};
  std::thread creator{[&] {
    // The callback can start before the synchronizer closes stream creation. Successful
    // creations are valid until closure, so retry until rejection or the gate is released.
    while (!gate.is_released.load(std::memory_order_acquire)) {
      try {
        (void)context.create_stream("pipeline=2/write");
      } catch (std::logic_error const&) {
        creation_was_rejected.store(true, std::memory_order_release);
        break;
      }
      std::this_thread::yield();
    }
    creation_completed.store(true, std::memory_order_release);
  }};

  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!creation_completed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  auto const creation_completed_while_synchronizing =
    creation_completed.load(std::memory_order_acquire);

  gate.is_released.store(true, std::memory_order_release);
  synchronizer.join();
  creator.join();

  EXPECT_FALSE(synchronization_failed.load(std::memory_order_acquire));
  EXPECT_TRUE(creation_completed_while_synchronizing);
  EXPECT_TRUE(creation_was_rejected.load(std::memory_order_acquire));
}
