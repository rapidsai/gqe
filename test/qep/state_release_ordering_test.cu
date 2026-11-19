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

#include "lifetime_probe.hpp"
#include "stream_gate.cuh"

#include <gqe/qep/state.hpp>
#include <gqe/utility/error.hpp>

#include <gtest/gtest.h>

#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <optional>
#include <span>

namespace {

namespace qep        = gqe::qep;
namespace state_kind = gqe::qep::state_kind;

/**
 * @brief Memory resource that poisons storage on its deallocation stream before releasing it.
 *
 * The poison makes premature deallocation observable in a pending device-to-device copy. The
 * caller must synchronize the deallocation stream before destroying the resource.
 */
class poisoning_memory_resource {
 public:
  poisoning_memory_resource()
    : _resource{[](std::size_t bytes, cuda::stream_ref stream, void* argument) {
                  auto& self = *static_cast<poisoning_memory_resource*>(argument);
                  return rmm::device_async_resource_ref{self._upstream}.allocate(
                    stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
                },
                [](void* pointer, std::size_t bytes, cuda::stream_ref stream, void* argument) {
                  auto& self = *static_cast<poisoning_memory_resource*>(argument);
                  self._deallocation_stream_status.store(cudaStreamQuery(stream.get()),
                                                         std::memory_order_relaxed);
                  GQE_CUDA_TRY_NO_THROW(cudaMemsetAsync(pointer, 0xA5, bytes, stream.get()));
                  rmm::device_async_resource_ref{self._upstream}.deallocate(
                    stream, pointer, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
                },
                this,
                this}
  {
  }

  [[nodiscard]] rmm::device_async_resource_ref get() { return _resource; }

  [[nodiscard]] cudaError_t deallocation_stream_status() const
  {
    return _deallocation_stream_status.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<cudaError_t> _deallocation_stream_status{cudaErrorUnknown};
  rmm::mr::cuda_async_memory_resource _upstream;
  rmm::mr::callback_memory_resource _resource;
};

[[nodiscard]] qep::shared_state make_int32_state(std::span<int32_t const> values,
                                                 gqe::query_stream const& stream_handle,
                                                 rmm::device_async_resource_ref mr)
{
  auto const stream = stream_handle.ref();
  auto column       = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  GQE_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<int32_t>(),
                               values.data(),
                               values.size_bytes(),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  stream.sync();
  return qep::make_shared_state(state_kind::cudf_column{std::move(column)}, stream_handle);
}

TEST(QepStateReleaseOrderingTest, PoisonsOnlyAfterThePendingReadWithoutBlockingTheHost)
{
  rmm::cuda_stream gate_stream;
  // These futures intentionally precede the gate. On assertion failure, reverse destruction opens
  // and synchronizes the gate before either future destructor can join a blocked asynchronous call.
  std::future<void> ordering_call;
  std::future<void> release_wait;
  gqe::query_context consuming_context{gqe::make_optimization_parameters(true)};
  auto const consuming_stream = consuming_context.create_stream("consumer");
  poisoning_memory_resource mr;
  gqe::test::stream_gate gate{gate_stream};
  std::array<int32_t, 2> const expected{10, 20};
  bool is_destroyed = false;
  qep::shared_state state;
  std::optional<gqe::query_stream> deallocation_stream;
  {
    gqe::query_context producing_context{gqe::make_optimization_parameters(true)};
    deallocation_stream.emplace(producing_context.create_stream("producer"));
    state = gqe::test::track_lifetime(make_int32_state(expected, *deallocation_stream, mr.get()),
                                      is_destroyed);
  }
  qep::state_container batch{state, state};
  rmm::device_buffer output(expected.size() * sizeof(int32_t), consuming_stream.ref());

  gate.wait(consuming_stream.ref());
  auto const input = *qep::to_column_view(state);
  GQE_CUDA_TRY(cudaMemcpyAsync(output.data(),
                               input.data<int32_t>(),
                               output.size(),
                               cudaMemcpyDeviceToDevice,
                               consuming_stream.ref().get()));
  ordering_call = std::async(
    std::launch::async, [&] { qep::order_deallocation_after_stream_use(batch, consuming_stream); });
  auto const ordering_status = ordering_call.wait_for(std::chrono::seconds{1});
  if (ordering_status != std::future_status::ready) {
    // Unblock the synchronization fallback before joining a broken implementation.
    gate.release();
  }
  ordering_call.get();
  ASSERT_EQ(ordering_status, std::future_status::ready);
  auto const consumer_status = cudaStreamQuery(consuming_stream.ref().get());

  state.reset();
  batch.clear();
  std::promise<void> release_started;
  auto release_started_future       = release_started.get_future();
  release_wait                      = std::async(std::launch::async, [&] {
    release_started.set_value();
    deallocation_stream->ref().sync();
  });
  auto const release_started_status = release_started_future.wait_for(std::chrono::seconds{1});
  if (release_started_status != std::future_status::ready) { gate.release(); }
  ASSERT_EQ(release_started_status, std::future_status::ready);
  auto const release_status = release_wait.wait_for(std::chrono::seconds{1});

  gate.release();
  release_wait.get();
  consuming_stream.ref().sync();
  gate_stream.synchronize();

  EXPECT_EQ(consumer_status, cudaErrorNotReady);
  EXPECT_EQ(release_status, std::future_status::timeout);
  EXPECT_TRUE(is_destroyed);

  std::array<int32_t, 2> actual{};
  GQE_CUDA_TRY(cudaMemcpy(actual.data(), output.data(), output.size(), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, expected);
}

TEST(QepStateReleaseOrderingTest, StateKeepsItsDeallocationStreamAlive)
{
  poisoning_memory_resource mr;
  std::array<int32_t, 2> const values{10, 20};
  qep::shared_state state;
  {
    gqe::query_context context{gqe::make_optimization_parameters(true)};
    auto const stream = context.create_stream("producer");
    state             = make_int32_state(values, stream, mr.get());
  }

  state.reset();

  EXPECT_EQ(mr.deallocation_stream_status(), cudaSuccess);
}

TEST(QepStateReleaseOrderingTest, CopyAssignmentKeepsTheOldDeallocationStreamAlive)
{
  poisoning_memory_resource mr;
  std::array<int32_t, 2> const values{10, 20};
  qep::shared_state state;
  {
    gqe::query_context context{gqe::make_optimization_parameters(true)};
    auto const stream = context.create_stream("producer");
    state             = make_int32_state(values, stream, mr.get());
  }

  qep::shared_state const empty;
  state = empty;

  EXPECT_EQ(state, nullptr);
  EXPECT_EQ(mr.deallocation_stream_status(), cudaSuccess);
}

TEST(QepStateReleaseOrderingTest, MoveAssignmentKeepsTheOldDeallocationStreamAlive)
{
  poisoning_memory_resource mr;
  std::array<int32_t, 2> const values{10, 20};
  qep::shared_state state;
  {
    gqe::query_context context{gqe::make_optimization_parameters(true)};
    auto const stream = context.create_stream("producer");
    state             = make_int32_state(values, stream, mr.get());
  }

  state = qep::shared_state{};

  EXPECT_EQ(state, nullptr);
  EXPECT_EQ(mr.deallocation_stream_status(), cudaSuccess);
}

}  // namespace
