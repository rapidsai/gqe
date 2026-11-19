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

// The public entry points return early on an empty input, so this drives the table directly, where
// a row count of zero reaches a launch as a zero-block grid unless `bulk_insert` returns first. The
// driver rejects that at the launch but reports it at the next CUDA call, which is cub inside
// `build_rank`, naming a device error in another file. Both InsertOutput instantiations launch.

#include <gqe/executor/perfect_hash/detail/xor_hash_table.cuh>

#include <gtest/gtest.h>

#include <cuda/stream>
#include <cudf/types.hpp>
#include <rmm/cuda_stream.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace {

using gqe::perfect_hash::ConstCudaGpuBufferPointer;
using gqe::perfect_hash::CudaGpuArray;
using gqe::perfect_hash::xor_hash_table::CheckEquality;
using gqe::perfect_hash::xor_hash_table::InsertOutput;

TEST(empty_input_test, inserting_no_keys_launches_nothing)
{
  rmm::cuda_stream owned;
  cuda::stream_ref const stream = owned;

  ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "an earlier test left an error behind";

  // One allocated element that nothing reads, so the pointers are valid at a row count of zero.
  auto storage = CudaGpuArray<int32_t>(size_t{1}, stream);
  auto const keys =
    std::vector<ConstCudaGpuBufferPointer>{{storage.get().data(), cudf::type_id::INT32}};
  auto table = gqe::perfect_hash::xor_hash_table::make_hash_table(keys, 0, std::nullopt, stream);

  table.bulk_insert<CheckEquality::True, InsertOutput::False>(keys, 0, std::nullopt, stream);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "after the insert that returns nothing";

  auto const indices =
    table.bulk_insert<CheckEquality::True, InsertOutput::True>(keys, 0, std::nullopt, stream);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess) << "after the insert that returns indices";
  EXPECT_EQ(std::get<0>(indices).numel(), 0u);
  EXPECT_EQ(std::get<1>(indices).numel(), 0u);

  EXPECT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
}

}  // namespace
