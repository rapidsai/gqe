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

// Getting a 16-byte chunk's half backwards returns another slot's dense index rather than an
// obviously broken number, so this is exhaustive over pseudorandom patterns.

#include <gqe/executor/perfect_hash/detail/sparse_perfect_table.cuh>

#include <cudf_test/base_fixture.hpp>
#include <gtest/gtest.h>

#include <cuda/stream>
#include <rmm/cuda_stream.hpp>

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using gqe::perfect_hash::CudaGpuArray;
using gqe::perfect_hash::sparse_table::Block;
using gqe::perfect_hash::sparse_table::block_count;
using gqe::perfect_hash::sparse_table::ReadBeforeWrite;
using gqe::perfect_hash::sparse_table::SLOTS_PER_BLOCK;
using gqe::perfect_hash::sparse_table::WORDS_PER_BLOCK;

constexpr int THREADS = 256;

template <ReadBeforeWrite read_first>
__global__ void mark_kernel(Block* blocks, uint64_t const* slots, int64_t count)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    gqe::perfect_hash::sparse_table::set_present<read_first>(blocks, slots[i]);
  }
}

__global__ void rank_kernel(Block const* blocks, int64_t key_space, int64_t* out)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < key_space;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    out[i] = gqe::perfect_hash::sparse_table::rank(blocks, static_cast<uint64_t>(i));
  }
}

int grid_for(int64_t n) { return static_cast<int>((n + THREADS - 1) / THREADS); }

struct rank_layout_test : public cudf::test::BaseFixture {};

// An oracle sharing no code with the implementation: the definition of the dense index.
void check_at_occupancy(int64_t key_space, double occupancy, uint64_t seed)
{
  auto const blocks_count = block_count(key_space);

  std::mt19937_64 rng(seed);
  std::bernoulli_distribution is_present(occupancy);
  std::vector<bool> present(static_cast<std::size_t>(key_space), false);
  std::vector<uint64_t> marked;
  for (int64_t slot = 0; slot < key_space; slot++) {
    if (is_present(rng)) {
      present[static_cast<std::size_t>(slot)] = true;
      marked.push_back(static_cast<uint64_t>(slot));
    }
  }
  ASSERT_FALSE(marked.empty()) << "the pattern has to set something for the test to mean anything";

  rmm::cuda_stream owned;
  cuda::stream_ref const stream = owned;

  auto ranks_for = [&](auto mark) {
    auto blocks = CudaGpuArray<Block>(static_cast<size_t>(blocks_count), stream);
    blocks.fill_byte<0>(stream);
    auto slots = CudaGpuArray<uint64_t>(marked, stream);
    mark<<<grid_for(static_cast<int64_t>(marked.size())), THREADS, 0, stream.get()>>>(
      blocks.get().data(), slots.get().data(), static_cast<int64_t>(marked.size()));

    auto const returned = gqe::perfect_hash::sparse_table::build_rank(blocks, stream);
    EXPECT_EQ(returned, static_cast<int64_t>(marked.size()))
      << "the scan total has to be the number of distinct slots marked";

    auto ranks = CudaGpuArray<int64_t>(static_cast<size_t>(key_space), stream);
    rank_kernel<<<grid_for(key_space), THREADS, 0, stream.get()>>>(
      blocks.get().data(), key_space, ranks.get().data());

    std::vector<int64_t> host(static_cast<std::size_t>(key_space));
    EXPECT_EQ(cudaMemcpyAsync(host.data(),
                              ranks.get().data(),
                              host.size() * sizeof(int64_t),
                              cudaMemcpyDeviceToHost,
                              stream.get()),
              cudaSuccess);
    stream.sync();
    return host;
  };

  auto const host = ranks_for(mark_kernel<ReadBeforeWrite::No>);
  // The guard only skips an atomic setting an already-set bit, so both must leave the same bitmap.
  EXPECT_EQ(ranks_for(mark_kernel<ReadBeforeWrite::Yes>), host)
    << "reading the word before the atomic changed what got marked";

  int64_t seen       = 0;
  int64_t mismatches = 0;
  int64_t first_bad  = -1;
  for (int64_t slot = 0; slot < key_space; slot++) {
    auto const want = present[static_cast<std::size_t>(slot)] ? seen++ : -1;
    if (host[static_cast<std::size_t>(slot)] != want) {
      if (first_bad < 0) { first_bad = slot; }
      mismatches++;
    }
  }

  ASSERT_EQ(mismatches, 0) << "at " << occupancy << " occupancy over " << blocks_count
                           << " blocks, first at slot " << first_bad << " (block "
                           << first_bad / SLOTS_PER_BLOCK << ", word "
                           << (first_bad % SLOTS_PER_BLOCK) / 64 << ", bit " << first_bad % 64
                           << "): got " << host[static_cast<std::size_t>(first_bad)] << ", want "
                           << (present[static_cast<std::size_t>(first_bad)] ? "its dense index"
                                                                            : "-1");
}

// Enough blocks that the cross-block prefix does work, at three occupancies.
TEST_F(rank_layout_test, every_slot_gets_its_dense_index_at_a_third_full)
{
  check_at_occupancy(1 << 20, 0.33, 0x5EEDu);
}

TEST_F(rank_layout_test, every_slot_gets_its_dense_index_when_nearly_empty)
{
  check_at_occupancy(1 << 20, 0.02, 0xC0FFEEu);
}

TEST_F(rank_layout_test, every_slot_gets_its_dense_index_when_nearly_full)
{
  check_at_occupancy(1 << 20, 0.95, 0xBEEFu);
}

// The two extremes the pseudorandom patterns reach only by chance.
TEST_F(rank_layout_test, every_slot_gets_its_dense_index_when_completely_full)
{
  check_at_occupancy(1 << 18, 1.0, 0x1u);
}

// Stated here so a reorder is caught even when the device tests are skipped for want of a GPU.
TEST_F(rank_layout_test, count_follows_the_words_so_the_chunks_match_the_prefix_pairing)
{
  EXPECT_EQ(offsetof(Block, count), WORDS_PER_BLOCK * sizeof(uint64_t));
  EXPECT_EQ(sizeof(Block), 32u);
  EXPECT_EQ(SLOTS_PER_BLOCK, 64 * WORDS_PER_BLOCK);
}

}  // namespace
