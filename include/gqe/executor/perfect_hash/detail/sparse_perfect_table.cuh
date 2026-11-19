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

#pragma once

// Sparse storage for a perfect hash table: one presence bit per slot, row indices packed densely,
// a clear bit a miss and the rank of a set bit its dense index.

#include <gqe/executor/perfect_hash/detail/query_common.cuh>
#include <gqe/executor/perfect_hash/detail/query_common.hpp>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/array>
#include <cuda/std/bit>

#include <cstddef>
#include <type_traits>

namespace gqe::perfect_hash {
namespace sparse_table {

constexpr int WORDS_PER_BLOCK = 3;
constexpr int SLOTS_PER_BLOCK = 64 * WORDS_PER_BLOCK;  // 192

constexpr int THREADS_PER_BLOCK = 128;

// Three words keep the block in one 32-byte sector; `count` trails them to share a lookup's chunk.
struct alignas(32) Block {
  cuda::std::array<uint64_t, WORDS_PER_BLOCK> words;
  uint64_t count;
};
static_assert(sizeof(Block) == 32, "a block has to be exactly one sector");
static_assert(std::is_standard_layout_v<Block>,
              "a block's member layout has to be fixed, because `rank` reads it as two "
              "`ulonglong2` chunks");
static_assert(offsetof(Block, count) == WORDS_PER_BLOCK * sizeof(uint64_t),
              "`rank` reads the block as 16-byte chunks and relies on `count` following the words");

// `count` needs 34 bits; the rest hold a prefix per word pair, sparing `rank` a popcount chain.
constexpr int COUNT_BITS   = 34;
constexpr int PREFIX_BITS  = 9;
constexpr int PREFIX_COUNT = (WORDS_PER_BLOCK - 1) / 2;

constexpr uint64_t COUNT_MASK  = (uint64_t(1) << COUNT_BITS) - 1;
constexpr uint64_t PREFIX_MASK = (uint64_t(1) << PREFIX_BITS) - 1;

static_assert(COUNT_BITS + PREFIX_COUNT * PREFIX_BITS <= 64,
              "the count and its packed prefixes have to share one 64-bit word");
static_assert((1 << PREFIX_BITS) > SLOTS_PER_BLOCK,
              "a prefix counts up to SLOTS_PER_BLOCK bits and has to be able to represent it");
static_assert(PREFIX_COUNT * 2 + 1 == WORDS_PER_BLOCK,
              "the packing assumes an odd WORDS_PER_BLOCK with one prefix per word pair");

// `count` takes the odd last word's place, so a lookup there gets word and count in one load.
constexpr int COUNT_CHUNK = WORDS_PER_BLOCK >> 1;
static_assert(COUNT_CHUNK * 2 + 1 == WORDS_PER_BLOCK,
              "`count` has to be the odd half of the block's last 16-byte chunk");

__host__ __device__ constexpr int64_t block_count(int64_t key_space)
{
  return (key_space + SLOTS_PER_BLOCK - 1) / SLOTS_PER_BLOCK;
}

// Bytes needed for the rank counters, bitmap, and dense values.
__host__ __device__ constexpr int64_t byte_count(int64_t key_space,
                                                 int64_t keys_numel,
                                                 int64_t bytes_per_value)
{
  return block_count(key_space) * static_cast<int64_t>(sizeof(Block)) +
         keys_numel * bytes_per_value;
}

// Whether `set_present` reads a word first, the read being there to make the write unnecessary.
enum class ReadBeforeWrite : bool { No = false, Yes = true };

// Rows per slot at which reading first pays, above the measured 5 to 10 to absorb estimate error.
constexpr int64_t GUARD_ROWS_PER_SLOT = 16;

// Mark the hash value as present. Discarding the result lets ptxas emit the SASS reduction RED,
// which the thread does not wait on, rather than ATOM, which has to return the old value.
template <ReadBeforeWrite read_first>
__device__ inline void set_present(Block* blocks, uint64_t hash_value)
{
  auto const block  = hash_value / SLOTS_PER_BLOCK;
  auto const offset = static_cast<unsigned>(hash_value % SLOTS_PER_BLOCK);
  auto* const word  = reinterpret_cast<unsigned long long*>(&blocks[block].words[offset >> 6]);
  auto const mask   = 1ull << (offset & 63);
  if constexpr (read_first == ReadBeforeWrite::Yes) {
    // Atomic because other threads reduce into this word concurrently, so a plain load races.
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> const present{*word};
    // Relaxed suffices: nothing is ordered on this, and a stale bit only costs a redundant RED.
    if (present.load(cuda::memory_order_relaxed) & mask) { return; }
  }
  // Not fetch_or, whose returned old value would force the ATOM.
  atomicOr(word, mask);
}

// Return the hash value's dense position, or -1 if absent.
__device__ inline int64_t rank(Block const* blocks, uint64_t hash_value)
{
  auto const block_index = hash_value / SLOTS_PER_BLOCK;
  auto const offset      = static_cast<unsigned>(hash_value % SLOTS_PER_BLOCK);
  auto const word_index  = offset >> 6;
  auto const bit         = offset & 63;

  // Two words per chunk, so the chunk holding words[w] is the same w >> 1 that selects its prefix.
  auto const* chunks      = reinterpret_cast<ulonglong2 const*>(blocks + block_index);
  auto const prefix_index = word_index >> 1;
  ulonglong2 const own    = chunks[prefix_index];

  auto const word = (word_index & 1) ? own.y : own.x;
  if (!((word >> bit) & 1ull)) { return -1; }

  // The second and last load, skipped entirely when the word was in the count's own chunk.
  ulonglong2 const tail = prefix_index == COUNT_CHUNK ? own : chunks[COUNT_CHUNK];
  auto const packed     = tail.y;
  auto ret              = static_cast<int64_t>(packed & COUNT_MASK);

  // Selected, not branched on, so both ternaries become predicated moves and the warp converges.
  auto const shift = COUNT_BITS + (prefix_index ? prefix_index - 1 : 0) * PREFIX_BITS;
  ret += static_cast<int64_t>((packed >> shift) & (prefix_index ? PREFIX_MASK : 0ull));

  // An odd word has only its predecessor left, and the chunk above already brought that in.
  if (word_index & 1) { ret += cuda::std::popcount(own.x); }

  // Avoid an undefined full-width shift; bit zero contributes nothing.
  if (bit) { ret += cuda::std::popcount(word & ((1ull << bit) - 1)); }
  return ret;
}

// Count set bits per block and append zero for the prefix scan.
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__
  static void popcount_kernel(Block const* blocks, int64_t* counts, int64_t blocks_count)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i <= blocks_count;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (i == blocks_count) {
      counts[i] = 0;
      continue;
    }
    int64_t total = 0;
    for (int w = 0; w < WORDS_PER_BLOCK; w++) {
      total += cuda::std::popcount(blocks[i].words[w]);
    }
    counts[i] = total;
  }
}

// Write each prefix count to its block, with the in-block word prefixes packed above it.
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__
  static void scatter_counts_kernel(Block* blocks, int64_t const* prefix, int64_t blocks_count)
{
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < blocks_count;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    auto packed  = static_cast<uint64_t>(prefix[i]) & COUNT_MASK;
    int64_t seen = 0;
    for (int j = 0; j < PREFIX_COUNT; j++) {
      seen += cuda::std::popcount(blocks[i].words[2 * j]) +
              cuda::std::popcount(blocks[i].words[2 * j + 1]);
      packed |= static_cast<uint64_t>(seen) << (COUNT_BITS + j * PREFIX_BITS);
    }
    blocks[i].count = packed;
  }
}

// Build rank counters from the bitmap and return the number of present keys.
inline int64_t build_rank(CudaGpuArray<Block>& blocks, cuda::stream_ref stream)
{
  auto const blocks_count = static_cast<int64_t>(blocks.numel());
  if (blocks_count == 0) { return 0; }

  auto const scan_count = blocks_count + 1;
  auto counts           = CudaGpuArray<int64_t>(scan_count, stream);
  auto prefix           = CudaGpuArray<int64_t>(scan_count, stream);

  auto const grid =
    static_cast<unsigned>(div_round_up(scan_count, static_cast<int64_t>(THREADS_PER_BLOCK)));
  PUSH_RANGE("popcount the blocks", 1);
  popcount_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.get()>>>(
    blocks.get().data(), counts.get().data(), blocks_count);
  GQE_CUDA_TRY(cudaPeekAtLastError());
  POP_RANGE();

  PUSH_RANGE("scan the blocks", 1);
  // cub returns a rejected launch instead of signalling it on the stream, and `prefix` is not
  // zeroed, so an unchecked failure would hand back the pool's previous tenant as the key count.
  void* temp_storage_ptr    = nullptr;
  size_t temp_storage_bytes = 0;
  GQE_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp_storage_ptr,
                                             temp_storage_bytes,
                                             counts.get().data(),
                                             prefix.get().data(),
                                             scan_count,
                                             stream.get()));
  auto temp_storage = CudaGpuArray<uint8_t>(temp_storage_bytes, stream);

  GQE_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp_storage.get().data(),
                                             temp_storage_bytes,
                                             counts.get().data(),
                                             prefix.get().data(),
                                             scan_count,
                                             stream.get()));
  POP_RANGE();

  scatter_counts_kernel<<<grid, THREADS_PER_BLOCK, 0, stream.get()>>>(
    blocks.get().data(), prefix.get().data(), blocks_count);
  GQE_CUDA_TRY(cudaPeekAtLastError());

  int64_t present = 0;
  GQE_CUDA_TRY(cudaMemcpyAsync(&present,
                               prefix.get().data() + blocks_count,
                               sizeof(present),
                               cudaMemcpyDefault,
                               stream.get()));
  stream.sync();

  // The caller's guard compares against the dense array, a tie here, so it sees only overshoots.
  auto const slots = blocks_count * static_cast<int64_t>(SLOTS_PER_BLOCK);
  if (present < 0 || present > slots) {
    std::stringstream what;
    what << "Error: sparse rank counted " << present << " present keys, outside the " << slots
         << " slots the bitmap holds.";
    throw std::runtime_error(annotate_line(what.str()));
  }
  return present;
}

}  // namespace sparse_table
}  // namespace gqe::perfect_hash
