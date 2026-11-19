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

#pragma once

#include <gqe/utility/helpers.hpp>

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/open_addressing/functors.cuh>
#include <cuda/atomic>
#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace gqe::qep::mark_join_detail {
namespace cg = cooperative_groups;

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
__host__ __device__ constexpr T mark_mask() noexcept
{
  size_t shift = sizeof(T) * size_t{8} - size_t{1};
  auto mask    = T{1} << shift;
  return mask;
}

// Returns a hashed value with the mark set.
template <typename T>
__host__ __device__ constexpr T set_mark(T value) noexcept
{
  return value | mark_mask<T>();
}

// Checks if the mark bit is set in a hash.
template <typename T>
__host__ __device__ constexpr bool is_marked(T value) noexcept
{
  return value & mark_mask<T>();
}

// This operator represents a mark scan operation. It generally iterates a hash table for
// scanned entries.
template <typename StructureType,
          typename IteratorType,
          typename InputType,
          uint32_t bucket_size,
          bool is_anti_join>
class mark_join_scan_operator {
 public:
  // mark scan operator relies on cuco types, so will fail if empty_key_sentinel or
  // erased_key_sentinel undefined.
  mark_join_scan_operator(StructureType data, IteratorType iterator, cudf::size_type num_elements)
    : _iterator{iterator},
      _num_elements{num_elements},
      _is_filled{cuco::detail::open_addressing_ns::slot_is_filled<false, InputType>{
        data.empty_key_sentinel(), data.erased_key_sentinel()}}
  {
  }
  // Predicate is true only if the slot contains a valid entry and is either marked, or
  // if this is an anti-join, not marked.
  __device__ __forceinline__ bool predicate(const InputType& slot) const
  {
    return _is_filled(slot) && (is_marked(slot.first) ^ is_anti_join);
  }

  // When we accept an entry, we only write its row index to output and omit the hash.
  __device__ constexpr __forceinline__ auto accept(const InputType& slot) const
  {
    return static_cast<cudf::size_type>(slot.second);
  }

  // Underlying iterator must support bracket indexing.
  __device__ __forceinline__ auto get_bucket(const cudf::size_type index) const
  {
    return _iterator[index];
  }

  __device__ constexpr __forceinline__ cudf::size_type elements_per_bucket() const
  {
    return bucket_size;
  }

  __device__ constexpr __forceinline__ cudf::size_type num_buckets() const { return _num_elements; }

 private:
  const cudf::size_type _num_elements;
  const IteratorType _iterator;
  const cuco::detail::open_addressing_ns::slot_is_filled<false, InputType> _is_filled;
};

/**
 * @brief This kernel performs the mark probe functionality to evaluate if input
 * rows are present in the hash table, and mark them if they are included.
 * @param[in] build_conditional_table the build table with references to condition-evaluated columns
 * @param[in] probe_conditional_table the probe table with references to condition-evaluated columns
 * @param[in] expression_data the device pointer to the parsed expression data for AST evaluation
 * @param[in,out] mark_set_ref a reference to the hash table used in the probe
 * @param[in] join_predicate the operator that will perform equality condition evaluation
 * @param[in] probe_rows the rows in the probe table to evaluate against
 * @param[in] num_rows the number of rows in probe_rows
 * @param[out] global_mark_counter_ref a global counter for the number of marks set, modified
 * atomically
 * @tparam is_mixed determines if we handle mixed conditions (enables cuDF AST evaluation) in
 * addition to equality
 * @tparam has_nulls is passed through to cuDF AST evaluation
 * @tparam is_low_selectivity is a heuristic hint for compile-time optimization of some branches
 * @tparam has_probe_mask indicates if a probe mask is valud or not
 */
template <int32_t block_size,
          typename ComparatorAdapterType,
          typename MarkSetRefType,
          typename ProbeKeyType,
          uint32_t cg_size,
          bool has_nulls,
          bool is_mixed,
          bool is_low_selectivity,
          bool has_probe_mask>
__global__ __launch_bounds__(block_size) void mark_probe(
  cudf::table_device_view const build_conditional_table,
  cudf::table_device_view const probe_conditional_table,
  cudf::ast::detail::expression_device_view const expression_data,
  MarkSetRefType mark_set_ref,
  ComparatorAdapterType join_predicate,
  ProbeKeyType const* __restrict__ probe_rows,
  bool const* probe_mask,
  cudf::size_type num_rows,
  cudf::size_type* global_mark_counter_ref)
{
  constexpr uint32_t warp_size = 32;
  // Grid control groups
  const auto grid  = cooperative_groups::this_grid();
  const auto block = cg::this_thread_block();
  const auto tile  = cg::tiled_partition<warp_size>(block);
  // The bucket tile is a sub-warp tile for vectorized hash bucket probing.
  const auto bucket_tile = cg::tiled_partition<cg_size, cg::thread_block>(block);

  // set up hash table walk
  auto probing_scheme          = mark_set_ref.probing_scheme();
  auto bucket_extent           = mark_set_ref.bucket_extent();
  auto storage                 = mark_set_ref.storage_ref();
  auto empty_key_sentinel_key  = mark_set_ref.empty_key_sentinel();
  bool const allow_duplicates  = true;
  using key_type_for_predicate = typename MarkSetRefType::key_type;
  auto predicate =
    cuco::detail::equal_wrapper<key_type_for_predicate, ComparatorAdapterType, allow_duplicates>{
      mark_set_ref.empty_key_sentinel(), mark_set_ref.erased_key_sentinel(), join_predicate};

  // set up mixed expression evaluation (should be optimized out if not enabled by template)
  extern __shared__ std::byte predicate_scratch[];
  auto intermediate_scratch =
    reinterpret_cast<cudf::ast::detail::IntermediateDataType<has_nulls>*>(predicate_scratch);
  auto predicate_storage =
    &intermediate_scratch[block.thread_rank() * expression_data.num_intermediates];
  cudf::ast::detail::expression_evaluator<has_nulls> evaluator{
    build_conditional_table, probe_conditional_table, expression_data};

  // set up mark counting to advise mark join output buffer size
  cudf::size_type mark_counter = 0;
  // shared counter for block reduce
  __shared__ cuda::atomic<cudf::size_type, cuda::thread_scope_block> cta_mark_counter;
  // global counter as kernel output parameter
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_mark_counter{
    *global_mark_counter_ref};
  cg::invoke_one(cg::this_thread_block(),
                 [&]() { cta_mark_counter.store(0, cuda::memory_order_relaxed); });
  block.sync();
  // This kernel only requires warp synchronization. Inter-block communication is
  // performed by atomics. Therefore, closest-warp is sufficient.
  const auto loop_bound = gqe::utility::divide_round_up(num_rows, warp_size) * warp_size;

  for (cudf::size_type i = grid.thread_rank() / cg_size; i < loop_bound;
       i += grid.num_threads() / cg_size) {
    bool is_active    = (i < num_rows) && (!has_probe_mask || probe_mask[i]);
    int32_t sync_mask = tile.ballot(is_active);

    if (is_active) {
      // Warp divergence slows down the read.
      __syncwarp(sync_mask);
      ProbeKeyType query = probe_rows[i];

      auto probing_iter =
        probing_scheme.template make_iterator<MarkSetRefType::bucket_size>(query, bucket_extent);
      cuco::detail::equal_result status{};
      do {
        auto mutable_entry = storage.data() + *probing_iter;
        auto entry_value   = *(mutable_entry + bucket_tile.thread_rank());
        // In the low_selectivity case the branch is optimized out. If selectivity is high, we
        // may save some time by checking if the entry is already marked before doing the
        // full condition evaluation.
        if (is_low_selectivity || !is_marked(entry_value.first)) {
          status = predicate.operator()<cuco::detail::is_insert::NO>(query, entry_value);
          // if status is equal, the entry is not empty and all equality conditions pass, continue
          // on to mixed conditions
          if (status == cuco::detail::equal_result::EQUAL) {
            constexpr cudf::size_type output_row = 0;
            cudf::ast::detail::value_expression_result<bool, has_nulls> output{};
            // if this is a mixed join, evaluate. otherwise, if statement is optimized out
            if constexpr (is_mixed) {
              cudf::size_type query_index = static_cast<cudf::size_type>(query.second);
              cudf::size_type entry_index = static_cast<cudf::size_type>(entry_value.second);

              evaluator.evaluate(output, entry_index, query_index, output_row, predicate_storage);
            }
            // if equality only, there is no branch. if mixed, evaluate if we match, then enter if
            // true.
            if (!is_mixed || (output.is_valid() && output.value())) {
              auto expected = query.first;
              auto desired  = set_mark(expected);

              cuda::atomic_ref<cudf::hash_value_type, cuda::thread_scope_device> key{
                (mutable_entry + bucket_tile.thread_rank())->first};
              // Set mark on hash table entry iff it's not already marked.
              auto is_success =
                key.compare_exchange_strong(expected, desired, cuda::memory_order_relaxed);
              if (is_success) {
                // The marked entries count is the join result size. Don't double-count marked
                // entries.
                ++mark_counter;
              }
            }
          }
        }
        ++probing_iter;
      } while (status != cuco::detail::equal_result::EMPTY);
    }
  }

  // final output reduction for mark_counter
  auto warp_sum = cg::reduce(tile, mark_counter, cg::plus<int32_t>{});
  cg::invoke_one(tile, [&]() { cta_mark_counter.fetch_add(warp_sum, cuda::memory_order_relaxed); });
  block.sync();

  cg::invoke_one(cg::this_thread_block(), [&]() {
    global_mark_counter.fetch_add(cta_mark_counter.load(cuda::memory_order_relaxed),
                                  cuda::memory_order_relaxed);
  });
}

/**
 * @brief This kernel is effectively a buffer writer - conditionally accept elements from an input
 * source, and write them in a coalesced manner using shared memory buffer to the output buffer.
 * Accepts an operator that provides (1) the iterator functionality, (2) a predicate condition
 * for acceptance, and (3) an 'accept' operator that transforms the input into the output.
 * Used for scanning selected join positions.
 * The API for TaskOperator is as follows:
 * 1) The TaskOperator must include its own storage for the input data.
 * 2) get_bucket(index) must exist and return a storage type that supports the [] bracket operators.
 * For 1-D storage this can just be a pointer 3) predicate(InputType) must accept an element of the
 * underlying storage type returned by get_bucket and return a boolean determining if we should
 * include the element in the output set 4) accept(InputType) must accept the InputType, optionally
 * perform a transform, and return OutputType to be stored in the final output 5) num_buckets() must
 * return the total number of buckets 6) elements_per_bucket() must indicate the total number of
 * elements in each bucket, indexable by []
 *
 * @tparam block_size the block_size used to launch this kernel, as a static parameter
 * @tparam InputType the type of element retrieved from the TaskOperator
 * @tparam OutputType the type of each element in the output array
 * @tparam TaskOperator a suitable operator based on the TaskOperator API
 * @param[out] out the output storage array
 * @param[out] global_offset_ref a counter that will store how many elements are in the final output
 * array
 * @param[in] op the input operator
 */
template <int32_t block_size,
          typename InputType,
          typename OutputType,
          typename TaskOperator,
          bool has_mask>
__global__ __launch_bounds__(block_size) void iterator_to_vector_if(
  OutputType* out, cudf::size_type* global_offset_ref, TaskOperator op, bool const* mask)
{
  // currently, we use 1 element per bucket for performance.
  // leaving this functionality in because it doesn't hurt perf and we may find a way to tune it
  // later
  constexpr uint32_t cg_size   = op.elements_per_bucket();
  constexpr uint32_t warp_size = 32;
  const auto grid              = cooperative_groups::this_grid();
  const auto block             = cg::this_thread_block();
  const auto bucket_tile       = cg::tiled_partition<cg_size, cg::thread_block>(block);
  const auto tile              = cg::tiled_partition<warp_size>(block);

  // grid stride loop
  const auto loop_bound = gqe::utility::divide_round_up(op.num_buckets(), warp_size) * warp_size;

  // shared reducer tooling
  constexpr int buffer_capacity_factor = 4;
  constexpr int buffer_capacity        = block_size * buffer_capacity_factor;
  constexpr int warp_buffer_capacity   = warp_size * buffer_capacity_factor;
  const int warp_buffer_offset         = (warp_buffer_capacity * tile.meta_group_rank());
  uint32_t build_buffer_offset         = 0;
  // Shared buffer divided into exclusive per-warp chunks.
  __shared__ alignas(buffer_capacity) OutputType build_buffer[buffer_capacity];
  // Output reference.
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset(*global_offset_ref);

  for (cudf::size_type i = grid.thread_rank() / cg_size; i < loop_bound;
       i += grid.num_threads() / cg_size) {
    InputType slot;
    bool do_fill = false;
    // Retrieve element and evaluate predicate.
    if (i < op.num_buckets()) {
      auto bucket = op.get_bucket(i);
      slot        = bucket[bucket_tile.thread_rank()];
      do_fill     = op.predicate(slot) && (!has_mask || mask[i]);
    }
    // If any thread has output to write, enter buffer filling stage, else move on.
    bool work_todo = tile.any(do_fill);
    while (work_todo) {
      uint32_t offset = 0;
      if (do_fill) {
        cg::coalesced_group active_group = cg::coalesced_threads();
        offset                           = build_buffer_offset + active_group.thread_rank();
        if (offset < warp_buffer_capacity) {
          build_buffer[offset + warp_buffer_offset] = op.accept(slot);
          do_fill                                   = false;
        }
      }
      // re-unify build_buffer_offset - add 1, because the slot represented by max has already
      // been used.
      offset              = cg::reduce(tile, offset, cg::greater<uint32_t>{});
      build_buffer_offset = offset + 1;
      // This tests if the buffer is full so we can do fully-aligned writes to global, otherwise
      // wait. we set the while loop condition here because it's equivalent to tile.any(do_fill) but
      // avoids a warpsync; no break potentially enables compiler to do a better job on instruction
      // layout.
      if (work_todo = (offset >= warp_buffer_capacity)) {
        // reset shared offset
        build_buffer_offset = 0;
        // grab global write offset
        cudf::size_type flush_offset = cg::invoke_one_broadcast(tile, [&]() {
          return global_offset.fetch_add(warp_buffer_capacity, cuda::memory_order_relaxed);
        });
#pragma unroll
        // use warp_size explicitly here instead of tile.size() to guarantee we can unroll
        for (int16_t k = tile.thread_rank(); k < warp_buffer_capacity; k += warp_size) {
          out[flush_offset + k] = build_buffer[k + warp_buffer_offset];
        }
      }
    }
  }
  // Epilogue pass for straggling buffer entries.
  if (build_buffer_offset > 0) {
    cudf::size_type flush_offset = cg::invoke_one_broadcast(tile, [&]() {
      return global_offset.fetch_add(build_buffer_offset, cuda::memory_order_relaxed);
    });

    for (int16_t k = tile.thread_rank(); k < build_buffer_offset; k += tile.num_threads()) {
      out[flush_offset + k] = build_buffer[k + warp_buffer_offset];
    }
  }
}

// We can't use thrust lambdas in constructor, so we use this quick kernel.
template <typename InputIteratorType, typename OutputType, int32_t block_size>
__global__ __launch_bounds__(block_size) void iter_to_column(InputIteratorType first,
                                                             OutputType* out,
                                                             cudf::size_type num_rows)
{
  constexpr uint32_t warp_size = 32;
  const auto loop_bound        = gqe::utility::divide_round_up(num_rows, warp_size) * warp_size;
  for (cudf::size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < loop_bound;
       i += blockDim.x * gridDim.x) {
    if (i < num_rows) { out[i] = *(first + i); }
  }
}

// Helper to declutter template instantiation.
template <int32_t block_size,
          typename ComparatorAdapterType,
          typename MarkSetRefType,
          typename InputIteratorType,
          uint32_t cg_size,
          bool has_nulls,
          bool is_mixed,
          bool has_probe_mask>
static auto get_mark_probe_kernel(bool is_low_selectivity)
{
  if (is_low_selectivity) {
    return mark_probe<block_size,
                      ComparatorAdapterType,
                      MarkSetRefType,
                      InputIteratorType,
                      cg_size,
                      has_nulls,
                      is_mixed,
                      true,
                      has_probe_mask>;
  } else {
    return mark_probe<block_size,
                      ComparatorAdapterType,
                      MarkSetRefType,
                      InputIteratorType,
                      cg_size,
                      has_nulls,
                      is_mixed,
                      false,
                      has_probe_mask>;
  }
}

}  // namespace gqe::qep::mark_join_detail
