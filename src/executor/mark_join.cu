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

#include <gqe/executor/mark_join.hpp>

#include "mark_join.cuh"

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/kernel_fusion_helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/open_addressing/functors.cuh>  // slot_is_filled
#include <cuco/operator.hpp>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/atomic>
#include <cudf/ast/detail/expression_evaluator.cuh>
#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/hashing.hpp>  // hash_value_type
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace cg = cooperative_groups;

namespace gqe {

namespace {

// This is just a utility function for computing bloom filter blocks, borrowed from
// hardcoded Q21. Can be removed later when a robust version is added to utility.
/**
 * @brief Get the bloom filter number of blocks based on total number of bytes that bloom filter
 * takes up.
 *
 * @tparam FilterType
 * @param[in] filter_bytes The total number of bytes that the bloom filter takes up.
 */
template <typename FilterType>
cuco::extent<std::size_t> get_bloom_filter_blocks(std::size_t filter_bytes)
{
  std::size_t num_sub_filters =
    filter_bytes / (sizeof(typename FilterType::word_type) * FilterType::words_per_block);

  // returning 0 here causes a lot of problems but is common in unit tests, so check
  return num_sub_filters ? num_sub_filters : 1;
}

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
  auto probing_scheme         = mark_set_ref.probing_scheme();
  auto bucket_extent          = mark_set_ref.bucket_extent();
  auto storage                = mark_set_ref.storage_ref();
  auto empty_key_sentinel_key = mark_set_ref.empty_key_sentinel();
  auto predicate              = cuco::detail::equal_wrapper{
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

      auto probing_iter = probing_scheme(query, bucket_extent);
      cuco::detail::equal_result status{};
      do {
        auto mutable_entry = (storage.data() + *probing_iter)->data();
        auto entry_value   = *(mutable_entry + bucket_tile.thread_rank());
        // In the low_selectivity case the branch is optimized out. If selectivity is high, we
        // may save some time by checking if the entry is already marked before doing the
        // full condition evaluation.
        if (is_low_selectivity || !detail::is_marked(entry_value.first)) {
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
              auto desired  = detail::set_mark(expected);

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
 * Used for both scanning the hash table to materialize join positions, and for prefiltering probe
 * rows.
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
}  // namespace

namespace detail {

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

mark_join::mark_join(cudf::table_view const& build,
                     cudf::column_view const& build_mask,
                     bool is_cached,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
  : _build{build},
    _build_num_valid_rows(gqe::utility::get_num_active_keys(build.num_rows(), build_mask)),
    _nulls_equal{compare_nulls},
    _mark_set{static_cast<size_t>(_build_num_valid_rows),
              load_factor,
              cuco::empty_key{
                cuco::pair{detail::unset_mark(std::numeric_limits<cudf::hash_value_type>::max()),
                           cudf::experimental::row::rhs_index_type{-1}}},  // empty_key_sentinel
              equality_comparator_adapter{*cudf::table_device_view::create(build, stream),
                                          *cudf::table_device_view::create({}, stream)},
              probing_scheme_type{},
              cuco::thread_scope_device,
              cuco::storage<_slots_per_bucket>{},
              rmm::mr::stream_allocator_adaptor(rmm::mr::polymorphic_allocator<key_type>{}, stream),
              stream.value()},
    _bloom_filter{get_bloom_filter_blocks<bloom_filter_type>(build.num_rows()),
                  cuco::cuda_thread_scope<cuda::thread_scope_device>{},
                  bloom_filter_policy_type{},
                  bloom_filter_allocator_type{bloom_filter_allocator_instance_type{}, stream},
                  stream.value()},
    _num_marks(0),
    _is_cached(is_cached)
{
  GQE_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");
  constexpr int block_size  = 1024;
  auto build_device_view    = cudf::table_device_view::create(_build, stream);
  auto const d_build_hasher = build_hasher_type{*build_device_view};

  auto const build_hash_row_pairs =
    cudf::detail::make_counting_transform_iterator(0, d_build_hasher);

  rmm::device_uvector<cudf::hash_value_type> hashes(build_device_view->num_rows(), stream, mr);
  auto iter_kernel =
    iter_to_column<decltype(build_hash_row_pairs), cudf::hash_value_type, block_size>;

  int iter_grid_size = utility::detect_launch_grid_size(iter_kernel, block_size);
  iter_kernel<<<iter_grid_size, block_size, 0, stream>>>(
    build_hash_row_pairs, hashes.data(), build_device_view->num_rows());

  const create_input_pair_from_column<cudf::hash_value_type,
                                      cudf::experimental::row::rhs_index_type>
    build_input_pair{hashes.data()};
  auto const build_iter = cudf::detail::make_counting_transform_iterator(0, build_input_pair);

  if (build_mask.is_empty()) {
    _mark_set.insert_async(build_iter, build_iter + build_device_view->num_rows(), stream.value());
    _bloom_filter.add_async(hashes.begin(), hashes.end(), stream.value());
  } else {
    GQE_EXPECTS(build_mask.type().id() == cudf::type_id::BOOL8,
                "The build mask of mark join is not a boolean column");
    _mark_set.insert_if_async(build_iter,
                              build_iter + build_device_view->num_rows(),
                              build_mask.data<bool>(),
                              gqe::utility::identity_pred{},
                              stream.value());
    _bloom_filter.add_if_async(hashes.begin(),
                               hashes.end(),
                               build_mask.data<bool>(),
                               gqe::utility::identity_pred{},
                               stream.value());
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join::perform_mark_join(cudf::table_view const& probe,
                             cudf::column_view const& probe_mask,
                             bool is_anti_join,
                             cudf::table_view const& left_conditional,
                             cudf::table_view const& right_conditional,
                             cudf::ast::expression const* binary_predicate,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  GQE_LOG_TRACE(
    "perform mark join is_anti={} eq cols={} right eq cols={} left condition cols={} right "
    "condition "
    "cols={}",
    is_anti_join,
    _build.num_columns(),
    probe.num_columns(),
    left_conditional.num_columns(),
    right_conditional.num_columns());

  auto const build_equality_keys_view = cudf::table_device_view::create(_build, stream);
  auto const probe_equality_keys_view = cudf::table_device_view::create(probe, stream);
  auto const build_conditional_keys_view =
    cudf::table_device_view::create(left_conditional, stream);
  auto const probe_conditional_keys_view =
    cudf::table_device_view::create(right_conditional, stream);

  auto comparator_adapter =
    equality_comparator_adapter(*build_equality_keys_view, *probe_equality_keys_view);

  using comparator_type = decltype(comparator_adapter);

  std::unique_ptr<cudf::ast::detail::expression_parser> parser;
  cudf::ast::detail::expression_device_view device_expression_data;
  uint32_t shared_memory_per_thread = 0;
  if (binary_predicate) {
    constexpr bool is_mixed = true;
    const bool has_nulls =
      binary_predicate->may_evaluate_null(left_conditional, right_conditional, stream);
    parser = std::make_unique<cudf::ast::detail::expression_parser>(
      *binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr);
    shared_memory_per_thread = parser->shmem_per_thread;
    device_expression_data   = parser->device_expression_data;
    if (has_nulls) {
      return _perform_mark_join<comparator_type, true, is_mixed>(*probe_equality_keys_view,
                                                                 *build_conditional_keys_view,
                                                                 *probe_conditional_keys_view,
                                                                 probe_mask,
                                                                 device_expression_data,
                                                                 comparator_adapter,
                                                                 is_anti_join,
                                                                 shared_memory_per_thread,
                                                                 stream,
                                                                 mr);
    } else {
      return _perform_mark_join<comparator_type, false, is_mixed>(*probe_equality_keys_view,
                                                                  *build_conditional_keys_view,
                                                                  *probe_conditional_keys_view,
                                                                  probe_mask,
                                                                  device_expression_data,
                                                                  comparator_adapter,
                                                                  is_anti_join,
                                                                  shared_memory_per_thread,
                                                                  stream,
                                                                  mr);
    }
  }
  constexpr bool has_nulls = false;
  constexpr bool is_mixed  = false;
  // if no mixed conditions, we supply an always-true evaluator that should get optimized out
  return _perform_mark_join<comparator_type, has_nulls, is_mixed>(*probe_equality_keys_view,
                                                                  *build_conditional_keys_view,
                                                                  *probe_conditional_keys_view,
                                                                  probe_mask,
                                                                  device_expression_data,
                                                                  comparator_adapter,
                                                                  is_anti_join,
                                                                  shared_memory_per_thread,
                                                                  stream,
                                                                  mr);
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

bool mark_join::is_low_selectivity() const
{
  // Current heuristic depends on if we are using hash map caching.
  return !_is_cached;
}

template <typename ComparatorAdapterType, bool has_nulls, bool is_mixed>
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join::_perform_mark_join(
  cudf::table_device_view const& probe_equality_device_view,
  cudf::table_device_view const& build_conditional_device_view,
  cudf::table_device_view const& probe_conditional_device_view,
  cudf::column_view const& probe_mask,
  cudf::ast::detail::expression_device_view const& expression_device_view,
  ComparatorAdapterType const& comparator_adapter,
  bool is_anti_join,
  uint32_t shared_memory_per_thread,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  // Kernels in mark join are primarily warp-synchronous; very large blocks give good performance.
  constexpr int block_size = 1024;

  // Prefilter stage: instantiate iterator to materialize (hash, row index) iterator from probe
  // side.
  using build_hasher_type = device_row_hasher<cudf::hashing::detail::default_hash>;
  using ProbeKeyType = cuco::pair<cudf::hash_value_type, cudf::experimental::row::lhs_index_type>;
  auto const d_probe_hasher = build_hasher_type{probe_equality_device_view};
  const create_input_pair<build_hasher_type, cudf::experimental::row::lhs_index_type>
    probe_input_pair{d_probe_hasher};
  auto const row_iter        = cudf::detail::make_counting_transform_iterator(0, probe_input_pair);
  auto const probe_mask_iter = probe_mask.is_empty() ? nullptr : probe_mask.data<bool>();
  GQE_EXPECTS(probe_mask.is_empty() || probe_mask.type().id() == cudf::type_id::BOOL8,
              "The probe mask of mark join is not a boolean column");

  auto bloom_filter_ref = _bloom_filter.ref();
  // Output remaining rows and the counter for remaining rows.
  rmm::device_uvector<ProbeKeyType> probe_rows(probe_equality_device_view.num_rows(), stream, mr);
  rmm::device_scalar<cudf::size_type> row_offset_counter(0, stream, mr);
  // Build an operator that functions as iterator and provides predicate, accept operations.
  mark_join_prefilter_operator<decltype(row_iter), decltype(bloom_filter_ref), ProbeKeyType>
    prefilter_op{row_iter, bloom_filter_ref, probe_equality_device_view.num_rows()};
  // row filter kernel
  auto row_filter_kernel =
    probe_mask.is_empty()
      ? iterator_to_vector_if<block_size, ProbeKeyType, ProbeKeyType, decltype(prefilter_op), false>
      : iterator_to_vector_if<block_size, ProbeKeyType, ProbeKeyType, decltype(prefilter_op), true>;
  int row_filter_grid_size = utility::detect_launch_grid_size(row_filter_kernel, block_size, 0);
  row_filter_kernel<<<row_filter_grid_size, block_size, 0, stream>>>(
    probe_rows.data(), row_offset_counter.data(), prefilter_op, probe_mask_iter);
  GQE_LOG_TRACE("mark rowfilter count offset={}", row_offset_counter.value(stream));

  auto mark_set_ref                 = _mark_set.ref(cuco::op::contains_tag{});
  using mark_set_key_type           = typename decltype(mark_set_ref)::key_type;
  const int shared_memory_per_block = shared_memory_per_thread * block_size;
  // Perform hash table probe against remaining rows
  auto mark_probe_kernel =
    get_mark_probe_kernel<block_size,
                          ComparatorAdapterType,
                          decltype(mark_set_ref),
                          ProbeKeyType,
                          _slots_per_bucket,  // cg_size
                          has_nulls,
                          is_mixed,
                          /*has_probe_mask=*/false>(this->is_low_selectivity());
  int probe_grid_size =
    utility::detect_launch_grid_size(mark_probe_kernel, block_size, shared_memory_per_block);

  rmm::device_scalar<cudf::size_type> mark_counter(0, stream, mr);
  mark_probe_kernel<<<probe_grid_size, block_size, shared_memory_per_block, stream>>>(
    build_conditional_device_view,
    probe_conditional_device_view,
    expression_device_view,
    mark_set_ref,
    comparator_adapter,
    probe_rows.data(),
    /*probe_mask=*/nullptr,  // already applied in prefilter
    row_offset_counter.value(stream),
    mark_counter.data());
  cudf::size_type marked_row_count = mark_counter.value(stream);

  GQE_LOG_TRACE("mark probe count offset={}", marked_row_count);
  // Update our mark counter - must be kept up to date for scan to work properly; atomic for
  // multi-gpu
  _num_marks.fetch_add(marked_row_count);
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_positions;
  // a pair is expected even though we only have one set of positions
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left;
  if (_is_cached) {
    // if it's cached we don't instantiate the positions yet
    build_positions = std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  } else {
    // If not cached, go ahead and perform mark_scan and retrieve active positions to return.
    build_positions = _compute_positions_list_from_map(is_anti_join, stream, mr);
  }
  return std::make_pair(std::move(left), std::move(build_positions));
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::_compute_positions_list_from_map(
  bool is_anti_join, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (is_anti_join) {
    return _mark_scan<true>(stream, mr);
  } else {
    return _mark_scan<false>(stream, mr);
  }
}

template <bool is_anti_join>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_join::_mark_scan(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  constexpr int shared_memory_per_block = 0;
  constexpr int block_size              = 1024;
  auto mark_set_ref                     = _mark_set.ref(cuco::op::contains_tag{});
  using mark_set_key_type               = typename decltype(mark_set_ref)::key_type;
  // num_marks should not change while we are in this function; loading once should be fine
  cudf::size_type num_marks        = _num_marks.load();
  cudf::size_type result_row_count = is_anti_join ? _build_num_valid_rows - num_marks : num_marks;

  rmm::device_uvector<cudf::size_type> build_positions(result_row_count, stream, mr);
  rmm::device_scalar<cudf::size_type> mark_scan_offset(0, stream, mr);

  mark_join_scan_operator<decltype(mark_set_ref),
                          decltype(mark_set_ref.storage_ref()),
                          mark_set_key_type,
                          _slots_per_bucket,
                          is_anti_join>
    mark_scan_op{mark_set_ref,
                 mark_set_ref.storage_ref(),
                 static_cast<cudf::size_type>(mark_set_ref.storage_ref().num_buckets())};

  auto mark_scan_kernel = iterator_to_vector_if<block_size,
                                                mark_set_key_type,
                                                cudf::size_type,
                                                decltype(mark_scan_op),
                                                /*has_mask=*/false>;

  int scan_grid_size =
    utility::detect_launch_grid_size(mark_scan_kernel, block_size, shared_memory_per_block);

  mark_scan_kernel<<<scan_grid_size, block_size, 0, stream>>>(
    build_positions.data(), mark_scan_offset.data(), mark_scan_op, /*mask=*/nullptr);

  GQE_LOG_TRACE("mark join scan offset={}", mark_scan_offset.value(stream));

  return std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(build_positions));
}

}  // namespace detail

mark_join::~mark_join() = default;

mark_join::mark_join(cudf::table_view const& build,
                     cudf::column_view const& build_mask,
                     bool is_cached,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  _impl = std::make_unique<detail::mark_join>(
    build, build_mask, is_cached, compare_nulls, load_factor, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join::perform_mark_join(cudf::table_view const& probe,
                             cudf::column_view const& probe_mask,
                             bool is_anti_join,
                             cudf::table_view const& left_conditional,
                             cudf::table_view const& right_conditional,
                             cudf::ast::expression const* binary_predicate,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  return _impl->perform_mark_join(probe,
                                  probe_mask,
                                  is_anti_join,
                                  left_conditional,
                                  right_conditional,
                                  binary_predicate,
                                  stream,
                                  mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>>
mark_join::compute_positions_list_from_cached_map(bool is_anti_join,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) const
{
  return _impl->_compute_positions_list_from_map(is_anti_join);
}

// wrapper functions for mark_join
static inline std::unique_ptr<rmm::device_uvector<cudf::size_type>> uniform_left_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::null_equality compare_nulls,
  bool is_anti_join,
  double load_factor,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::column_view empty_mask;
  // there is no AST to evaluate for equality join
  constexpr cudf::ast::expression const* binary_predicate = nullptr;
  constexpr bool is_cached                                = false;
  auto build_equality_keys_view = cudf::table_device_view::create(left_equality, stream);
  auto probe_equality_keys_view = cudf::table_device_view::create(right_equality, stream);
  auto join_obj =
    gqe::mark_join(left_equality, empty_mask, is_cached, compare_nulls, load_factor, stream, mr);
  auto positions = join_obj.perform_mark_join(right_equality,
                                              empty_mask,
                                              is_anti_join,
                                              cudf::table_view({}),
                                              cudf::table_view({}),
                                              binary_predicate,
                                              stream,
                                              mr);
  return std::move(positions.second);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_semi_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  constexpr bool is_anti_join = false;

  GQE_LOG_TRACE("mark join left semi eq cols={} right eq cols={}",
                left_equality.num_columns(),
                right_equality.num_columns());
  return uniform_left_mark_join(
    left_equality, right_equality, compare_nulls, is_anti_join, load_factor, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_anti_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  constexpr bool is_anti_join = true;

  GQE_LOG_TRACE("mark join left anti eq cols={} right eq cols={}",
                left_equality.num_columns(),
                right_equality.num_columns());

  return uniform_left_mark_join(
    left_equality, right_equality, compare_nulls, is_anti_join, load_factor, stream, mr);
}

static inline std::unique_ptr<rmm::device_uvector<cudf::size_type>> mixed_left_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::null_equality compare_nulls,
  bool is_anti_join,
  double load_factor,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::column_view empty_mask;
  constexpr bool is_cached = false;
  auto join_obj =
    gqe::mark_join(left_equality, empty_mask, is_cached, compare_nulls, load_factor, stream, mr);
  auto positions = join_obj.perform_mark_join(right_equality,
                                              empty_mask,
                                              is_anti_join,
                                              left_conditional,
                                              right_conditional,
                                              &binary_predicate,
                                              stream,
                                              mr);
  return std::move(positions.second);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mixed_left_semi_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  constexpr bool is_anti_join = false;

  GQE_LOG_TRACE(
    "mark join left semi eq cols={} right eq cols={} left condition cols={} right condition "
    "cols={}",
    left_equality.num_columns(),
    right_equality.num_columns(),
    left_conditional.num_columns(),
    right_conditional.num_columns());

  return mixed_left_mark_join(left_equality,
                              right_equality,
                              left_conditional,
                              right_conditional,
                              binary_predicate,
                              compare_nulls,
                              is_anti_join,
                              load_factor,
                              stream,
                              mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> mixed_left_anti_mark_join(
  cudf::table_view const& left_equality,
  cudf::table_view const& right_equality,
  cudf::table_view const& left_conditional,
  cudf::table_view const& right_conditional,
  cudf::ast::expression const& binary_predicate,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  constexpr bool is_anti_join = true;

  GQE_LOG_TRACE(
    "mark join left anti eq cols={} right eq cols={} left condition cols={} right condition "
    "cols={}",
    left_equality.num_columns(),
    right_equality.num_columns(),
    left_conditional.num_columns(),
    right_conditional.num_columns());

  return mixed_left_mark_join(left_equality,
                              right_equality,
                              left_conditional,
                              right_conditional,
                              binary_predicate,
                              compare_nulls,
                              is_anti_join,
                              load_factor,
                              stream,
                              mr);
}

}  // namespace gqe
