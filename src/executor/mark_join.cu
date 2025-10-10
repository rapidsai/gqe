/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "mark_join.cuh"
#include <gqe/executor/mark_join.hpp>

#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>

#include <cuco/detail/equal_wrapper.cuh>
#include <cuco/detail/open_addressing/functors.cuh>  // slot_is_filled
#include <cuco/operator.hpp>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_multiset.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>

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

#include <gqe/utility/cuda.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace cg = cooperative_groups;

namespace gqe {

namespace {

template <int32_t block_size,
          typename key_type,
          typename comparator_adapter_type,
          typename mark_set_ref_type,
          typename input_it_type,
          bool is_low_selectivity>
__global__ __launch_bounds__(block_size) void mark_probe(mark_set_ref_type mark_set_ref,
                                                         comparator_adapter_type join_predicate,
                                                         input_it_type first,
                                                         cudf::size_type num_rows,
                                                         cudf::size_type* global_mark_counter_ref)
{
  constexpr uint32_t warp_size = 32;

  const auto grid  = cooperative_groups::this_grid();
  const auto block = cg::this_thread_block();
  const auto tile  = cg::tiled_partition<warp_size>(block);

  auto probing_scheme         = mark_set_ref.probing_scheme();
  auto bucket_extent          = mark_set_ref.bucket_extent();
  auto storage                = mark_set_ref.storage_ref();
  auto empty_key_sentinel_key = mark_set_ref.empty_key_sentinel();
  auto predicate              = cuco::detail::equal_wrapper{
    mark_set_ref.empty_key_sentinel(), mark_set_ref.erased_key_sentinel(), join_predicate};

  cudf::size_type mark_counter = 0;
  __shared__ cuda::atomic<cudf::size_type, cuda::thread_scope_block> cta_mark_counter;
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_mark_counter{
    *global_mark_counter_ref};

  cg::invoke_one(cg::this_thread_block(),
                 [&]() { cta_mark_counter.store(0, cuda::memory_order_relaxed); });
  __syncthreads();

  const auto loop_bound = gqe::utility::divide_round_up(num_rows, warp_size) * warp_size;

  for (cudf::size_type i = grid.thread_rank(); i < loop_bound; i += grid.num_threads()) {
    int32_t sync_mask = tile.ballot(i < num_rows);

    if (i < num_rows) {
      constexpr int slot_index = 0;

      // Warp divergence slows down the read.
      __syncwarp(sync_mask);
      auto query = *(first + i);

      auto probing_iter = probing_scheme(query, bucket_extent);
      while (true) {
        auto mutable_entry = (storage.data() + *probing_iter)->data() + slot_index;
        auto entry_value   = *mutable_entry;
        // In the low_selectivity case the if is optimized out. If selectivity is high, we
        // may save some time by checking if the entry is already marked before doing the
        // full condition evaluation.
        if (is_low_selectivity || !detail::is_marked(entry_value.first)) {
          auto status = predicate.operator()<cuco::detail::is_insert::NO>(query, entry_value);
          if (status == cuco::detail::equal_result::EQUAL) {
            auto expected = query.first;
            auto desired  = detail::set_mark(expected);

            cuda::atomic_ref<cudf::hash_value_type, cuda::thread_scope_device> key{
              mutable_entry->first};
            auto is_success =
              key.compare_exchange_strong(expected, desired, cuda::memory_order_relaxed);
            if (is_success) {
              // The marked entries count is the join result size. Thus, don't double-count marked
              // entries.
              ++mark_counter;
            }
          } else if (status == cuco::detail::equal_result::EMPTY) {
            break;
          }
        }
        ++probing_iter;
      }
    }
  }

  auto warp_sum = cg::reduce(tile, mark_counter, cg::plus<int32_t>{});
  cg::invoke_one(tile, [&]() { cta_mark_counter.fetch_add(warp_sum, cuda::memory_order_relaxed); });
  __syncthreads();

  cg::invoke_one(cg::this_thread_block(), [&]() {
    global_mark_counter.fetch_add(cta_mark_counter.load(cuda::memory_order_relaxed),
                                  cuda::memory_order_relaxed);
  });
}

template <int32_t block_size, typename key_type, typename mark_set_ref_type>
__global__ __launch_bounds__(block_size) void mark_scan(mark_set_ref_type mark_set_ref,
                                                        cudf::size_type* build_positions,
                                                        cudf::size_type* global_offset_ref,
                                                        const bool is_anti_join)
{
  assert(block_size == blockDim.x);

  constexpr uint32_t warp_size  = 32;
  constexpr int buffer_capacity = block_size * 4;

  const auto grid            = cooperative_groups::this_grid();
  const auto block           = cg::this_thread_block();
  const auto block_partition = cg::tiled_partition<block_size, cg::thread_block>(block);
  const auto tile            = cg::tiled_partition<warp_size>(block);

  __shared__ cudf::size_type build_buffer[buffer_capacity];
  __shared__ cuda::atomic<int16_t, cuda::thread_scope_block> buffer_offset;
  cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> global_offset(*global_offset_ref);

  auto storage_ref = mark_set_ref.storage_ref();
  static_assert(mark_set_ref_type::storage_ref_type::bucket_type::bucket_size == 1);

  constexpr bool has_payload = false;
  auto const is_filled = cuco::detail::open_addressing_ns::slot_is_filled<has_payload, key_type>{
    mark_set_ref.empty_key_sentinel(), mark_set_ref.erased_key_sentinel()};

  cg::invoke_one(block, [&]() { buffer_offset.store(0, cuda::memory_order_relaxed); });
  block.sync();

  const auto loop_bound =
    gqe::utility::divide_round_up(mark_set_ref.capacity(), block_size) * block_size;
  for (cudf::size_type index = grid.thread_rank(); index < loop_bound;
       index += grid.num_threads()) {
    bool do_fill = false;
    cuco::pair<cudf::hash_value_type, cudf::experimental::row::rhs_index_type> slot{};

    if (index < mark_set_ref.capacity()) {
      slot    = storage_ref[index][0];
      do_fill = is_filled(slot) && (detail::is_marked(slot.first) ^ is_anti_join);
    }

    while (true) {
      int32_t offset = 0;  // Only 32-bit types have hardware-accelerated reduce.
      // Shared memory allocations are allocated on a per-warp basis.
      int32_t num_slots = cg::reduce(tile, static_cast<int32_t>(do_fill), cg::plus<int32_t>{});
      if (do_fill) {
        cg::coalesced_group active_group = cg::coalesced_threads();
        offset =
          cg::invoke_one_broadcast(
            active_group,
            [&]() { return buffer_offset.fetch_add(num_slots, cuda::memory_order_relaxed); }) +
          active_group.thread_rank();
        if (offset < buffer_capacity) {
          build_buffer[offset] = static_cast<cudf::size_type>(slot.second);
          do_fill              = false;
        }
      }
      // Full block will sync on reduce before entering the write-buffer segment.
      auto max_offset = cg::reduce(block_partition, offset, cg::greater<int32_t>{});
      if (max_offset >= buffer_capacity) {
        // Full block shares the global memory allocation.
        cudf::size_type flush_offset = cg::invoke_one_broadcast(block_partition, [&]() {
          buffer_offset.store(0, cuda::memory_order_relaxed);
          return global_offset.fetch_add(buffer_capacity, cuda::memory_order_relaxed);
        });
        for (int16_t i = block.thread_rank(); i < buffer_capacity; i += block.num_threads()) {
          build_positions[flush_offset + i] = build_buffer[i];
        }
      } else {
        // All threads in block share max_offset, so all threads will break if we fail the
        // condition.
        break;
      }
      // During while loop execution, all threads hit this sync if we continue, so each iteration is
      // block-synchronous. This will ensure the buffer_offset = 0 is visible to all threads,
      // preventing a buffer race.
      block.sync();
    }
  }

  block.sync();

  auto current_offset = buffer_offset.load(cuda::memory_order_relaxed);
  if (current_offset > 0) {
    cudf::size_type flush_offset = cg::invoke_one_broadcast(block_partition, [&]() {
      return global_offset.fetch_add(current_offset, cuda::memory_order_relaxed);
    });

    for (int16_t i = block.thread_rank(); i < current_offset; i += block.num_threads()) {
      build_positions[flush_offset + i] = build_buffer[i];
    }
  }
}
}  // namespace

namespace detail {

mark_join::mark_join(cudf::table_view const& build,
                     bool is_cached,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
  : _build{build},
    _nulls_equal{compare_nulls},
    _mark_set{build.num_rows(),
              load_factor,
              cuco::empty_key{
                cuco::pair{detail::unset_mark(std::numeric_limits<cudf::hash_value_type>::max()),
                           cudf::experimental::row::rhs_index_type{-1}}},  // empty_key_sentinel
              equality_comparator_adapter{*cudf::table_device_view::create(build, stream),
                                          *cudf::table_device_view::create({}, stream)},
              probing_scheme_type{},
              cuco::thread_scope_device,
              cuco::storage<1>{},
              rmm::mr::stream_allocator_adaptor(rmm::mr::polymorphic_allocator<key_type>{}, stream),
              stream.value()},
    _num_marks(0),
    _is_cached(is_cached)
{
  GQE_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");

  auto build_device_view    = cudf::table_device_view::create(_build, stream);
  auto const d_build_hasher = build_hasher_type{*build_device_view};
  const create_input_pair<build_hasher_type, cudf::experimental::row::rhs_index_type>
    build_input_pair{d_build_hasher};

  auto const build_iter = cudf::detail::make_counting_transform_iterator(0, build_input_pair);

  _mark_set.insert_async(build_iter, build_iter + build_device_view->num_rows(), stream.value());
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join::perform_mark_join(cudf::table_view const& probe,
                             bool is_anti_join,
                             cudf::table_view const& left_conditional,
                             cudf::table_view const& right_conditional,
                             cudf::ast::expression const* binary_predicate,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  auto build_equality_keys_view     = cudf::table_device_view::create(_build, stream);
  auto probe_equality_keys_view     = cudf::table_device_view::create(probe, stream);
  uint32_t shared_memory_per_thread = 0;

  GQE_LOG_TRACE(
    "perform mark join is_anti={} eq cols={} right eq cols={} left condition cols={} right "
    "condition "
    "cols={}",
    is_anti_join,
    _build.num_columns(),
    probe.num_columns(),
    left_conditional.num_columns(),
    right_conditional.num_columns());

  // Support 3 cases; equality-only, mixed with nulls, and mixed without nulls.
  if (!binary_predicate) {
    auto comparator_adapter =
      equality_comparator_adapter(*build_equality_keys_view, *probe_equality_keys_view);
    return _perform_mark_join(*build_equality_keys_view,
                              is_anti_join,
                              *probe_equality_keys_view,
                              comparator_adapter,
                              shared_memory_per_thread,
                              stream,
                              mr);
  } else {
    auto build_conditional_keys_view = cudf::table_device_view::create(left_conditional, stream);
    auto probe_conditional_keys_view = cudf::table_device_view::create(right_conditional, stream);
    auto const has_nulls =
      binary_predicate->may_evaluate_null(left_conditional, right_conditional, stream);
    auto const parser = cudf::ast::detail::expression_parser{
      *binary_predicate, left_conditional, right_conditional, has_nulls, stream, mr};
    shared_memory_per_thread = parser.shmem_per_thread;
    if (has_nulls) {
      auto comparator_adapter = mixed_comparator_adapter<true>(*build_equality_keys_view,
                                                               *probe_equality_keys_view,
                                                               *build_conditional_keys_view,
                                                               *probe_conditional_keys_view,
                                                               parser.device_expression_data);
      return _perform_mark_join(*build_equality_keys_view,
                                is_anti_join,
                                *probe_equality_keys_view,
                                comparator_adapter,
                                shared_memory_per_thread,
                                stream,
                                mr);
    } else {
      auto comparator_adapter = mixed_comparator_adapter<false>(*build_equality_keys_view,
                                                                *probe_equality_keys_view,
                                                                *build_conditional_keys_view,
                                                                *probe_conditional_keys_view,
                                                                parser.device_expression_data);
      return _perform_mark_join(*build_equality_keys_view,
                                is_anti_join,
                                *probe_equality_keys_view,
                                comparator_adapter,
                                shared_memory_per_thread,
                                stream,
                                mr);
    }
  }
}

template <int32_t block_size,
          typename mark_set_key_type,
          typename comparator_adapter_type,
          typename mark_set_ref_type,
          typename input_it_type>
static auto get_mark_probe_kernel(bool is_low_selectivity)
{
  if (is_low_selectivity) {
    return mark_probe<block_size,
                      mark_set_key_type,
                      comparator_adapter_type,
                      mark_set_ref_type,
                      input_it_type,
                      true>;
  } else {
    return mark_probe<block_size,
                      mark_set_key_type,
                      comparator_adapter_type,
                      mark_set_ref_type,
                      input_it_type,
                      false>;
  }
}

bool mark_join::is_low_selectivity() const
{
  // Current heuristic depends on if we are using hash map caching.
  return !_is_cached;
}

template <typename comparator_adapter_type>
std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join::_perform_mark_join(cudf::table_device_view const& build_equality_keys_view,
                              bool is_anti_join,
                              cudf::table_device_view const& probe_equality_keys_view,
                              comparator_adapter_type const& comparator_adapter,
                              uint32_t shared_memory_per_thread,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr) const
{
  constexpr int block_size          = 128;
  const int shared_memory_per_block = shared_memory_per_thread * block_size;

  using build_hasher_type   = device_row_hasher<cudf::hashing::detail::default_hash>;
  auto const d_probe_hasher = build_hasher_type{probe_equality_keys_view};
  const create_input_pair<build_hasher_type, cudf::experimental::row::lhs_index_type>
    probe_input_pair{d_probe_hasher};

  auto const probe_iter   = cudf::detail::make_counting_transform_iterator(0, probe_input_pair);
  auto mark_set_ref       = _mark_set.ref(cuco::op::contains_tag{});
  using mark_set_key_type = typename decltype(mark_set_ref)::key_type;

  auto mark_probe_kernel = get_mark_probe_kernel<block_size,
                                                 mark_set_key_type,
                                                 comparator_adapter_type,
                                                 decltype(mark_set_ref),
                                                 decltype(probe_iter)>(this->is_low_selectivity());
  int probe_grid_size =
    utility::detect_launch_grid_size(mark_probe_kernel, block_size, shared_memory_per_block);

  rmm::device_scalar<cudf::size_type> mark_counter(0, stream, mr);
  mark_probe_kernel<<<probe_grid_size, block_size, shared_memory_per_block, stream>>>(
    mark_set_ref,
    comparator_adapter,
    probe_iter,
    probe_equality_keys_view.num_rows(),
    mark_counter.data());
  cudf::size_type marked_row_count = mark_counter.value(stream);

  _num_marks += marked_row_count;

  // return ptr.
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> build_positions;
  // a pair is expected even though we only have one set of positions
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> left;
  if (_is_cached) {
    // if it's cached we don't instantiate the positions yet
    build_positions = std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  } else {
    // not cached - we need to do a scan and materialize the positions
    cudf::size_type result_row_count =
      is_anti_join ? build_equality_keys_view.num_rows() - marked_row_count : marked_row_count;

    rmm::device_uvector<cudf::size_type> positions(result_row_count, stream, mr);
    rmm::device_scalar<cudf::size_type> mark_scan_offset(0, stream, mr);

    auto mark_scan_kernel = mark_scan<block_size, mark_set_key_type, decltype(mark_set_ref)>;
    int scan_grid_size =
      utility::detect_launch_grid_size(mark_probe_kernel, block_size, shared_memory_per_block);

    mark_scan_kernel<<<scan_grid_size, block_size, 0, stream>>>(
      mark_set_ref, positions.data(), mark_scan_offset.data(), is_anti_join);

    GQE_LOG_TRACE("mark join scan offset={}", mark_scan_offset.value(stream));

    build_positions = std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(positions));
  }
  return std::make_pair(std::move(left), std::move(build_positions));
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>>
mark_join::_compute_positions_list_from_cached_map(bool is_anti_join,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
{
  constexpr int block_size         = 128;
  auto mark_set_ref                = _mark_set.ref(cuco::op::contains_tag{});
  using mark_set_key_type          = typename decltype(mark_set_ref)::key_type;
  cudf::size_type result_row_count = is_anti_join ? _build.num_rows() - _num_marks : _num_marks;

  rmm::device_uvector<cudf::size_type> build_positions(result_row_count, stream, mr);
  rmm::device_scalar<cudf::size_type> mark_scan_offset(0, stream, mr);

  auto mark_scan_kernel = mark_scan<block_size, mark_set_key_type, decltype(mark_set_ref)>;
  int scan_grid_size    = utility::detect_launch_grid_size(mark_scan_kernel, block_size);

  mark_scan_kernel<<<scan_grid_size, block_size, 0, stream>>>(
    mark_set_ref, build_positions.data(), mark_scan_offset.data(), is_anti_join);

  GQE_LOG_TRACE("cached mark join scan offset={}", mark_scan_offset.value(stream));

  return std::make_unique<rmm::device_uvector<cudf::size_type>>(std::move(build_positions));
}

}  // namespace detail

mark_join::~mark_join() = default;

mark_join::mark_join(cudf::table_view const& build,
                     bool is_cached,
                     cudf::null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  _impl =
    std::make_unique<detail::mark_join>(build, is_cached, compare_nulls, load_factor, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
mark_join::perform_mark_join(cudf::table_view const& probe,
                             bool is_anti_join,
                             cudf::table_view const& left_conditional,
                             cudf::table_view const& right_conditional,
                             cudf::ast::expression const* binary_predicate,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  return _impl->perform_mark_join(
    probe, is_anti_join, left_conditional, right_conditional, binary_predicate, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>>
mark_join::compute_positions_list_from_cached_map(bool is_anti_join,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr) const
{
  return _impl->_compute_positions_list_from_cached_map(is_anti_join);
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
  // there is no AST to evaluate for equality join
  constexpr cudf::ast::expression const* binary_predicate = nullptr;
  constexpr bool is_cached                                = false;
  auto build_equality_keys_view = cudf::table_device_view::create(left_equality, stream);
  auto probe_equality_keys_view = cudf::table_device_view::create(right_equality, stream);
  auto join_obj  = gqe::mark_join(left_equality, is_cached, compare_nulls, load_factor, stream, mr);
  auto positions = join_obj.perform_mark_join(right_equality,
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
  constexpr bool is_cached = false;
  auto join_obj  = gqe::mark_join(left_equality, is_cached, compare_nulls, load_factor, stream, mr);
  auto positions = join_obj.perform_mark_join(right_equality,
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
