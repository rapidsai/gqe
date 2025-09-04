/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "post_processing.cuh"
#include "row_aggregator.cuh"

#include <gqe/device_properties.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cuco/static_set.cuh>

#include <cudf/detail/aggregation/device_aggregators.cuh>

#include <cudf/reduction.hpp>

#include <rmm/device_scalar.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>

#include <cmath>
#include <cooperative_groups.h>
#include <iostream>

namespace gqe {
namespace groupby {

namespace hash {

int constexpr cg_size     = 1;  ///< Number of threads used to handle each input key
using probing_scheme_type = cuco::linear_probing<
  cg_size,
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC>>;

__device__ __host__ size_t round_to_multiple_of_8(size_t num)
{
  size_t constexpr multiple_of = 8;
  return gqe::utility::divide_round_up(num, multiple_of) * multiple_of;
}

size_t get_previous_multiple_of_8(size_t number) { return number / 8 * 8; }

template <typename SetType>
__device__ void find_local_mapping(cudf::size_type cur_idx,
                                   cudf::size_type num_input_rows,
                                   cudf::size_type* cardinality,
                                   SetType shared_set,
                                   cudf::size_type* local_mapping_index,
                                   cudf::size_type* shared_set_indices,
                                   thrust::optional<cudf::column_device_view> active_mask)
{
  cudf::size_type result_idx;
  bool row_inserted = false;
  bool row_valid    = (cur_idx < num_input_rows);
  bool row_active   = active_mask ? active_mask->element<bool>(cur_idx) : true;

  if (row_valid && row_active) {
    auto const result = shared_set.insert_and_find(cur_idx);
    result_idx        = *result.first;
    row_inserted      = result.second;

    // inserted a new element
    if (result.second) {
      auto shared_set_index                = atomicAdd(cardinality, 1);
      shared_set_indices[shared_set_index] = cur_idx;
      local_mapping_index[cur_idx]         = shared_set_index;
    }
  }

  else if (row_valid && !row_active) {
    local_mapping_index[cur_idx] = -1;
  }

  // Syncing the thread block is needed so that updates in `local_mapping_index` are visible to all
  // threads in the thread block.
  __syncthreads();

  if (row_valid && row_active && !row_inserted) {
    // element was already in set
    local_mapping_index[cur_idx] = local_mapping_index[result_idx];
  }
}

template <typename SetType>
__device__ void find_global_mapping(cudf::size_type cur_idx,
                                    SetType global_set,
                                    cudf::size_type* shared_set_indices,
                                    cudf::size_type* global_mapping_index,
                                    cudf::size_type shared_set_num_elements)
{
  auto input_idx = shared_set_indices[cur_idx];
  auto result    = global_set.insert_and_find(input_idx);
  global_mapping_index[blockIdx.x * shared_set_num_elements + cur_idx] = *result.first;
}

/*
 * Inserts keys into the shared memory hash set, and stores the row index of the local
 * pre-aggregate table in `local_mapping_index`. If the number of unique keys found in a
 * threadblock exceeds `cardinality_threshold`, the threads in that block will exit without
 * updating `global_set` or setting `global_mapping_index`. Else, we insert the unique keys found
 * to the global hash set, and save the row index of the global sparse table in
 * `global_mapping_index`.
 */
template <class SetRef,
          cudf::size_type shared_set_num_elements,
          cudf::size_type cardinality_threshold,
          typename GlobalSetType,
          typename KeyEqual,
          typename RowHasher,
          typename BucketExtent>
__global__ void compute_mapping_indices(GlobalSetType global_set,
                                        cudf::size_type num_input_rows,
                                        BucketExtent bucket_extent,
                                        cuco::empty_key<cudf::size_type> empty_key_sentinel,
                                        KeyEqual d_key_equal,
                                        RowHasher d_row_hash,
                                        cudf::size_type* local_mapping_index,
                                        cudf::size_type* global_mapping_index,
                                        cudf::size_type* block_cardinality,
                                        bool* direct_aggregations,
                                        thrust::optional<cudf::column_device_view> active_mask)
{
  __shared__ cudf::size_type shared_set_indices[shared_set_num_elements];

  // Shared set initialization
  __shared__ typename SetRef::bucket_type buckets[bucket_extent.value()];
  auto storage = SetRef::storage_ref_type(bucket_extent, buckets);
  auto shared_set =
    SetRef(empty_key_sentinel, d_key_equal, probing_scheme_type{d_row_hash}, {}, storage);
  auto const block = cooperative_groups::this_thread_block();
  shared_set.initialize(block);
  block.sync();

  auto shared_insert_ref = std::move(shared_set).rebind_operators(cuco::insert_and_find);

  __shared__ cudf::size_type cardinality;

  if (threadIdx.x == 0) { cardinality = 0; }

  __syncthreads();

  int num_loops =
    gqe::utility::divide_round_up(num_input_rows, (cudf::size_type)(blockDim.x * gridDim.x));
  auto end_idx = num_loops * blockDim.x * gridDim.x;

  for (auto cur_idx = blockDim.x * blockIdx.x + threadIdx.x; cur_idx < end_idx;
       cur_idx += blockDim.x * gridDim.x) {
    find_local_mapping(cur_idx,
                       num_input_rows,
                       &cardinality,
                       shared_insert_ref,
                       local_mapping_index,
                       shared_set_indices,
                       active_mask);

    __syncthreads();

    if (cardinality >= cardinality_threshold) {
      if (threadIdx.x == 0) { *direct_aggregations = true; }
      break;
    }

    __syncthreads();
  }

  // Insert unique keys from shared to global hash set
  if (cardinality < cardinality_threshold) {
    for (auto cur_idx = threadIdx.x; cur_idx < cardinality; cur_idx += blockDim.x) {
      find_global_mapping(
        cur_idx, global_set, shared_set_indices, global_mapping_index, shared_set_num_elements);
    }
  }

  if (threadIdx.x == 0) block_cardinality[blockIdx.x] = cardinality;
}

__device__ void calculate_columns_to_aggregate(int& col_start,
                                               int& col_end,
                                               cudf::mutable_table_device_view output_values,
                                               int num_input_cols,
                                               std::byte** s_aggregates_pointer,
                                               bool** s_aggregates_valid_pointer,
                                               std::byte* shared_set_aggregates,
                                               cudf::size_type cardinality,
                                               int total_agg_size)
{
  if (threadIdx.x == 0) {
    col_start           = col_end;
    int bytes_allocated = 0;
    int valid_col_size  = round_to_multiple_of_8(sizeof(bool) * cardinality);
    while ((bytes_allocated < total_agg_size) && (col_end < num_input_cols)) {
      int next_col_size =
        round_to_multiple_of_8(sizeof(output_values.column(col_end).type()) * cardinality);
      int next_col_total_size = valid_col_size + next_col_size;

      if (bytes_allocated + next_col_total_size > total_agg_size) { break; }

      s_aggregates_pointer[col_end] = shared_set_aggregates + bytes_allocated;
      s_aggregates_valid_pointer[col_end] =
        reinterpret_cast<bool*>(shared_set_aggregates + bytes_allocated + next_col_size);

      bytes_allocated += next_col_total_size;
      col_end++;
    }
  }
}

__device__ void initialize_shared_memory_aggregates(int col_start,
                                                    int col_end,
                                                    cudf::mutable_table_device_view output_values,
                                                    std::byte** s_aggregates_pointer,
                                                    bool** s_aggregates_valid_pointer,
                                                    cudf::size_type cardinality,
                                                    cudf::aggregation::Kind const* aggs)
{
  for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
    for (auto idx = threadIdx.x; idx < cardinality; idx += blockDim.x) {
      cudf::detail::dispatch_type_and_aggregation(output_values.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  gqe::initialize_shmem{},
                                                  s_aggregates_pointer[col_idx],
                                                  idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

__device__ void compute_pre_aggregrates(int col_start,
                                        int col_end,
                                        cudf::table_device_view input_values,
                                        cudf::size_type num_input_rows,
                                        cudf::size_type* local_mapping_index,
                                        std::byte** s_aggregates_pointer,
                                        bool** s_aggregates_valid_pointer,
                                        cudf::aggregation::Kind const* aggs,
                                        cudf::size_type agg_location_offset)
{
  for (cudf::size_type block_start_idx = blockDim.x * blockIdx.x; block_start_idx < num_input_rows;
       block_start_idx += blockDim.x * gridDim.x) {
    auto cur_idx = block_start_idx + threadIdx.x;

    // Synchronization is necessary to ensure coalesced global memory accesses
    __syncthreads();

    if (cur_idx >= num_input_rows) { continue; }
    if (local_mapping_index[cur_idx] == -1) { continue; }

    auto map_idx = agg_location_offset + local_mapping_index[cur_idx];

    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto input_col = input_values.column(col_idx);

      cudf::detail::dispatch_type_and_aggregation(input_col.type(),
                                                  aggs[col_idx],
                                                  gqe::shmem_element_aggregator{},
                                                  s_aggregates_pointer[col_idx],
                                                  map_idx,
                                                  s_aggregates_valid_pointer[col_idx],
                                                  input_col,
                                                  cur_idx);
    }
  }
}

template <int shared_set_num_elements>
__device__ void compute_final_aggregates(int col_start,
                                         int col_end,
                                         cudf::table_device_view input_values,
                                         cudf::mutable_table_device_view output_values,
                                         cudf::size_type cardinality,
                                         cudf::size_type num_agg_locations,
                                         cudf::size_type* global_mapping_index,
                                         std::byte** s_aggregates_pointer,
                                         bool** s_aggregates_valid_pointer,
                                         cudf::aggregation::Kind const* aggs)
{
  for (auto cur_idx = threadIdx.x; cur_idx < num_agg_locations; cur_idx += blockDim.x) {
    auto out_idx =
      global_mapping_index[blockIdx.x * shared_set_num_elements + cur_idx % cardinality];
    for (auto col_idx = col_start; col_idx < col_end; col_idx++) {
      auto output_col = output_values.column(col_idx);

      cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  gqe::gmem_element_aggregator{},
                                                  output_col,
                                                  out_idx,
                                                  input_values.column(col_idx),
                                                  s_aggregates_pointer[col_idx],
                                                  cur_idx,
                                                  s_aggregates_valid_pointer[col_idx]);
    }
  }
}

/* Takes the local_mapping_index and global_mapping_index to compute
 * pre (shared) and final (global) aggregates*/
template <cudf::size_type shared_set_num_elements, cudf::size_type cardinality_threshold>
__global__ void compute_aggregates(cudf::size_type* local_mapping_index,
                                   cudf::size_type* global_mapping_index,
                                   cudf::size_type* block_cardinality,
                                   cudf::table_device_view input_values,
                                   cudf::mutable_table_device_view output_values,
                                   cudf::size_type num_input_rows,
                                   cudf::aggregation::Kind const* aggs,
                                   int total_agg_size,
                                   int pointer_size,
                                   cudf::size_type min_shmem_agg_locations)
{
  cudf::size_type cardinality = block_cardinality[blockIdx.x];
  if (cardinality >= cardinality_threshold || cardinality == 0) { return; }

  cudf::size_type multiplication_factor = min_shmem_agg_locations / cardinality;
  cudf::size_type num_agg_locations =
    multiplication_factor > 1 ? multiplication_factor * cardinality : cardinality;
  cudf::size_type const agg_location_offset =
    multiplication_factor > 1 ? (threadIdx.x % multiplication_factor) * cardinality : 0;

  int num_input_cols = output_values.num_columns();
  extern __shared__ std::byte shared_set_aggregates[];
  std::byte** s_aggregates_pointer =
    reinterpret_cast<std::byte**>(shared_set_aggregates + total_agg_size);
  bool** s_aggregates_valid_pointer =
    reinterpret_cast<bool**>(shared_set_aggregates + total_agg_size + pointer_size);

  __shared__ int col_start;
  __shared__ int col_end;
  if (threadIdx.x == 0) {
    col_start = 0;
    col_end   = 0;
  }
  __syncthreads();

  while (col_end < num_input_cols) {
    // We need all the threads to enter the loop,
    // before thread 0 updates col_end in calculate_columns_to_aggregate
    __syncthreads();

    calculate_columns_to_aggregate(col_start,
                                   col_end,
                                   output_values,
                                   num_input_cols,
                                   s_aggregates_pointer,
                                   s_aggregates_valid_pointer,
                                   shared_set_aggregates,
                                   num_agg_locations,
                                   total_agg_size);
    __syncthreads();

    initialize_shared_memory_aggregates(col_start,
                                        col_end,
                                        output_values,
                                        s_aggregates_pointer,
                                        s_aggregates_valid_pointer,
                                        num_agg_locations,
                                        aggs);
    __syncthreads();

    compute_pre_aggregrates(col_start,
                            col_end,
                            input_values,
                            num_input_rows,
                            local_mapping_index,
                            s_aggregates_pointer,
                            s_aggregates_valid_pointer,
                            aggs,
                            agg_location_offset);
    __syncthreads();

    compute_final_aggregates<shared_set_num_elements>(col_start,
                                                      col_end,
                                                      input_values,
                                                      output_values,
                                                      cardinality,
                                                      num_agg_locations,
                                                      global_mapping_index,
                                                      s_aggregates_pointer,
                                                      s_aggregates_valid_pointer,
                                                      aggs);
  }
}

template <typename SetType>
struct compute_direct_aggregates {
  SetType set;
  cudf::table_device_view input_values;
  cudf::mutable_table_device_view output_values;
  cudf::aggregation::Kind const* __restrict__ aggs;
  cudf::size_type* block_cardinality;
  int stride;
  int block_size;
  cudf::size_type cardinality_threshold;
  thrust::optional<cudf::column_device_view> active_mask;

  compute_direct_aggregates(SetType set,
                            cudf::table_device_view input_values,
                            cudf::mutable_table_device_view output_values,
                            cudf::aggregation::Kind const* aggs,
                            cudf::size_type* block_cardinality,
                            int stride,
                            int block_size,
                            cudf::size_type cardinality_threshold,
                            thrust::optional<cudf::column_device_view> active_mask)
    : set(set),
      input_values(input_values),
      output_values(output_values),
      aggs(aggs),
      block_cardinality(block_cardinality),
      stride(stride),
      block_size(block_size),
      cardinality_threshold(cardinality_threshold),
      active_mask(active_mask)
  {
  }

  __device__ void operator()(cudf::size_type i)
  {
    int block_id    = (i % stride) / block_size;
    bool row_active = !active_mask || active_mask->element<bool>(i);
    if (row_active && (block_cardinality[block_id] >= cardinality_threshold)) {
      auto const result = set.insert_and_find(i);
      cudf::detail::aggregate_row(output_values, *result.first, input_values, i, aggs);
    }
  }
};

template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<cudf::size_type>& populated_keys,
                            rmm::cuda_stream_view stream)
{
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.value());

  populated_keys.resize(std::distance(populated_keys.begin(), keys_end), stream);
}

struct initialize_sparse_table {
  cudf::size_type const* row_indices;
  cudf::mutable_table_device_view sparse_table;
  cudf::aggregation::Kind const* __restrict__ aggs;

  initialize_sparse_table(cudf::size_type const* row_indices,
                          cudf::mutable_table_device_view sparse_table,
                          cudf::aggregation::Kind const* aggs)
    : row_indices(row_indices), sparse_table(sparse_table), aggs(aggs)
  {
  }

  __device__ void operator()(cudf::size_type i)
  {
    auto key_idx = row_indices[i];
    for (auto col_idx = 0; col_idx < sparse_table.num_columns(); col_idx++) {
      cudf::detail::dispatch_type_and_aggregation(sparse_table.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  gqe::initialize_gmem{},
                                                  sparse_table.column(col_idx),
                                                  key_idx);
    }
  }
};

// TODO: copied from cudf::detail::initialize_with_identity
void initialize_with_identity(cudf::mutable_table_view& table,
                              std::vector<cudf::aggregation::Kind> const& aggs,
                              rmm::cuda_stream_view stream)
{
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto col = table.column(i);
    cudf::detail::dispatch_type_and_aggregation(
      col.type(), aggs[i], cudf::detail::identity_initializer{}, col, stream);
  }
}

template <typename GlobalSetType>
auto create_sparse_results_table(cudf::table_view const& flattened_values,
                                 const cudf::aggregation::Kind* d_aggs,
                                 std::vector<cudf::aggregation::Kind> aggs,
                                 bool direct_aggregations,
                                 GlobalSetType const& global_set,
                                 rmm::device_uvector<cudf::size_type>& populated_keys,
                                 rmm::cuda_stream_view stream)
{
  // TODO single allocation - room for performance improvement
  std::vector<std::unique_ptr<cudf::column>> sparse_columns;
  std::transform(
    flattened_values.begin(),
    flattened_values.end(),
    aggs.begin(),
    std::back_inserter(sparse_columns),
    [stream](auto const& col, auto const& agg) {
      bool nullable = (agg == cudf::aggregation::COUNT_VALID or agg == cudf::aggregation::COUNT_ALL)
                        ? false
                        : (col.has_nulls() or agg == cudf::aggregation::VARIANCE or
                           agg == cudf::aggregation::STD);
      auto mask_flag = (nullable) ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED;

      auto col_type = cudf::is_dictionary(col.type())
                        ? cudf::dictionary_column_view(col).keys().type()
                        : col.type();

      auto target_type = cudf::detail::target_type(col_type, agg);
      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
    });

  cudf::table sparse_table(std::move(sparse_columns));

  // If no direct aggregations, initialize the sparse table
  // only for the keys inserted in global hash set
  if (!direct_aggregations) {
    auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);
    extract_populated_keys(global_set, populated_keys, stream);
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       populated_keys.size(),
                       initialize_sparse_table{populated_keys.data(), *d_sparse_table, d_aggs});
  }

  // Else initialise the whole table
  else {
    cudf::mutable_table_view sparse_table_view = sparse_table.mutable_view();
    gqe::groupby::hash::initialize_with_identity(sparse_table_view, aggs, stream);
  }

  return sparse_table;
}

template <typename FuncType>
size_t find_shmem_size(FuncType func,
                       int block_size,
                       int grid_size,
                       gqe::device_properties const& device_properties)
{
  auto device_id = utility::current_cuda_device_id();
  int num_sms    = device_properties.get<device_properties::multiProcessorCount>(device_id);
  auto active_blocks_per_sm = gqe::utility::divide_round_up(grid_size, num_sms);

  size_t dynamic_smem_size;
  GQE_CUDA_TRY(cudaOccupancyAvailableDynamicSMemPerBlock(
    &dynamic_smem_size, func, active_blocks_per_sm, block_size));
  return get_previous_multiple_of_8(0.5 * dynamic_smem_size);
}

template <cudf::size_type shared_set_num_elements, cudf::size_type cardinality_threshold>
void launch_compute_aggregates(int block_size,
                               int grid_size,
                               cudf::size_type* local_mapping_index,
                               cudf::size_type* global_mapping_index,
                               cudf::size_type* block_cardinality,
                               cudf::table_device_view input_values,
                               cudf::mutable_table_device_view output_values,
                               cudf::size_type num_input_rows,
                               cudf::aggregation::Kind const* aggs,
                               gqe::device_properties const& device_properties,
                               rmm::cuda_stream_view stream)
{
  auto compute_aggregates_fn_ptr =
    compute_aggregates<shared_set_num_elements, cardinality_threshold>;
  size_t d_shmem_size =
    find_shmem_size(compute_aggregates_fn_ptr, block_size, grid_size, device_properties);

  // For each aggregation, need two pointers to arrays in shmem
  // One where the aggregation is performed, one indicating the validity of the aggregation
  auto shmem_agg_pointer_size =
    round_to_multiple_of_8(sizeof(std::byte*) * output_values.num_columns());
  // The rest of shmem is utilized for the actual arrays in shmem
  auto shmem_agg_size = d_shmem_size - shmem_agg_pointer_size * 2;

  // Determined through benchmarking experiments
  // Minimum shared memory locations for aggregation to avoid high shmem atomic contention
  cudf::size_type constexpr min_shmem_agg_locations = 32;

  compute_aggregates<shared_set_num_elements, cardinality_threshold>
    <<<grid_size, block_size, d_shmem_size, stream>>>(local_mapping_index,
                                                      global_mapping_index,
                                                      block_cardinality,
                                                      input_values,
                                                      output_values,
                                                      num_input_rows,
                                                      aggs,
                                                      shmem_agg_size,
                                                      shmem_agg_pointer_size,
                                                      min_shmem_agg_locations);
}

template <typename SetType, typename KeyEqual, typename RowHasher>
rmm::device_uvector<cudf::size_type> compute_single_pass_set_aggs(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  SetType& global_set,
  rmm::cuda_stream_view stream,
  cuco::empty_key<cudf::size_type> empty_key_sentinel,
  KeyEqual d_key_equal,
  RowHasher d_row_hash,
  cudf::column_view const& active_mask,
  gqe::device_properties const& device_properties)
{
  auto const [flattened_values, agg_kinds, aggs] = gqe::flatten_single_pass_aggs(requests);

  auto const num_input_rows = keys.num_rows();
  auto d_values             = cudf::table_device_view::create(flattened_values, stream);
  auto const d_aggs         = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, rmm::mr::get_current_device_resource());

  auto constexpr bucket_size                      = 1;
  constexpr int block_size                        = 128;
  constexpr cudf::size_type cardinality_threshold = 128;

  // We add additional `block_size`, because after the number of elements in the local hash set
  // exceeds the threshold, all threads in the thread block can still insert one more element.
  constexpr cudf::size_type shared_set_num_elements = cardinality_threshold + block_size;
  constexpr float load_factor                       = 0.7;
  constexpr std::size_t shared_set_size             = (1.0 / load_factor) * shared_set_num_elements;

  using extent_type = cuco::extent<cudf::size_type, shared_set_size>;
  using shared_set_type =
    cuco::static_set<cudf::size_type,
                     extent_type,
                     cuda::thread_scope_block,
                     cudf::experimental::row::equality::device_row_comparator<
                       false,
                       cudf::nullate::DYNAMIC,
                       cudf::experimental::row::equality::nan_equal_physical_equality_comparator>,
                     probing_scheme_type,
                     cuco::cuda_allocator<cudf::size_type>,
                     cuco::storage<bucket_size>>;
  using shared_set_ref_type    = typename shared_set_type::ref_type<>;
  auto constexpr bucket_extent = cuco::make_bucket_extent<shared_set_ref_type>(extent_type{});

  auto global_set_ref = global_set.ref(cuco::op::insert_and_find);

  auto compute_mapping_indices_fn_ptr = compute_mapping_indices<shared_set_ref_type,
                                                                shared_set_num_elements,
                                                                cardinality_threshold,
                                                                decltype(global_set_ref),
                                                                KeyEqual,
                                                                RowHasher,
                                                                decltype(bucket_extent)>;
  int grid_size =
    utility::detect_launch_grid_size(device_properties, compute_mapping_indices_fn_ptr, block_size);
  int needed_active_blocks = gqe::utility::divide_round_up(num_input_rows, block_size);
  grid_size                = std::min(grid_size, needed_active_blocks);

  // 'local_mapping_index' maps from the global row index of the input table to the row index of
  // the local pre-aggregate table
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_input_rows, stream);

  // 'global_mapping_index' maps from  the local pre-aggregate table to the row index of
  // global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(grid_size * shared_set_num_elements,
                                                            stream);
  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream);

  rmm::device_scalar<bool> direct_aggregations(false, stream);

  thrust::optional<cudf::column_device_view> active_mask_device_view =
    active_mask.is_empty()
      ? thrust::nullopt
      : thrust::optional<cudf::column_device_view>(*cudf::column_device_view::create(active_mask));

  compute_mapping_indices<shared_set_ref_type, shared_set_num_elements, cardinality_threshold>
    <<<grid_size, block_size, 0, stream>>>(global_set_ref,
                                           num_input_rows,
                                           bucket_extent,
                                           empty_key_sentinel,
                                           d_key_equal,
                                           d_row_hash,
                                           local_mapping_index.data(),
                                           global_mapping_index.data(),
                                           block_cardinality.data(),
                                           direct_aggregations.data(),
                                           active_mask_device_view);

  stream.synchronize();

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(keys.num_rows(), stream);

  cudf::table sparse_table = create_sparse_results_table(flattened_values,
                                                         d_aggs.data(),
                                                         agg_kinds,
                                                         direct_aggregations.value(stream),
                                                         global_set,
                                                         populated_keys,
                                                         stream);

  auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);

  launch_compute_aggregates<shared_set_num_elements, cardinality_threshold>(
    block_size,
    grid_size,
    local_mapping_index.data(),
    global_mapping_index.data(),
    block_cardinality.data(),
    *d_values,
    *d_sparse_table,
    num_input_rows,
    d_aggs.data(),
    device_properties,
    stream);

  if (direct_aggregations.value(stream)) {
    int stride = block_size * grid_size;
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator(0),
                       keys.num_rows(),
                       compute_direct_aggregates{global_set_ref,
                                                 *d_values,
                                                 *d_sparse_table,
                                                 d_aggs.data(),
                                                 block_cardinality.data(),
                                                 stride,
                                                 block_size,
                                                 cardinality_threshold,
                                                 active_mask_device_view});
    extract_populated_keys(global_set, populated_keys, stream);
  }

  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }

  return populated_keys;
}

int64_t get_num_active_keys(cudf::table_view const& keys, cudf::column_view const& active_mask)
{
  if (!active_mask.is_empty()) {
    auto total_active_keys = cudf::reduce(active_mask,
                                          *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                                          cudf::data_type{cudf::type_id::INT64});

    return static_cast<cudf::numeric_scalar<int64_t>*>(total_active_keys.get())->value();
  } else {
    auto const num_keys = keys.num_rows();
    return static_cast<int64_t>(num_keys);
  }
}

rmm::device_uvector<cudf::size_type> groupby(
  cudf::detail::result_cache* dense_results,
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::column_view const& active_mask,
  gqe::device_properties const& device_properties,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto const null_keys_are_equal = cudf::null_equality::EQUAL;
  auto const has_null            = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(keys)};

  rmm::mr::polymorphic_allocator<cudf::size_type> polly_alloc;
  auto stream_alloc = rmm::mr::stream_allocator_adaptor(polly_alloc, stream);

  auto preprocessed_keys = cudf::experimental::row::hash::preprocessed_table::create(keys, stream);
  auto const comparator  = cudf::experimental::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash    = cudf::experimental::row::hash::row_hasher{std::move(preprocessed_keys)};
  auto const d_row_hash  = row_hash.device_hasher(has_null);

  if (cudf::detail::has_nested_columns(keys)) {
    throw std::runtime_error("Keys with nested columns are not supported");
  }
  auto const d_key_equal = comparator.equal_to<false>(has_null, null_keys_are_equal);

  cudf::size_type constexpr key_sentinel = -1;  ///< Sentinel value indicating an empty slot
  auto empty_key_sentinel                = cuco::empty_key{key_sentinel};
  double load_factor                     = 0.5;

  auto const global_agg_set = cuco::static_set{get_num_active_keys(keys, active_mask),
                                               load_factor,
                                               empty_key_sentinel,
                                               d_key_equal,
                                               probing_scheme_type{d_row_hash},
                                               {},
                                               {},
                                               stream_alloc,
                                               stream.value()};
  cudf::detail::result_cache sparse_results(requests.size());

  auto gather_map = compute_single_pass_set_aggs(keys,
                                                 requests,
                                                 &sparse_results,
                                                 global_agg_set,
                                                 stream,
                                                 empty_key_sentinel,
                                                 d_key_equal,
                                                 d_row_hash,
                                                 active_mask,
                                                 device_properties);

  gqe::sparse_to_dense_results(keys,
                               requests,
                               &sparse_results,
                               dense_results,
                               gather_map,
                               global_agg_set.ref(cuco::find),
                               0,
                               cudf::null_policy::INCLUDE,
                               stream,
                               mr);
  return gather_map;
}

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::groupby::aggregation_result>> groupby(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::column_view const& active_mask,
  gqe::device_properties const& device_properties,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  cudf::detail::result_cache dense_results(requests.size());

  auto gather_map =
    groupby(&dense_results, keys, requests, active_mask, device_properties, stream, mr);

  auto unique_keys = cudf::detail::gather(keys,
                                          gather_map,
                                          cudf::out_of_bounds_policy::DONT_CHECK,
                                          cudf::detail::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          mr);

  return std::pair(std::move(unique_keys),
                   gqe::extract_results(requests, dense_results, stream, mr));
}

}  // namespace hash
}  // namespace groupby
}  // namespace gqe
