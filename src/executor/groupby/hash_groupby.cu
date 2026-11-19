/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "hash_groupby.hpp"

#include "gather.hpp"
#include "post_processing.cuh"
#include "result_cache.hpp"
#include "row_aggregator.cuh"

#include <gqe/aggregation.hpp>
#include <gqe/device_properties.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/kernel_fusion_helpers.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/optional>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/iterator/counting_iterator.h>

#include <cooperative_groups.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace gqe {
namespace groupby {

namespace hash {

int constexpr cg_size     = 1;  ///< Number of threads used to handle each input key
using probing_scheme_type = cuco::linear_probing<
  cg_size,
  cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                             cudf::nullate::DYNAMIC>>;

__device__ __host__ size_t round_up_to_alignment(size_t num, size_t alignment)
{
  return gqe::utility::divide_round_up(num, alignment) * alignment;
}

size_t get_previous_multiple_of_8(size_t number) { return number / 8 * 8; }

// Element storage width from runtime type_id. Do not use sizeof(cudf::data_type) — that is
// type metadata (id + scale), not column payload size. cudf::size_of() is host-only.
__device__ __host__ size_t fixed_width_size(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::UINT8:
    case cudf::type_id::BOOL8: return 1;
    case cudf::type_id::INT16:
    case cudf::type_id::UINT16: return 2;
    case cudf::type_id::INT32:
    case cudf::type_id::UINT32:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DECIMAL32: return 4;
    case cudf::type_id::INT64:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
    case cudf::type_id::DECIMAL64: return 8;
    case cudf::type_id::DECIMAL128: return 16;
    default:
      // Unknown fixed-width runtime type: fail loudly in debug, and return a sentinel in
      // release so downstream size computations do not silently proceed with an incorrect width.
      assert(false && "Unsupported fixed-width cudf::type_id");
      return std::numeric_limits<size_t>::max();
  }
}

__device__ __host__ size_t shared_agg_alignment(cudf::data_type type)
{
  // Keep the default 8-byte alignment used by the existing kernels while
  // honoring 16-byte alignment required by DECIMAL128 payloads.
  auto const element_size = fixed_width_size(type.id());
  return element_size >= 16 ? 16 : 8;
}

template <typename SetType>
__device__ void find_local_mapping(cudf::size_type cur_idx,
                                   cudf::size_type num_input_rows,
                                   cudf::size_type* cardinality,
                                   SetType shared_set,
                                   cudf::size_type* local_mapping_index,
                                   cudf::size_type* shared_set_indices,
                                   cuda::std::optional<cudf::column_device_view> active_mask)
{
  cudf::size_type result_idx;
  bool row_inserted = false;
  bool row_valid    = (cur_idx < num_input_rows);
  bool row_active   = row_valid && (!active_mask || active_mask->element<bool>(cur_idx));

  if (row_active) {
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

  if (row_active && !row_inserted) {
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
                                        cuda::std::optional<cudf::column_device_view> active_mask)
{
  __shared__ cudf::size_type shared_set_indices[shared_set_num_elements];

  // Shared set initialization
  using storage_ref_type = typename SetRef::storage_ref_type;
  __shared__
    typename SetRef::value_type buckets[bucket_extent.value() * storage_ref_type::bucket_size];
  auto storage = storage_ref_type(bucket_extent, buckets);
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
                                               cudf::size_type num_agg_locations,
                                               int total_agg_size)
{
  if (threadIdx.x == 0) {
    col_start           = col_end;
    int bytes_allocated = 0;
    while ((bytes_allocated < total_agg_size) && (col_end < num_input_cols)) {
      // Make sure the beginning of the next column aggregation is aligned
      // This padding is only added to the bool values since these preceed the next column values
      auto const value_type = output_values.column(col_end).type();
      auto const next_col_alignment =
        col_end != num_input_cols - 1
          ? shared_agg_alignment(output_values.column(col_end + 1).type())
          : 1;

      int const col_size       = fixed_width_size(value_type.id()) * num_agg_locations;
      int const valid_col_size = sizeof(bool) * num_agg_locations;
      int const total_col_size = col_size + valid_col_size;
      int const next_bytes_allocated =
        round_up_to_alignment(bytes_allocated + total_col_size, next_col_alignment);

      if (next_bytes_allocated > total_agg_size) { break; }

      s_aggregates_pointer[col_end] = shared_set_aggregates + bytes_allocated;
      s_aggregates_valid_pointer[col_end] =
        reinterpret_cast<bool*>(shared_set_aggregates + bytes_allocated + col_size);

      bytes_allocated = next_bytes_allocated;
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

__device__ void compute_pre_aggregates(int col_start,
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
  alignas(16) extern __shared__ std::byte
    shared_set_aggregates[];  // 16 byte alignment for use in calculate_columns_to_aggregate with
                              // potentially 16-byte type output columns
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

    compute_pre_aggregates(col_start,
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
  cuda::std::optional<cudf::column_device_view> active_mask;

  compute_direct_aggregates(SetType set,
                            cudf::table_device_view input_values,
                            cudf::mutable_table_device_view output_values,
                            cudf::aggregation::Kind const* aggs,
                            cudf::size_type* block_cardinality,
                            int stride,
                            int block_size,
                            cudf::size_type cardinality_threshold,
                            cuda::std::optional<cudf::column_device_view> active_mask)
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
      auto const result       = set.insert_and_find(i);
      auto const target_index = *result.first;
      for (auto col_idx = 0; col_idx < output_values.num_columns(); ++col_idx) {
        cudf::detail::dispatch_type_and_aggregation(input_values.column(col_idx).type(),
                                                    aggs[col_idx],
                                                    cudf::detail::element_aggregator{},
                                                    output_values.column(col_idx),
                                                    target_index,
                                                    input_values.column(col_idx),
                                                    i);
      }
    }
  }
};

template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<cudf::size_type>& populated_keys,
                            cuda::stream_ref stream)
{
  auto const keys_end = key_set.retrieve_all(populated_keys.begin(), stream.get());

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

struct initialize_sparse_table_direct {
  cudf::mutable_table_device_view sparse_table;
  cudf::aggregation::Kind const* __restrict__ aggs;

  initialize_sparse_table_direct(cudf::mutable_table_device_view sparse_table,
                                 cudf::aggregation::Kind const* aggs)
    : sparse_table(sparse_table), aggs(aggs)
  {
  }

  __device__ void operator()(cudf::size_type i)
  {
    for (auto col_idx = 0; col_idx < sparse_table.num_columns(); col_idx++) {
      cudf::detail::dispatch_type_and_aggregation(sparse_table.column(col_idx).type(),
                                                  aggs[col_idx],
                                                  gqe::initialize_gmem{},
                                                  sparse_table.column(col_idx),
                                                  i);
    }
  }
};

/**
 * @brief Allocate the sparse aggregation table and initialize the selected aggregation path.
 *
 * Low-cardinality aggregation extracts the populated slots from `global_set` into
 * `populated_keys` and initializes only those slots. Direct aggregation initializes every slot in
 * the table. All device work is ordered on `stream`, and all table allocations use `mr`.
 *
 * @tparam GlobalSetType Device hash-set type used to extract populated slots.
 * @param[in] flattened_values Aggregation input columns.
 * @param[in] d_aggs Device aggregation kinds corresponding to `flattened_values`.
 * @param[in] aggs Host aggregation kinds corresponding to `flattened_values`.
 * @param[in] direct_aggregations Whether to initialize the direct-aggregation path.
 * @param[in] global_set Global hash set populated by the mapping kernel.
 * @param[in,out] populated_keys Buffer resized to the populated slots for low-cardinality input.
 * @param[in] stream CUDA stream used for device work.
 * @param[in] mr Memory resource used for table allocations.
 * @return Sparse aggregation result table.
 */
template <typename GlobalSetType>
auto create_sparse_results_table(cudf::table_view const& flattened_values,
                                 const cudf::aggregation::Kind* d_aggs,
                                 std::vector<cudf::aggregation::Kind> aggs,
                                 bool direct_aggregations,
                                 GlobalSetType const& global_set,
                                 rmm::device_uvector<cudf::size_type>& populated_keys,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  // TODO single allocation - room for performance improvement
  std::vector<std::unique_ptr<cudf::column>> sparse_columns;
  std::transform(
    flattened_values.begin(),
    flattened_values.end(),
    aggs.begin(),
    std::back_inserter(sparse_columns),
    [stream, mr](auto const& col, auto const& agg) {
      bool nullable = (agg == cudf::aggregation::COUNT_VALID or agg == cudf::aggregation::COUNT_ALL)
                        ? false
                        : (col.has_nulls() or agg == cudf::aggregation::VARIANCE or
                           agg == cudf::aggregation::STD);
      auto mask_flag = (nullable) ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED;

      auto col_type = cudf::is_dictionary(col.type())
                        ? cudf::dictionary_column_view(col).keys().type()
                        : col.type();

      auto const target_type = gqe::detail::compute_target_type(col_type, agg);
      return make_fixed_width_column(target_type, col.size(), mask_flag, stream, mr);
    });

  cudf::table sparse_table(std::move(sparse_columns));

  // Initialize only the slots inserted into the global hash set for low-cardinality input.
  if (!direct_aggregations) {
    auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);
    extract_populated_keys(global_set, populated_keys, stream);
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       populated_keys.size(),
                       initialize_sparse_table{populated_keys.data(), *d_sparse_table, d_aggs});
  }

  // Otherwise, initialize the whole table.
  else {
    auto d_sparse_table = cudf::mutable_table_device_view::create(sparse_table, stream);
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
                       thrust::make_counting_iterator(0),
                       sparse_table.num_rows(),
                       initialize_sparse_table_direct{*d_sparse_table, d_aggs});
  }

  return sparse_table;
}

template <typename FuncType>
size_t find_shmem_size(FuncType func, int block_size, int grid_size)
{
  auto device_id = utility::current_cuda_device_id();
  int num_sms =
    device_properties::instance().get<device_properties::multiProcessorCount>(device_id);
  auto active_blocks_per_sm = gqe::utility::divide_round_up(grid_size, num_sms);

  // Get max blocks we can schedule assuming 0 KB of shared memory usage
  int max_active_blocks_per_sm = 0;
  GQE_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, func, block_size, 0));

  // Number of active blocks on an SM can be atmost the minimum of blocks we want to schedule
  // and blocks that "can" be scheudled by the hardware.
  active_blocks_per_sm = std::min(active_blocks_per_sm, max_active_blocks_per_sm);

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
                               cuda::stream_ref stream)
{
  auto compute_aggregates_fn_ptr =
    compute_aggregates<shared_set_num_elements, cardinality_threshold>;
  size_t d_shmem_size = find_shmem_size(compute_aggregates_fn_ptr, block_size, grid_size);

  // For each aggregation, need two pointers to arrays in shmem
  // One where the aggregation is performed, one indicating the validity of the aggregation
  auto shmem_agg_pointer_size =
    round_up_to_alignment(sizeof(std::byte*) * output_values.num_columns(), 8);
  // The rest of shmem is utilized for the actual arrays in shmem
  auto shmem_agg_size = d_shmem_size - shmem_agg_pointer_size * 2;

  // Determined through benchmarking experiments
  // Minimum shared memory locations for aggregation to avoid high shmem atomic contention
  cudf::size_type constexpr min_shmem_agg_locations = 32;

  compute_aggregates<shared_set_num_elements, cardinality_threshold>
    <<<grid_size, block_size, d_shmem_size, stream.get()>>>(local_mapping_index,
                                                            global_mapping_index,
                                                            block_cardinality,
                                                            input_values,
                                                            output_values,
                                                            num_input_rows,
                                                            aggs,
                                                            shmem_agg_size,
                                                            shmem_agg_pointer_size,
                                                            min_shmem_agg_locations);
  GQE_CUDA_TRY(cudaPeekAtLastError());
}

template <typename SetType, typename KeyEqual, typename RowHasher>
rmm::device_uvector<cudf::size_type> compute_single_pass_set_aggs(
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  gqe::groupby::detail::result_cache* sparse_results,
  SetType& global_set,
  cuda::stream_ref stream,
  cuco::empty_key<cudf::size_type> empty_key_sentinel,
  KeyEqual d_key_equal,
  RowHasher d_row_hash,
  cudf::column_view const& active_mask,
  rmm::device_async_resource_ref mr)
{
  auto const [flattened_values, agg_kinds, aggs] = gqe::flatten_single_pass_aggs(requests);

  auto const num_input_rows = keys.num_rows();
  auto d_values             = cudf::table_device_view::create(flattened_values, stream);
  auto const d_aggs         = cudf::detail::make_device_uvector_async(agg_kinds, stream, mr);

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
                     cudf::detail::row::equality::device_row_comparator<
                       false,
                       cudf::nullate::DYNAMIC,
                       cudf::detail::row::equality::nan_equal_physical_equality_comparator>,
                     probing_scheme_type,
                     cuco::cuda_allocator<cudf::size_type>,
                     cuco::storage<bucket_size>>;
  using shared_set_ref_type = typename shared_set_type::ref_type<>;
  auto constexpr bucket_extent =
    cuco::make_valid_extent<probing_scheme_type, cuco::storage<bucket_size>>(extent_type{});

  auto global_set_ref = global_set.ref(cuco::op::insert_and_find);

  auto compute_mapping_indices_fn_ptr = compute_mapping_indices<shared_set_ref_type,
                                                                shared_set_num_elements,
                                                                cardinality_threshold,
                                                                decltype(global_set_ref),
                                                                KeyEqual,
                                                                RowHasher,
                                                                decltype(bucket_extent)>;
  int grid_size = utility::detect_launch_grid_size(compute_mapping_indices_fn_ptr, block_size);
  int needed_active_blocks = gqe::utility::divide_round_up(num_input_rows, block_size);
  grid_size                = std::min(grid_size, needed_active_blocks);

  // 'local_mapping_index' maps from the global row index of the input table to the row index of
  // the local pre-aggregate table
  rmm::device_uvector<cudf::size_type> local_mapping_index(num_input_rows, stream, mr);

  // 'global_mapping_index' maps from  the local pre-aggregate table to the row index of
  // global aggregate table
  rmm::device_uvector<cudf::size_type> global_mapping_index(
    grid_size * shared_set_num_elements, stream, mr);
  rmm::device_uvector<cudf::size_type> block_cardinality(grid_size, stream, mr);

  rmm::device_scalar<bool> direct_aggregations(stream, mr);
  direct_aggregations.set_value_to_zero_async(stream);

  cuda::std::optional<cudf::column_device_view> active_mask_device_view =
    active_mask.is_empty() ? cuda::std::nullopt
                           : cuda::std::optional<cudf::column_device_view>(
                               *cudf::column_device_view::create(active_mask, stream));

  compute_mapping_indices<shared_set_ref_type, shared_set_num_elements, cardinality_threshold>
    <<<grid_size, block_size, 0, stream.get()>>>(global_set_ref,
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
  GQE_CUDA_TRY(cudaPeekAtLastError());
  auto const use_direct_aggregation = direct_aggregations.value(stream);

  // 'populated_keys' contains inserted row_indices (keys) of global hash set
  rmm::device_uvector<cudf::size_type> populated_keys(keys.num_rows(), stream, mr);

  cudf::table sparse_table = create_sparse_results_table(flattened_values,
                                                         d_aggs.data(),
                                                         agg_kinds,
                                                         use_direct_aggregation,
                                                         global_set,
                                                         populated_keys,
                                                         stream,
                                                         mr);

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
    stream);

  if (use_direct_aggregation) {
    int stride = block_size * grid_size;
    thrust::for_each_n(rmm::exec_policy_nosync(stream),
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

rmm::device_uvector<cudf::size_type> groupby(
  gqe::groupby::detail::result_cache* dense_results,
  cudf::table_view const& keys,
  cudf::host_span<cudf::groupby::aggregation_request const> requests,
  cudf::column_view const& active_mask,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const null_keys_are_equal = cudf::null_equality::EQUAL;
  auto const has_null            = cudf::nullate::DYNAMIC{cudf::has_nested_nulls(keys)};

  rmm::mr::polymorphic_allocator<cudf::size_type> polly_alloc{mr};

  auto preprocessed_keys = cudf::detail::row::hash::preprocessed_table::create(keys, stream, mr);
  auto const comparator  = cudf::detail::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash    = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
  auto const d_row_hash  = row_hash.device_hasher(has_null);

  if (cudf::detail::has_nested_columns(keys)) {
    throw std::runtime_error("Keys with nested columns are not supported");
  }
  auto const d_key_equal = comparator.equal_to<false>(has_null, null_keys_are_equal);

  cudf::size_type constexpr key_sentinel = -1;  ///< Sentinel value indicating an empty slot
  auto empty_key_sentinel                = cuco::empty_key{key_sentinel};
  double load_factor                     = 0.5;

  auto const num_active_keys = static_cast<std::size_t>(
    gqe::utility::get_num_active_keys(keys.num_rows(), active_mask, stream, mr));

  // size_t avoids overflow issues with large tables
  auto const global_agg_set = cuco::static_set{cuco::extent<std::size_t>{num_active_keys},
                                               load_factor,
                                               empty_key_sentinel,
                                               d_key_equal,
                                               probing_scheme_type{d_row_hash},
                                               {},
                                               {},
                                               polly_alloc,
                                               stream.get()};
  gqe::groupby::detail::result_cache sparse_results(requests.size());

  auto gather_map = compute_single_pass_set_aggs(keys,
                                                 requests,
                                                 &sparse_results,
                                                 global_agg_set,
                                                 stream,
                                                 empty_key_sentinel,
                                                 d_key_equal,
                                                 d_row_hash,
                                                 active_mask,
                                                 mr);

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
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  gqe::groupby::detail::result_cache dense_results(requests.size());

  auto gather_map = groupby(&dense_results, keys, requests, active_mask, stream, mr);

  auto unique_keys = gqe::groupby::detail::gather(keys,
                                                  gather_map,
                                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                                  cudf::negative_index_policy::NOT_ALLOWED,
                                                  stream,
                                                  mr);

  return std::pair(std::move(unique_keys),
                   gqe::extract_results(requests, dense_results, stream, mr));
}

}  // namespace hash
}  // namespace groupby
}  // namespace gqe
