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

#include "scatter_aggregate.hpp"

#include "query_common.cuh"
#include "query_common.hpp"

#include <gqe/device_properties.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>

#include <optional>

namespace libperfect {

template <cudf::aggregation::Kind aggregation_kind, typename output_type>
__host__ __device__ output_type get_identity()
{
  if constexpr (aggregation_kind == cudf::aggregation::SUM ||
                aggregation_kind == cudf::aggregation::COUNT_VALID ||
                aggregation_kind == cudf::aggregation::COUNT_ALL) {
    return 0;
  } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
    return 1;
  } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
    return std::numeric_limits<output_type>::max();
  } else {
    static_assert(aggregation_kind < 0);  // Support the other aggregation types.
  }
}

template <int thread_count,
          cudf::aggregation::Kind aggregation_kind,
          int scatter_size,
          bool has_row_mask,
          bool has_output_map,
          typename value_type,
          typename index_type,
          typename row_mask_type,
          typename output_map_type,
          typename output_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
//__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void scatter_private_shmem_aggregate_kernel(const value_type values,
                                                       const index_type indices,
                                                       const row_mask_type row_mask,
                                                       const output_map_type output_map,
                                                       int64_t length,
                                                       output_type output)
{
  auto block_count                 = gridDim.x;
  constexpr auto threads_per_block = thread_count;
  auto block_index                 = blockIdx.x;
  auto thread_in_block_index       = threadIdx.x;
  auto thread_in_grid_index        = threads_per_block * block_index + thread_in_block_index;

  // Zero all storage.
  __shared__
    typename std::pointer_traits<output_type>::element_type sum[thread_count][scatter_size];
  for (size_t i = 0; i < scatter_size; i++) {
    sum[thread_in_block_index][i] =
      get_identity<aggregation_kind, typename std::pointer_traits<output_type>::element_type>();
  }
  __shared__ typename std::pointer_traits<output_type>::element_type block_sum[scatter_size];
  for (size_t i = thread_in_block_index; i < scatter_size; i += threads_per_block) {
    block_sum[i] =
      get_identity<aggregation_kind, typename std::pointer_traits<output_type>::element_type>();
  }
  __syncthreads();

  // Do private_shmem sums.
  for (size_t i = thread_in_grid_index; i < length; i += threads_per_block * block_count) {
    if constexpr (has_row_mask) {
      if (!row_mask[i]) { continue; }
    }
    auto index_to_write = indices[i];
    if constexpr (has_output_map) { index_to_write = output_map[index_to_write]; }
    if constexpr (aggregation_kind == cudf::aggregation::SUM) {
      sum[thread_in_block_index][index_to_write] += values[i];
    } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
      sum[thread_in_block_index][index_to_write] =
        min(sum[thread_in_block_index][index_to_write],
            static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
      sum[thread_in_block_index][index_to_write] *= values[i];
    } else if constexpr (aggregation_kind == cudf::aggregation::COUNT_VALID ||
                         aggregation_kind == cudf::aggregation::COUNT_ALL) {
      sum[thread_in_block_index][index_to_write]++;
    } else {
      static_assert(aggregation_kind < 0);  // Support the other aggregation types.
    }
  }

  // Aggregate private_shmems into shmem.
  for (size_t i = 0; i < scatter_size; i++) {
    if constexpr (aggregation_kind == cudf::aggregation::SUM ||
                  aggregation_kind == cudf::aggregation::COUNT_VALID ||
                  aggregation_kind == cudf::aggregation::COUNT_ALL) {
      atomicAdd(&block_sum[i], sum[thread_in_block_index][i]);
    } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
      if constexpr (std::is_same_v<typename std::pointer_traits<output_type>::element_type, int> ||
                    std::is_same_v<typename std::pointer_traits<output_type>::element_type,
                                   unsigned>) {
        auto warp_sum = reduce_min_sync(0xffffffff, sum[thread_in_block_index][i]);
        if (thread_in_block_index % 32 == 0) { atomicMin(&block_sum[i], warp_sum); }
      } else {
        atomicMin(&block_sum[i], sum[thread_in_block_index][i]);
      }
    } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
      atomicProduct(&block_sum[i], sum[thread_in_block_index][i]);
    } else {
      static_assert(aggregation_kind < 0);  // Support the other aggregation types.
    }
  }
  __syncthreads();

  // Sum shmem into global memory.
  for (size_t i = thread_in_block_index; i < scatter_size; i += threads_per_block) {
    if constexpr (aggregation_kind == cudf::aggregation::SUM ||
                  aggregation_kind == cudf::aggregation::COUNT_VALID ||
                  aggregation_kind == cudf::aggregation::COUNT_ALL) {
      atomicAdd(&output[i], block_sum[i]);
    } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
      atomicMin(&output[i], block_sum[i]);
    } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
      atomicProduct(&output[i], block_sum[i]);
    } else {
      static_assert(aggregation_kind < 0);  // Support the other aggregation types.
    }
  }
}

template <cudf::aggregation::Kind aggregation_kind,
          bool has_row_mask,
          bool has_output_map,
          typename value_type,
          typename index_type,
          typename row_mask_type,
          typename output_map_type,
          typename output_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
//__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void scatter_shmem_aggregate_kernel(const value_type values,
                                               const index_type indices,
                                               const row_mask_type row_mask,
                                               const output_map_type output_map,
                                               int64_t scatter_size,
                                               int64_t length,
                                               output_type output)
{
  auto block_count           = gridDim.x;
  auto threads_per_block     = blockDim.x;
  auto block_index           = blockIdx.x;
  auto thread_in_block_index = threadIdx.x;
  auto thread_in_grid_index  = threads_per_block * block_index + thread_in_block_index;

  // Zero all storage.
  extern __shared__ char block_aggregate_memory[];
  auto block_sum = reinterpret_cast<typename std::pointer_traits<output_type>::element_type*>(
    block_aggregate_memory);
  for (size_t i = thread_in_block_index; i < scatter_size; i += threads_per_block) {
    block_sum[i] =
      get_identity<aggregation_kind, typename std::pointer_traits<output_type>::element_type>();
  }
  __syncthreads();

  // Do shmem sums.
  for (size_t i = thread_in_grid_index; i < length; i += threads_per_block * block_count) {
    if constexpr (has_row_mask) {
      if (!row_mask[i]) { continue; }
    }
    auto index_to_write = indices[i];
    if constexpr (has_output_map) { index_to_write = output_map[index_to_write]; }
    if constexpr (aggregation_kind == cudf::aggregation::SUM) {
      atomicAdd(&block_sum[index_to_write],
                static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
      atomicMin(&block_sum[index_to_write],
                static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
      atomicProduct(
        &block_sum[index_to_write],
        static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::COUNT_VALID ||
                         aggregation_kind == cudf::aggregation::COUNT_ALL) {
      atomicAdd(&block_sum[index_to_write],
                static_cast<typename std::pointer_traits<output_type>::element_type>(1));
    } else {
      static_assert(aggregation_kind < 0);  // Support the other aggregation types.
    }
  }
  __syncthreads();

  // Sum shmem into global memory.
  for (size_t i = thread_in_block_index; i < scatter_size; i += threads_per_block) {
    if constexpr (aggregation_kind == cudf::aggregation::SUM ||
                  aggregation_kind == cudf::aggregation::COUNT_VALID ||
                  aggregation_kind == cudf::aggregation::COUNT_ALL) {
      atomicAdd(&output[i], block_sum[i]);
    } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
      atomicProduct(&output[i], block_sum[i]);
    } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
      atomicMin(&output[i], block_sum[i]);
    } else {
      static_assert(aggregation_kind < 0);  // Support the other aggregation types.
    }
  }
}

template <cudf::aggregation::Kind aggregation_kind,
          bool has_row_mask,
          bool has_output_map,
          typename value_type,
          typename index_type,
          typename row_mask_type,
          typename output_map_type,
          typename output_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
//__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void scatter_global_aggregate_kernel(const value_type values,
                                                const index_type indices,
                                                const row_mask_type row_mask,
                                                const output_map_type output_map,
                                                int64_t scatter_size,
                                                int64_t length,
                                                output_type output)
{
  auto block_count           = gridDim.x;
  auto threads_per_block     = blockDim.x;
  auto block_index           = blockIdx.x;
  auto thread_in_block_index = threadIdx.x;
  auto thread_in_grid_index  = threads_per_block * block_index + thread_in_block_index;

  // Do global sums.
  for (size_t i = thread_in_grid_index; i < length; i += threads_per_block * block_count) {
    if constexpr (has_row_mask) {
      if (!row_mask[i]) { continue; }
    }
    auto index_to_write = indices[i];
    if constexpr (has_output_map) { index_to_write = output_map[index_to_write]; }
    if constexpr (aggregation_kind == cudf::aggregation::SUM) {
      atomicAdd(&output[index_to_write],
                static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::MIN) {
      atomicMin(&output[index_to_write],
                static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::PRODUCT) {
      atomicProduct(
        &output[index_to_write],
        static_cast<typename std::pointer_traits<output_type>::element_type>(values[i]));
    } else if constexpr (aggregation_kind == cudf::aggregation::COUNT_VALID ||
                         aggregation_kind == cudf::aggregation::COUNT_ALL) {
      atomicAdd(&output[index_to_write],
                static_cast<typename std::pointer_traits<output_type>::element_type>(1));
    } else {
      static_assert(aggregation_kind < 0);  // Support the other aggregation types.
    }
  }
}

template <typename output_type,
          cudf::aggregation::Kind aggregation_kind,
          bool has_row_mask,
          bool has_output_map,
          typename value_type,
          typename row_mask_type,
          typename output_map_type,
          typename index_type>
CudaGpuArray<output_type> scatter_aggregate_helper(const value_type values,
                                                   const index_type indices,
                                                   const row_mask_type row_mask,
                                                   const output_map_type output_map,
                                                   int64_t max_index,
                                                   int64_t length,
                                                   rmm::cuda_stream_view stream)
{
  // TODO: Optimize the below logic for performance, figuring out
  // which algorithm to run for each case.
  // Get SM count using memoized device properties.
  int sm_count =
    gqe::device_properties::instance().get<gqe::device_properties::multiProcessorCount>();
  constexpr auto BLOCKS_PER_SM = 64;

  auto block_count            = BLOCKS_PER_SM * sm_count;
  constexpr auto thread_count = 64;
  CudaGpuArray<output_type> result(max_index, stream);
  result.fill(get_identity<aggregation_kind, output_type>(), stream);
  // This code causes hard to trace bug in gqe-python tpch q17 with
  // sf100.  It's unclear if the code is even faster than shmem
  // aggregate.
  if (max_index == 0) {
    // nothing to compute, so return 0 sized result
    return result;
  } else if (max_index == 1) {
    scatter_private_shmem_aggregate_kernel<thread_count,
                                           aggregation_kind,
                                           1,
                                           has_row_mask,
                                           has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, length, result.get());
  } else if (max_index <= 2) {
    scatter_private_shmem_aggregate_kernel<thread_count,
                                           aggregation_kind,
                                           2,
                                           has_row_mask,
                                           has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, length, result.get());
  } else if (max_index <= 4) {
    scatter_private_shmem_aggregate_kernel<thread_count,
                                           aggregation_kind,
                                           4,
                                           has_row_mask,
                                           has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, length, result.get());
  } else if (max_index <= 8) {
    scatter_private_shmem_aggregate_kernel<thread_count,
                                           aggregation_kind,
                                           8,
                                           has_row_mask,
                                           has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, length, result.get());
  } else if (max_index <= 16) {
    scatter_private_shmem_aggregate_kernel<thread_count,
                                           aggregation_kind,
                                           16,
                                           has_row_mask,
                                           has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, length, result.get());
  } else if (max_index <= 32) {
    scatter_private_shmem_aggregate_kernel<thread_count,
                                           aggregation_kind,
                                           32,
                                           has_row_mask,
                                           has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, length, result.get());
  } else if (sizeof(output_type) * max_index <= 32768) {
    // 32k fits into 48k shmem for sure.
    scatter_shmem_aggregate_kernel<aggregation_kind, has_row_mask, has_output_map>
      <<<block_count, thread_count, sizeof(output_type) * max_index, stream.value()>>>(
        values, indices, row_mask, output_map, max_index, length, result.get());
  } else {
    scatter_global_aggregate_kernel<aggregation_kind, has_row_mask, has_output_map>
      <<<block_count, thread_count, 0, stream.value()>>>(
        values, indices, row_mask, output_map, max_index, length, result.get());
  }
  // gpu_assert(cudaDeviceSynchronize());
  // gpu_assert(cudaPeekAtLastError());
  return result;
}

template <typename output_type,
          cudf::aggregation::Kind aggregation_kind,
          bool has_row_mask,
          bool has_output_map,
          typename indices_type,
          typename value_type,
          typename row_mask_type,
          typename output_map_type>
CudaGpuArray<output_type> scatter_aggregate_helper(const value_type values,
                                                   rmm::device_uvector<indices_type> const& indices,
                                                   const row_mask_type row_mask,
                                                   const output_map_type output_map,
                                                   int64_t max_index,
                                                   rmm::cuda_stream_view stream)
{
  return scatter_aggregate_helper<output_type, aggregation_kind, has_row_mask, has_output_map>(
    values, indices.cbegin(), row_mask, output_map, max_index, indices.size(), stream);
}

template <typename output_type,
          cudf::aggregation::Kind aggregation_kind,
          bool has_row_mask,
          typename indices_type,
          typename value_type,
          typename row_mask_type>
CudaGpuArray<output_type> scatter_aggregate_helper(
  value_type const values,
  rmm::device_uvector<indices_type> const& indices,
  row_mask_type const row_mask,
  const std::optional<cudf::column_view> output_map,
  int64_t max_index,
  rmm::cuda_stream_view stream)
{
  if (!output_map.has_value()) {
    // Don't use nullptr because nvcc is broken.
    return scatter_aggregate_helper<output_type, aggregation_kind, has_row_mask, false>(
      values, indices, row_mask, 0, max_index, stream);
  }
  switch (output_map->type().id()) {
    case cudf::type_id::INT8:
      return scatter_aggregate_helper<output_type, aggregation_kind, has_row_mask, true>(
        values, indices, row_mask, output_map->data<int8_t const>(), max_index, stream);
    default:
      std::stringstream what;
      what << "Error: Yet unsupported output_map type: "
           << static_cast<int32_t>(output_map->type().id()) << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line((what.str())));
  }
}

template <typename output_type,
          cudf::aggregation::Kind aggregation_kind,
          typename indices_type,
          typename value_type>
CudaGpuArray<output_type> scatter_aggregate_helper(
  value_type const values,
  rmm::device_uvector<indices_type> const& indices,
  const std::optional<ConstCudaGpuBufferPointer> row_mask,
  const std::optional<cudf::column_view> output_map,
  int64_t max_index,
  rmm::cuda_stream_view stream)
{
  if (!row_mask.has_value()) {
    // Don't use nullptr because nvcc is broken.
    return scatter_aggregate_helper<output_type, aggregation_kind, false>(
      values, indices, nullptr, output_map, max_index, stream);
  }
  switch (row_mask->get_id()) {
    case cudf::type_id::INT32:
      return scatter_aggregate_helper<output_type, aggregation_kind, true>(
        values, indices, row_mask->as<int32_t const>(), output_map, max_index, stream);
    case cudf::type_id::BOOL8:
      return scatter_aggregate_helper<output_type, aggregation_kind, true>(
        values, indices, row_mask->as<int8_t const>(), output_map, max_index, stream);
    default:
      std::stringstream what;
      what << "Error: Yet unsupported value type: " << static_cast<int32_t>(row_mask->get_id())
           << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
  }
}

template <typename output_type, cudf::aggregation::Kind aggregation_kind, typename indices_type>
CudaGpuArray<output_type> scatter_aggregate_helper(
  cudf::column_view const& values,
  rmm::device_uvector<indices_type> const& indices,
  const std::optional<ConstCudaGpuBufferPointer> row_mask,
  const std::optional<cudf::column_view> output_map,
  int64_t max_index,
  rmm::cuda_stream_view stream)
{
  switch (values.type().id()) {
    case cudf::type_id::INT32:
      return scatter_aggregate_helper<output_type, aggregation_kind>(
        values.data<int32_t const>(), indices, row_mask, output_map, max_index, stream);
    case cudf::type_id::INT64:
      return scatter_aggregate_helper<output_type, aggregation_kind>(
        values.data<int64_t const>(), indices, row_mask, output_map, max_index, stream);
    case cudf::type_id::FLOAT32:
      return scatter_aggregate_helper<output_type, aggregation_kind>(
        values.data<float const>(), indices, row_mask, output_map, max_index, stream);
    case cudf::type_id::FLOAT64:
      return scatter_aggregate_helper<output_type, aggregation_kind>(
        values.data<double const>(), indices, row_mask, output_map, max_index, stream);
    default:
      std::stringstream what;
      what << "Error: Yet unsupported value type: " << static_cast<int32_t>(values.type().id())
           << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
  }
}

template <typename output_type, typename indices_type>
CudaGpuArray<output_type> scatter_aggregate_helper(
  cudf::column_view const& values,
  rmm::device_uvector<indices_type> const& indices,
  cudf::column_view const& mask,
  const std::optional<cudf::column_view> output_map,
  const cudf::aggregation::Kind aggregation_kind,
  int64_t max_index,
  rmm::cuda_stream_view stream)
{
  PUSH_RANGE("scatter aggregate", 0);
  std::optional<ConstCudaGpuBufferPointer> mask_buffer;
  if (!mask.is_empty()) { mask_buffer.emplace(mask.data<int>(), mask.type().id()); }
  switch (aggregation_kind) {
    case cudf::aggregation::SUM: {
      auto ret = scatter_aggregate_helper<output_type, cudf::aggregation::SUM>(
        values, indices, mask_buffer, output_map, max_index, stream);
      POP_RANGE();
      return ret;
    }
    case cudf::aggregation::PRODUCT: {
      auto ret = scatter_aggregate_helper<output_type, cudf::aggregation::PRODUCT>(
        values, indices, mask_buffer, output_map, max_index, stream);
      POP_RANGE();
      return ret;
    }
    case cudf::aggregation::MIN: {
      auto ret = scatter_aggregate_helper<output_type, cudf::aggregation::MIN>(
        values, indices, mask_buffer, output_map, max_index, stream);
      POP_RANGE();
      return ret;
    }
    case cudf::aggregation::COUNT_VALID: {
      auto ret = scatter_aggregate_helper<output_type, cudf::aggregation::COUNT_VALID>(
        values, indices, mask_buffer, output_map, max_index, stream);
      POP_RANGE();
      return ret;
    }
    case cudf::aggregation::COUNT_ALL: {
      auto ret = scatter_aggregate_helper<output_type, cudf::aggregation::COUNT_ALL>(
        values, indices, mask_buffer, output_map, max_index, stream);
      POP_RANGE();
      return ret;
    }
    default:
      std::stringstream what;
      what << "Error: Yet unsupported aggregation kind: " << static_cast<int32_t>(aggregation_kind)
           << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
  }
  POP_RANGE();
}

template <typename indices_type>
CudaGpuBuffer scatter_aggregate_helper(cudf::column_view const& values,
                                       rmm::device_uvector<indices_type> const& indices,
                                       cudf::column_view const& mask,
                                       const std::optional<cudf::column_view> output_map,
                                       const cudf::aggregation::Kind aggregation_kind,
                                       int64_t max_index,
                                       const cudf::type_id output_type_id,
                                       rmm::cuda_stream_view stream)
{
  switch (output_type_id) {
    case cudf::type_id::INT32:
      return CudaGpuBuffer(std::move(scatter_aggregate_helper<int32_t>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream)));
    case cudf::type_id::INT64:
      return CudaGpuBuffer(std::move(scatter_aggregate_helper<int64_t>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream)));
    case cudf::type_id::FLOAT32:
      return CudaGpuBuffer(std::move(scatter_aggregate_helper<float>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream)));
    case cudf::type_id::FLOAT64:
      return CudaGpuBuffer(std::move(scatter_aggregate_helper<double>(
        values, indices, mask, output_map, aggregation_kind, max_index, stream)));
    default:
      std::stringstream what;
      what << "Error: Yet unsupported output type: " << static_cast<int32_t>(output_type_id)
           << std::endl;
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
  }
}

template <typename indices_type>
cudf::column scatter_aggregate(cudf::column_view const& values,
                               rmm::device_uvector<indices_type> const& indices,
                               cudf::column_view const& mask,
                               const std::optional<cudf::column_view> output_map,
                               const cudf::aggregation::Kind aggregation_kind,
                               int64_t max_index,
                               const cudf::type_id output_type_id,
                               rmm::cuda_stream_view stream)
{
  if (values.size() == 0) {
    return *cudf::make_empty_column(cudf::data_type(output_type_id)).release();
  }
  auto ret = scatter_aggregate_helper<indices_type>(
    values, indices, mask, output_map, aggregation_kind, max_index, output_type_id, stream);
  return cudf::column(cudf::data_type(output_type_id),
                      ret.numel(),
                      std::move(ret.get_buffer()),
                      rmm::device_buffer(0, stream),
                      0);
}

template cudf::column scatter_aggregate(cudf::column_view const& values,
                                        rmm::device_uvector<int> const& indices,
                                        cudf::column_view const& mask,
                                        const std::optional<cudf::column_view> output_map,
                                        const cudf::aggregation::Kind aggregation_kind,
                                        int64_t max_index,
                                        const cudf::type_id output_type_id,
                                        rmm::cuda_stream_view stream);

}  // namespace libperfect
