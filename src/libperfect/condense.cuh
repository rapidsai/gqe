#pragma once

#include "query_common.cuh"
#include "query_common.hpp"

namespace libperfect {
namespace condense {

enum class WriteToInput : bool { False = false, True = true };

constexpr int THREADS_PER_BLOCK = 128;

template <WriteToInput write_to_input,
          typename input_type,
          typename output_type,
          typename input_count_type>
//__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
__launch_bounds__(THREADS_PER_BLOCK, 1) __global__ void condense_kernel(
  input_type input, int64_t input_count, output_type condensed, input_count_type condensed_count)
{
  auto block_count           = gridDim.x;
  auto threads_per_block     = THREADS_PER_BLOCK;
  auto block_index           = blockIdx.x;
  auto thread_in_block_index = threadIdx.x;
  auto thread_in_grid_index  = threads_per_block * block_index + thread_in_block_index;
  __shared__ typename std::pointer_traits<input_count_type>::element_type shared_unique_index;
  __shared__ typename std::pointer_traits<input_count_type>::element_type global_unique_index;
  if (threadIdx.x == 0) { shared_unique_index = 0; }
  __syncthreads();
  for (int64_t i = thread_in_grid_index; i < input_count; i += threads_per_block * block_count) {
    auto current_value = input[i];
    bool keep_value    = current_value != -1;
    typename std::pointer_traits<input_count_type>::element_type my_index;
    if (keep_value) {
      // This one needs to go into the condensed output.
      my_index =
        atomicAdd(&shared_unique_index,
                  static_cast<typename std::pointer_traits<input_count_type>::element_type>(1));
    }
    __syncthreads();
    if (keep_value && my_index == 0) {
      global_unique_index = atomicAdd(
        static_cast<typename std::pointer_traits<input_count_type>::element_type*>(condensed_count),
        shared_unique_index);
      shared_unique_index = 0;  // Reset for next time.
    }
    __syncthreads();  // Wait for everyone to get the shared_global_condensed_representatives_index.
    if (keep_value) {
      condensed[global_unique_index + my_index] = current_value;
      if (write_to_input == WriteToInput::True) { input[i] = global_unique_index + my_index; }
    }
  }
}

template <WriteToInput write_to_input, typename input_count_type, typename input_type>
CudaGpuArray<cudf::size_type> condense(input_type& input)
{
  // auto cuda_tensor_options = torch::TensorOptions().device(torch::kCUDA);
  auto input_count = input.numel();
  auto condensed =
    CudaGpuArray<typename std::pointer_traits<input_type>::element_type>(input_count);
  auto condensed_count = CudaGpuArray<input_count_type>(1);
  condensed_count.template fill_byte<0>();
  auto block_count = div_round_up(input_count, uint64_t(THREADS_PER_BLOCK));
  condense_kernel<write_to_input><<<block_count, THREADS_PER_BLOCK>>>(
    input.get(), input_count, condensed.get(), condensed_count.get());
  CudaPinnedArray<input_count_type> pinned_condensed_count(1);
  pinned_condensed_count.get().copy_from(
    condensed_count.get(), condensed_count.get() + 1, rmm::cuda_stream_default);
  gpu_throw(cudaDeviceSynchronize());
  condensed.resize(pinned_condensed_count.get()[0]);
  return condensed;
}

template <WriteToInput write_to_input, typename input_type>
CudaGpuArray<cudf::size_type> condense(input_type& input)
{
  auto input_count                = input.numel();
  auto input_count_fits_in_uint32 = input_count < (1ULL << 32);
  if (input_count_fits_in_uint32) {
    return condense<write_to_input, uint32_t>(input);
  } else {
    return condense<write_to_input, uint64_t>(input);
  }
}

}  // namespace condense
}  // namespace libperfect