#pragma once

#include <cub/cub.cuh>

#include "query_common.cuh"

namespace libperfect {

template <typename T>
struct AndOrPair {
  __device__ AndOrPair(T and_value, T or_value) : and_value(and_value), or_value(or_value) {}
  __device__ AndOrPair(T other) : and_value(other), or_value(other) {}
  __device__ AndOrPair() = default;
  __device__ AndOrPair& operator=(T const& other)
  {
    and_value = other;
    or_value  = other;
    return *this;
  }

  T and_value;
  T or_value;
};

struct AndOrPairCombiner {
  template <typename T>
  __device__ AndOrPair<T> operator()(const AndOrPair<T>& a, const AndOrPair<T>& b) const
  {
    return AndOrPair<T>(a.and_value & b.and_value, a.or_value | b.or_value);
  }
  template <typename T>
  __device__ AndOrPair<T> operator()(const AndOrPair<T>& a, const T& b) const
  {
    return AndOrPair<T>(a.and_value & b, a.or_value | b);
  }
};

template <typename T>
void reduce_and_or_helper(AndOrPair<T>* output, T const* inputs, int64_t inputs_size)
{
  AndOrPairCombiner combiner_op;
  auto init = AndOrPair(T(-1), T(0));

  // Determine temporary device storage requirements
  void* d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  PUSH_RANGE("prepare for reduce", 1);
  cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes, inputs, output, inputs_size, combiner_op, init);
  POP_RANGE();
  // Allocate temporary storage
  PUSH_RANGE("make temp storage for reduce", 6);
  auto temp_storage = CudaGpuArray<uint8_t>(temp_storage_bytes);
  POP_RANGE();
  d_temp_storage = temp_storage.get().data();
  // Run reduction
  PUSH_RANGE("do the actual reduce", 1);
  cub::DeviceReduce::Reduce(
    d_temp_storage, temp_storage_bytes, inputs, output, inputs_size, combiner_op, init);
  POP_RANGE();
};

// The first element of and_or gets the binary and reduction of all
// the inputs.  The second element of and_or gets the binary or
// reduction of all the inputs.
template <typename cuda_buffer_type>
cuda_buffer_type reduce_and_or_cuda_helper(ConstCudaGpuBufferPointer const& input,
                                           size_t input_numel)
{
  auto input_width = input.element_size();
  // auto tensor_options = torch::TensorOptions().dtype(input.dtype());
  PUSH_RANGE("make a buffer for reduce", 6);
  cuda_buffer_type and_or(2, input.get_id());
  POP_RANGE();
  PUSH_RANGE("call a helper", 6);
  switch (input_width) {
    case 1:
      reduce_and_or_helper(static_cast<AndOrPair<uint8_t>*>(and_or.get()),
                           static_cast<uint8_t const*>(input),
                           input_numel);
      break;
    case 2:
      reduce_and_or_helper(static_cast<AndOrPair<uint16_t>*>(and_or.get()),
                           static_cast<uint16_t const*>(input),
                           input_numel);
      break;
    case 4:
      reduce_and_or_helper(static_cast<AndOrPair<uint32_t>*>(and_or.get()),
                           static_cast<uint32_t const*>(input),
                           input_numel);
      break;
    case 8:
      reduce_and_or_helper(static_cast<AndOrPair<uint64_t>*>(and_or.get()),
                           static_cast<uint64_t const*>(input),
                           input_numel);
      break;
    default:
      std::stringstream what;
      what << "unsupported size: " << input_width << "\n";
      what << "Error: Just open the file and add a new one, it's really easy.";
      throw std::invalid_argument(annotate_line(what.str()));
      break;
  }
  POP_RANGE();
  return and_or;
}

template <bool pinned>
std::conditional_t<pinned, CudaPinnedBuffer, CudaGpuBuffer> reduce_and_or_cuda(
  ConstCudaGpuBufferPointer const& input, size_t input_numel)
{
  if constexpr (pinned) {
    return reduce_and_or_cuda_helper<CudaPinnedBuffer>(input, input_numel);
  } else {
    return reduce_and_or_cuda_helper<CudaGpuBuffer>(input, input_numel);
  }
}

}  // namespace libperfect
