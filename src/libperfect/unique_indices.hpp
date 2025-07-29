#pragma once

std::tuple<CudaGpuArray<cudf::size_type>, CudaGpuArray<cudf::size_type>> unique_indices(
  const std::vector<ConstCudaGpuBufferPointer>& keys,
  const size_t keys_numel,
  const std::optional<ConstCudaGpuBufferPointer>& mask);
