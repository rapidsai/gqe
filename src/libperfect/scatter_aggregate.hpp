#pragma once

template <typename indices_type>
CudaGpuBuffer scatter_aggregate(const ConstCudaGpuBufferPointer values,
                                const CudaGpuArray<indices_type>& indices,
                                const std::optional<ConstCudaGpuBufferPointer> row_mask,
                                const std::optional<ConstCudaGpuBufferPointer> output_map,
                                const cudf::aggregation::Kind aggregation_kind,
                                int64_t max_index,
                                const cudf::type_id output_type_id);
