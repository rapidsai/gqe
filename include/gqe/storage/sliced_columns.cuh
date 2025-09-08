#pragma once

#include <rmm/cuda_stream_view.hpp>

namespace gqe {
namespace storage {

template <typename offsets_type>
void adjust_offsets_api(offsets_type* offsets,
                        size_t num_offsets,
                        size_t partition_size,
                        const offsets_type* partition_offsets,
                        size_t char_array_size,
                        rmm::cuda_stream_view stream);

// Extern template declarations for adjust_offsets_api
extern template void adjust_offsets_api<int32_t>(int32_t* offsets,
                                                 size_t num_offsets,
                                                 size_t partition_size,
                                                 const int32_t* partition_offsets,
                                                 size_t char_array_size,
                                                 rmm::cuda_stream_view stream);

extern template void adjust_offsets_api<int64_t>(int64_t* offsets,
                                                 size_t num_offsets,
                                                 size_t partition_size,
                                                 const int64_t* partition_offsets,
                                                 size_t char_array_size,
                                                 rmm::cuda_stream_view stream);

}  // namespace storage
}  // namespace gqe