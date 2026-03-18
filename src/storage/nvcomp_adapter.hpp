/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <gqe/types.hpp>

#include <cudf/types.hpp>
#include <nvcomp/nvcompManager.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

// forward declarations
namespace gqe {
namespace storage {
class compression_manager;
}  // namespace storage
}  // namespace gqe
namespace nvcomp {
struct CPUHLIFManager;
struct nvcompManagerBase;
}  // namespace nvcomp

namespace gqe {
namespace storage {

/**
 * @brief An adapter class that calls supported nvCOMP manager APIs depending on whether public or
 * private nvCOMP is used.
 */
class nvcomp_manager_adapter {
 private:
  std::unique_ptr<nvcomp::nvcompManagerBase> _manager;
  void set_manager(std::unique_ptr<nvcomp::nvcompManagerBase> manager);

 public:
  static std::unique_ptr<nvcomp_manager_adapter> create_manager(
    gqe::storage::compression_manager const& comp_manager,
    gqe::compression_format comp_format,
    nvcompType_t data_type,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  nvcomp::CompressionConfig configure_compression(const size_t uncomp_buffer_size);

  nvcomp::DecompressionConfig configure_decompression(const uint8_t* comp_buffer,
                                                      const size_t* comp_size = nullptr);

  size_t get_compressed_output_size_host(const uint8_t* comp_buffer);

  size_t get_decompressed_output_size_host(const uint8_t* comp_buffer);

  void compress(const uint8_t* uncomp_buffer,
                uint8_t* comp_buffer,
                const nvcomp::CompressionConfig& comp_config,
                size_t* comp_size = nullptr);

  void compress(const uint8_t* const* uncomp_buffers,
                uint8_t* const* comp_buffers,
                const std::vector<nvcomp::CompressionConfig>& comp_configs,
                size_t* comp_sizes = nullptr);

  /**
   * @brief Function to perform compression of a batch of uncompressed buffers
   *
   * @param[in] comp_manager GQE compression manager object
   * @param[in] device_uncompressed Array of pointers to uncompressed buffers
   * @param[in] total_uncompressed_size Total size of the uncompressed column
   * @param[in] uncompressed_ptrs Array of pointers to uncompressed buffers
   * @param[in] comp_format Compression format to use
   * @param[in] data_type Data type to use for compression
   * @param[in] cudf_type CUDF data type of the column being compressed
   * @param[in] num_buffers Number of buffers to compress
   * @param[in] memory_kind Memory kind of the column being compressed
   * @param[out] compressed_ptrs Array of pointers to compressed buffers
   * @param[out] compression_ratio Compression ratio of the compressed column
   * @param[out] total_compressed_size Total size of the compressed column
   * @param[out] compressed_sizes Vector of sizes of the compressed buffers
   * @param[in] stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   * @return Vector of pointers to compressed buffers
   */
  static std::vector<std::unique_ptr<rmm::device_buffer>> compress_batch(
    gqe::storage::compression_manager const& comp_manager,
    const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
    const size_t total_uncompressed_size,
    uint8_t** uncompressed_ptrs,
    gqe::compression_format comp_format,
    nvcompType_t data_type,
    cudf::data_type cudf_type,
    size_t num_buffers,
    memory_kind::type memory_kind,
    uint8_t** compressed_ptrs,
    double& compression_ratio,
    size_t& total_compressed_size,
    std::vector<cudf::size_type>& compressed_sizes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  void decompress(uint8_t* decomp_buffer,
                  const uint8_t* comp_buffer,
                  const nvcomp::DecompressionConfig& decomp_config,
                  size_t* comp_size = nullptr);

  void decompress(uint8_t* const* decomp_buffers,
                  const uint8_t* const* device_comp_buffers,
                  const std::vector<nvcomp::DecompressionConfig>& decomp_configs,
                  const size_t batch_count,
                  const size_t* comp_sizes                = nullptr,
                  const uint8_t* const* host_comp_buffers = nullptr);

  size_t get_compressed_output_size(const uint8_t* comp_buffer);
};

/**
 * @brief An adapter class that calls supported CPU nvCOMP manager APIs depending on whether public
 * or private nvCOMP is used.
 */
class nvcomp_cpu_manager_adapter {
 private:
  std::unique_ptr<nvcomp::CPUHLIFManager> _cpu_manager;
  void set_cpu_manager(std::unique_ptr<nvcomp::CPUHLIFManager> cpu_manager);

 public:
  nvcomp_cpu_manager_adapter() = default;  // default constructor
  nvcomp_cpu_manager_adapter(
    nvcomp_cpu_manager_adapter&&);  // declaration only; definition in impl to
                                    // allow member unique_ptr of incomplete type
  nvcomp_cpu_manager_adapter& operator=(
    nvcomp_cpu_manager_adapter&&);  // declaration only; definition in impl to allow member
                                    // unique_ptr of incomplete type
  ~nvcomp_cpu_manager_adapter();  // declaration only; definition in impl to allow member unique_ptr
                                  // of incomplete type

  static std::unique_ptr<nvcomp_cpu_manager_adapter> create_cpu_manager(
    gqe::storage::compression_manager const& comp_manager,
    gqe::compression_format comp_format,
    int compression_level);

  void cpu_batch_compress(uint8_t** compressed_ptrs,
                          const uint8_t* const* uncomp_ptrs,
                          const size_t* uncomp_sizes,
                          size_t* compressed_sizes,
                          const size_t batch_count,
                          const int num_threads,
                          const size_t* max_comp_sizes,
                          const bool benchmark_mode = false,
                          const int iteration_count = 1);

  size_t get_compressed_output_size(const uint8_t* comp_buffer);

  size_t get_max_compressed_output_size(const size_t input_size);

  /**
   * @brief Function to perform compression of a batch of uncompressed buffers
   *
   * @param[in] comp_manager GQE compression manager object
   * @param[in] device_uncompressed Array of pointers to uncompressed buffers
   * @param[in] total_uncompressed_size Total size of the uncompressed column
   * @param[in] uncompressed_ptrs Array of pointers to uncompressed buffers
   * @param[in] comp_format Compression format to use
   * @param[in] data_type Data type to use for compression
   * @param[in] cudf_type CUDF data type of the column being compressed
   * @param[in] num_buffers Number of buffers to compress
   * @param[in] memory_kind Memory kind of the column being compressed
   * @param[out] compressed_ptrs Array of pointers to compressed buffers
   * @param[out] compression_ratio Compression ratio of the compressed column
   * @param[out] total_compressed_size Total size of the compressed column
   * @param[out] compressed_sizes Vector of sizes of the compressed buffers
   * @param[in] stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   * @return Vector of pointers to compressed buffers
   */
  static std::vector<std::unique_ptr<rmm::device_buffer>> compress_batch(
    gqe::storage::compression_manager const& comp_manager,
    const std::vector<std::unique_ptr<rmm::device_buffer>>& device_uncompressed,
    const size_t total_uncompressed_size,
    uint8_t** uncompressed_ptrs,
    gqe::compression_format comp_format,
    nvcompType_t data_type,
    cudf::data_type cudf_type,
    size_t num_buffers,
    memory_kind::type memory_kind,
    uint8_t** compressed_ptrs,
    double& compression_ratio,
    size_t& total_compressed_size,
    std::vector<cudf::size_type>& compressed_sizes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);
};

}  // namespace storage
}  // namespace gqe
