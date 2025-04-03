/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <gqe/types.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <memory>
#include <nvcomp.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <vector>

using namespace nvcomp;
using namespace gqe;

/**
 * @brief Class that contains functions for compression and decompression of columns in GQE.
 */
class compression_manager {
 private:
  gqe::compression_format _comp_format;
  nvcompType_t _data_type;
  int _chunk_size;
  void print_usage() const;
  /**
   * @brief Function that creates a nvcompManagerBase object based on the compression format.
   *
   * @param[in] supplied_stream Stream to use for compression/decompression
   * @param[in] mr Memory resource to use for allocator and deallocating scratch memory required for
   * nvcomp
   */
  std::unique_ptr<nvcompManagerBase> create_manager(rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref& mr) const;

 public:
  /**
   * @brief Constructor for class
   *
   * @param[in] comp_format Compression format to use
   * @param[in] data_format Data format to use for compression configuration
   * @param[in] explicit_chunk_size Chunk size to use for nvcomp
   */
  compression_manager(gqe::compression_format comp_format,
                      nvcompType_t data_format,
                      int explicit_chunk_size);

  /**
   * @brief Function to perform compression on a column. Retuns a unique pointer to the compressed
   * column.
   *
   * @param[in] uncompressed Data buffer to compress
   * @param[out] compression_ratio Compression ratio of the compressed column
   * @param[out] is_compressed Boolean flag that stores whether compression was viable or not
   * @param[in] supplied_stream Stream to use for compression
   * @param[in] mr Memory resource to use for compression
   */
  std::unique_ptr<rmm::device_buffer> do_compress(rmm::device_buffer const* uncompressed,
                                                  float& compression_ratio,
                                                  bool& is_compressed,
                                                  rmm::cuda_stream_view supplied_stream,
                                                  rmm::device_async_resource_ref mr);

  /**
   * @brief Function to perform dcompression of a compressed column. Returns a unique pointer to the
   * decompressed column.
   *
   * @param[in] compressed Data buffer containing compressed data
   * @param[in] supplied_stream Stream to use for decompression
   * @param[in] mr Memory resource to use for compression
   */
  std::unique_ptr<rmm::device_buffer> do_decompress(rmm::device_buffer const* compressed,
                                                    rmm::cuda_stream_view supplied_stream,
                                                    rmm::device_async_resource_ref mr);

  /**
   * @brief Function to fetch the compression format used by the compression manager.
   *
   * */
  gqe::compression_format get_comp_format() const;

  /**
   * @brief Function to fetch the data type used by the compression manager.
   * */
  nvcompType_t get_data_type() const;

  /**
   * @brief Function to fetch the chunk size used by the compression manager.
   * */
  int get_chunk_size() const;
};
