/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifdef ENABLE_CUSTOMIZED_PARQUET

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/parquet_reader.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <nvcomp/snappy.h>

#include <parquet/parquet_types.h>

#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <cudf/column/column_factories.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include <liburing.h>
#include <sched.h>

namespace cg = cooperative_groups;

namespace gqe::storage {

namespace {

/**
 * @brief Data type of a column.
 */
struct column_data_type {
  // Physical type dictates how the values are stored in the Parquet files.
  parquet::Type::type physical_type;
  // If `physical_type` is FIXED_LEN_BYTE_ARRAY, this is the byte length of the values.
  int32_t type_length = 0;
  // Logical type dictates how the values should be interpreted. This is also the output type of the
  // result cuDF column.
  // The Parquet specification documents how logical types are mapped to physical types:
  // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md
  cudf::data_type logical_type;
  // Whether the column can contain NULLs
  bool nullable;

  bool operator==(const column_data_type& rhs) const
  {
    return physical_type == rhs.physical_type && type_length == rhs.type_length &&
           logical_type == rhs.logical_type && nullable == rhs.nullable;
  }

  bool operator!=(const column_data_type& rhs) const { return !(*this == rhs); }

  // Return the size in byte for the physical type.
  int32_t physical_size() const
  {
    switch (physical_type) {
      case parquet::Type::INT32: return 4;
      case parquet::Type::INT64: return 8;
      case parquet::Type::FLOAT: return 4;
      case parquet::Type::DOUBLE: return 8;
      case parquet::Type::FIXED_LEN_BYTE_ARRAY: return type_length;
      default: throw unsupported_error("Customized Parquet reader: Unsupported physical type");
    }
  }
};

/**
 * @brief Information about a Parquet page
 */
struct page_info {
  int32_t compressed_page_size;    ///< Compressed size of the page
  int32_t uncompressed_page_size;  ///< Uncompressed size of the page
  int64_t column_chunk_offset;  ///< Distance from the start of the column chunk to the start of the
                                ///< page, after the page header
  void* uncompressed_ptr;       ///< Pointer to the decompressed data page in the device memory
  cudf::size_type row_idx;      ///< Start row index in the output column
};

/**
 * @brief Information about a Parquet column chunk
 */
struct column_chunk_info {
  std::size_t idx;  ///< Index of the column chunk in the file
  std::size_t
    estimated_num_blocks;   ///< Estimated number of blocks for the column chunk when using io_uring
  int fd, direct_fd;        ///< Openned file descriptor of the column chunk
  int64_t file_offset;      ///< File offset to the start of the column chunk
  int64_t compressed_size;  ///< Compressed size of the column chunk
  parquet::CompressionCodec::type compression_codec;  ///< Compression format of the column chunk
  void* host_ptr;    ///< Column chunk in host memory, required for decoding the page headers
  void* device_ptr;  ///< Column chunk in device memory
  std::vector<page_info> pages;              ///< Pages within the column chunk
  std::size_t out_column_idx;                ///< Column index in the output table
  void* out_column_base_ptr;                 ///< Base data pointer of the output column
  cudf::bitmask_type* out_bitmask_base_ptr;  ///< Base null mask pointer of the output column
  column_data_type type;                     ///< Data type of the column
  cudf::size_type row_idx;                   ///< Start row index in the output column
  cudf::size_type num_rows;                  ///< Number of rows in the column chunk
};

/**
 * @brief Helper function for converting a Parquet logical type into the corresponding cuDF type.
 */
cudf::data_type parquet_to_logical_type(parquet::LogicalType logical_type,
                                        parquet::Type::type physical_type)
{
  if (logical_type.__isset.DECIMAL) {
    int32_t scale     = logical_type.DECIMAL.scale;
    int32_t precision = logical_type.DECIMAL.precision;

    constexpr int32_t decimal32_precision_threshold = 9;
    constexpr int32_t decimal64_precision_threshold = 18;

    switch (physical_type) {
      case parquet::Type::INT32:
        if (precision > decimal32_precision_threshold) {
          throw std::logic_error(
            "Customized Parquet reader: Precision too large for decimal type with int32");
        }
        return cudf::data_type(cudf::type_id::DECIMAL32, -scale);
      case parquet::Type::INT64:
        if (precision > decimal64_precision_threshold) {
          throw std::logic_error(
            "Customized Parquet reader: Precision too large for decimal type with int64");
        }
        if (precision <= decimal32_precision_threshold) {
          GQE_LOG_WARN(
            "Customized Parquet reader: Using decimal type with int64 for precision <= 9 is "
            "wasteful");
        }
        return cudf::data_type(cudf::type_id::DECIMAL64, -scale);
      case parquet::Type::FIXED_LEN_BYTE_ARRAY:
        if (precision <= decimal32_precision_threshold) {
          return cudf::data_type(cudf::type_id::DECIMAL32, -scale);
        } else if (precision <= decimal64_precision_threshold) {
          return cudf::data_type(cudf::type_id::DECIMAL64, -scale);
        } else {
          throw unsupported_error("Customized Parquet reader: Precision too large");
        }
      default:
        throw unsupported_error(
          "Customized Parquet reader: Unsupported physical type for decimal types");
    }
  } else {
    throw unsupported_error("Customized Parquet reader: Unsupported logical type");
  }
}

/**
 * @brief Helper function for converting a Parquet physical type into the corresponding cuDF type.
 */
cudf::data_type physical_to_logical_type(parquet::Type::type type)
{
  switch (type) {
    case parquet::Type::INT32: return cudf::data_type(cudf::type_id::INT32);
    case parquet::Type::INT64: return cudf::data_type(cudf::type_id::INT64);
    case parquet::Type::FLOAT: return cudf::data_type(cudf::type_id::FLOAT32);
    case parquet::Type::DOUBLE: return cudf::data_type(cudf::type_id::FLOAT64);
    default: throw unsupported_error("Customized Parquet reader: Unsupported physical type");
  }
}

/**
 * @brief A collection of column chunks to be decompressed and decoded together.
 *
 * To use this class, first add the column chunks using `try_add()`, and then call `execute()` to
 * load the column chunks from Parquet files into the result cuDF columns.
 *
 * Note that `io_batch` does not own the host memory bounce buffer and the deivce memory bounce
 * buffer. Instead, the bounce buffers should be preallocated and reused to amortized the allocation
 * cost. The bounce buffers should be kept alive for the duration of this object.
 *
 * The bounce buffer size must be larger than any single column chunks.
 */
class io_batch {
 public:
  /**
   * @brief Construct an object to represent a collection of column chunks.
   *
   * After the construction, the object is empty (i.e., contains no column chunks).
   *
   * @param[in] host_buffer Host memory bounce buffer. It is best to be page-locked to support
   * overlapping. Must have at least `buffer_size` bytes.
   * @param[in] device_buffer Device memory bounce buffer. Must have at least `buffer_size` bytes.
   * @param[in] buffer_size Size of `host_buffer` and `device_buffer`.
   */
  io_batch(void* host_buffer, void* device_buffer, int64_t buffer_size)
    : _host_buffer(host_buffer), _device_buffer(device_buffer), _available_size(buffer_size)
  {
  }

  /**
   * @brief Add a column chunk to the batch.
   *
   * @param[in] chunk Column chunk to be added.
   *
   * @return `true` if the insertion is successful. `false` if there is not enough available space
   * for the chunk.
   */
  [[nodiscard]] bool try_add(column_chunk_info& chunk);

  /**
   * @brief Load the column chunks from Parquet files into cuDF columns.
   *
   * @param[in] num_auxiliary_threads Number of auxiliary threads to use for I/O operations.
   * @param[in] block_size Size of the I/O block in KB.
   * @param[in] engine I/O engine to use.
   * @param[in] pipelining Whether to use pipelining for I/O operations.
   * @param[in] alignment Alignment for I/O operations.
   * @param[in] disk_timer Timer for measuring disk bandwidth.
   * @param[in] h2d_timer Timer for measuring H2D bandwidth.
   * @param[in] decomp_timer Timer for measuring decompression bandwidth.
   * @param[in] decode_timer Timer for measuring decoding bandwidth.
   */
  void execute(std::size_t num_auxiliary_threads,
               std::size_t block_size,
               io_engine_type engine,
               bool pipelining,
               std::size_t alignment,
               gqe::utility::bandwidth_timer& disk_timer,
               gqe::utility::bandwidth_timer& h2d_timer,
               gqe::utility::bandwidth_timer& decomp_timer,
               gqe::utility::bandwidth_timer& decode_timer,
               rmm::cuda_stream_view stream)
  {
    copy(num_auxiliary_threads, block_size, engine, pipelining, alignment, disk_timer, h2d_timer);
    decode_header();
    decompress(decomp_timer, stream);
    decode(decode_timer, stream);
  }

 private:
  // Copy the column chunks from the storage into the CPU bounce buffer (`_host_buffer`) and then
  // into the GPU bounce buffer (`_device_buffer`)
  void copy(std::size_t num_auxiliary_threads,
            std::size_t block_size,
            io_engine_type engine,
            bool pipelining,
            std::size_t alignment,
            gqe::utility::bandwidth_timer& disk_timer,
            gqe::utility::bandwidth_timer& h2d_timer);
  // Decode the page headers and store the page information into `_column_chunks[idx].pages`
  void decode_header();
  // Decompress the pages and store the result in `_decompressed`
  void decompress(gqe::utility::bandwidth_timer& decomp_timer, rmm::cuda_stream_view stream);
  // Decode the pages and store the data into the result cuDF columns
  void decode(gqe::utility::bandwidth_timer& decode_timer, rmm::cuda_stream_view stream);

  void* const _host_buffer;
  void* const _device_buffer;
  int64_t _available_size;
  std::vector<std::reference_wrapper<column_chunk_info>> _column_chunks;
  rmm::device_buffer _decompressed;

  static constexpr int64_t IO_URING_SIZE_THRESHOLD = 1024 * 1024 * 1024;
  // static constexpr int64_t IO_URING_BUFFER_SIZE    = 1000 * 1024 * 1024;
};

bool io_batch::try_add(column_chunk_info& chunk)
{
  if (_available_size < chunk.compressed_size) {
    return false;
  } else {
    _column_chunks.push_back(chunk);
    _available_size -= chunk.compressed_size;
    return true;
  }
}

struct block_info {
  column_chunk_info& column_chunk;
  std::size_t offset;
  std::size_t size;
  void *host_ptr, *device_ptr;
};

void print_sq_poll_kernel_thread_status()
{
  if (system("ps --ppid 2 | grep io_uring-sq") == 0)
    GQE_LOG_INFO("Kernel thread io_uring-sq found running...\n");
  else
    GQE_LOG_WARN("Kernel thread io_uring-sq is not running.\n");
}

void io_batch::copy(std::size_t num_auxiliary_threads,
                    std::size_t block_size,
                    io_engine_type engine,
                    bool pipelining,
                    std::size_t alignment,
                    gqe::utility::bandwidth_timer& disk_timer,
                    gqe::utility::bandwidth_timer& h2d_timer)
{
  utility::nvtx_scoped_range range("io::copy");
  std::size_t block_bytes  = block_size * 1024;
  int estimated_num_blocks = 0;
  auto num_threads         = std::min(num_auxiliary_threads, _column_chunks.size());
  std::sort(_column_chunks.begin(), _column_chunks.end(), [](auto& a, auto& b) {
    return a.get().compressed_size > b.get().compressed_size;
  });
  // Calculate the location in `_host_buffer` and `_device_buffer` for each column chunk
  std::vector<int64_t> offsets;
  offsets.reserve(_column_chunks.size());

  int64_t current_offset = 0, chunk_count = 0;
  for (column_chunk_info& column_chunk : _column_chunks) {
    // Add padding to align the column chunk file offset to ensure alignment for io_uring read
    current_offset += column_chunk.file_offset % alignment;
    column_chunk.idx = chunk_count++;
    offsets.push_back(current_offset);
    column_chunk.host_ptr   = static_cast<std::byte*>(_host_buffer) + current_offset;
    column_chunk.device_ptr = static_cast<std::byte*>(_device_buffer) + current_offset;

    current_offset += column_chunk.compressed_size;
    // Add padding to memory buffer to ensure alignment for H2D copy
    current_offset = utility::divide_round_up(current_offset, alignment) * alignment;
    if (current_offset > _available_size) {
      throw std::runtime_error("Customized Parquet reader: Not enough available space");
    }
    column_chunk.estimated_num_blocks =
      utility::divide_round_up(column_chunk.compressed_size, block_bytes);
    estimated_num_blocks += column_chunk.estimated_num_blocks;
  }

  GQE_LOG_TRACE("chunk count: {}, read size: {} MB", _column_chunks.size(), current_offset / 1e6);

  auto effective_engine = engine;
  if (effective_engine == io_engine_type::AUTO) {  // AUTO
    if (current_offset > IO_URING_SIZE_THRESHOLD)
      effective_engine = io_engine_type::IO_URING;
    else
      effective_engine = io_engine_type::PSYNC;
  }

  disk_timer.start();
  disk_timer.add(current_offset);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<rmm::cuda_stream> copy_streams(num_threads);

  // Launch multiple threads to copy column chunks from disk into CPU memory and then into GPU
  // memory.
  // Using multiple threads increases the number of concurrent I/O requests to the storage, which
  // usually improves performance.
  if (effective_engine == io_engine_type::PSYNC) {
    for (std::size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
      threads.emplace_back(
        [this, thread_idx, num_auxiliary_threads, pipelining, &offsets, &copy_streams]() {
          rmm::cuda_stream& copy_stream = copy_streams[thread_idx];

          for (std::size_t chunk_idx = thread_idx; chunk_idx < _column_chunks.size();
               chunk_idx += num_auxiliary_threads) {
            column_chunk_info& column_chunk = _column_chunks[chunk_idx];
            auto host_ptr                   = column_chunk.host_ptr;
            auto device_ptr                 = column_chunk.device_ptr;
            auto const copy_size            = column_chunk.compressed_size;

            if (pread(column_chunk.fd, host_ptr, copy_size, column_chunk.file_offset) !=
                copy_size) {
              throw std::runtime_error("Customized Parquet reader: pread failure");
            }
            if (pipelining) {
              GQE_CUDA_TRY(cudaMemcpyAsync(
                device_ptr, host_ptr, copy_size, cudaMemcpyHostToDevice, copy_stream.value()));
            }
          }
          if (pipelining) { copy_stream.synchronize(); }
        });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    disk_timer.end();
    if (!pipelining) {
      h2d_timer.start();
      h2d_timer.add(current_offset);
      rmm::cuda_stream& copy_stream = copy_streams[0];
      for (auto& _column_chunk : _column_chunks) {
        auto column_chunk = _column_chunk.get();
        auto host_ptr     = column_chunk.host_ptr;
        auto device_ptr   = column_chunk.device_ptr;
        auto copy_size    = column_chunk.compressed_size;
        GQE_CUDA_TRY(cudaMemcpyAsync(
          device_ptr, host_ptr, copy_size, cudaMemcpyHostToDevice, copy_stream.value()));
      }
      copy_stream.synchronize();
      h2d_timer.end();
    }
  } else {
    print_sq_poll_kernel_thread_status();
    for (std::size_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
      threads.emplace_back([this,
                            thread_idx,
                            num_auxiliary_threads,
                            alignment,
                            pipelining,
                            &offsets,
                            &copy_streams,
                            block_bytes]() {
        int chunk_cnt                 = 0;
        rmm::cuda_stream& copy_stream = copy_streams[thread_idx];
        struct io_uring ring;
        struct io_uring_params params;
        memset(&params, 0, sizeof(params));
        params.flags |= IORING_SETUP_SQPOLL;
        auto estimated_num_blocks = 0;
        for (std::size_t chunk_idx = thread_idx; chunk_idx < _column_chunks.size();
             chunk_idx += num_auxiliary_threads) {
          column_chunk_info& column_chunk = _column_chunks[chunk_idx];
          estimated_num_blocks += column_chunk.estimated_num_blocks;
        }

        std::vector<block_info> blocks;
        blocks.reserve(estimated_num_blocks);
        int block_count = 0;
        if (io_uring_queue_init(estimated_num_blocks, &ring, 0) != 0) {
          throw std::runtime_error("Customized Parquet reader: io_uring_queue_init failure");
        }

        for (std::size_t chunk_idx = thread_idx; chunk_idx < _column_chunks.size();
             chunk_idx += num_auxiliary_threads) {
          ++chunk_cnt;
          column_chunk_info& column_chunk = _column_chunks[chunk_idx];
          auto head_padding               = column_chunk.file_offset % alignment;
          auto start_host_ptr    = static_cast<std::byte*>(column_chunk.host_ptr) - head_padding;
          auto start_file_offset = column_chunk.file_offset - head_padding;
          auto start_device_ptr  = static_cast<std::byte*>(column_chunk.device_ptr) - head_padding;
          int64_t total_size     = column_chunk.compressed_size + head_padding;
          int num_blocks         = utility::divide_round_up(total_size, block_bytes);
          for (std::size_t current_offset = 0; current_offset < num_blocks * block_bytes;
               current_offset += block_bytes) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            auto block_size          = std::min(block_bytes, total_size - current_offset);
            int fd           = block_size == block_bytes ? column_chunk.direct_fd : column_chunk.fd;
            auto file_offset = start_file_offset + current_offset;
            if (file_offset % alignment) {
              GQE_LOG_ERROR("column_chunk.file_offset %% io_alignment = {}",
                            file_offset % alignment);
            }
            io_uring_prep_read(sqe, fd, start_host_ptr + current_offset, block_size, file_offset);
            io_uring_sqe_set_data64(sqe, block_count);
            if (io_uring_submit(&ring) != 1) {
              // io_uring_submit should return 1 for a single submission success
              throw std::runtime_error("Customized Parquet reader: io_uring_submit failure");
            }
            block_info block{
              column_chunk,
              file_offset,
              block_size,
              start_host_ptr + current_offset,
              start_device_ptr + current_offset,
            };
            // We have calculated estimated_num_blocks before, which is upper bound of actual
            // blocks so the std::vector will not get resized
            blocks.push_back(block);
            block_count++;
          }
        }
        int count = 0;
        struct io_uring_cqe* cqe;
        while (count < block_count) {
          if (io_uring_peek_cqe(&ring, &cqe) != 0) { continue; }
          count++;
          if (!cqe || cqe->res < 0) { GQE_LOG_ERROR("io_uring_wait_cqe error: {}", cqe->res); }
          int bid           = io_uring_cqe_get_data64(cqe);
          auto& block       = blocks[bid];
          auto column_chunk = block.column_chunk;
          io_uring_cqe_seen(&ring, cqe);
          if (pipelining) {
            GQE_CUDA_TRY(cudaMemcpyAsync(block.device_ptr,
                                         block.host_ptr,
                                         block.size,
                                         cudaMemcpyHostToDevice,
                                         copy_stream.value()));
          }
        }
        io_uring_queue_exit(&ring);
        if (pipelining) { copy_stream.synchronize(); }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
    disk_timer.end();
    if (!pipelining) {
      rmm::cuda_stream& copy_stream = copy_streams[0];
      h2d_timer.start();
      h2d_timer.add(current_offset);
      for (auto& _column_chunk : _column_chunks) {
        auto column_chunk = _column_chunk.get();
        auto host_ptr     = column_chunk.host_ptr;
        auto device_ptr   = column_chunk.device_ptr;
        auto copy_size    = column_chunk.compressed_size;
        GQE_CUDA_TRY(cudaMemcpyAsync(
          device_ptr, host_ptr, copy_size, cudaMemcpyHostToDevice, copy_stream.value()));
      }
      copy_stream.synchronize();
      h2d_timer.end();
    }
  }  // end of liburing copy
}

void io_batch::decode_header()
{
  utility::nvtx_scoped_range range("io::decode_header");
  for (column_chunk_info& column_chunk : _column_chunks) {
    // Factory for thrift compact protocol used for decoding page headers
    apache::thrift::protocol::TCompactProtocolFactoryT<apache::thrift::transport::TMemoryBuffer>
      thrift_compact_protocol_factory;

    // Loop through all pages of the column chunk
    cudf::size_type num_rows_processed = 0;
    int64_t page_offset                = 0;
    while (num_rows_processed < column_chunk.num_rows) {
      // Decode the page header
      auto page_header_transport = std::make_shared<apache::thrift::transport::TMemoryBuffer>(
        static_cast<uint8_t*>(column_chunk.host_ptr) + page_offset,
        column_chunk.compressed_size - page_offset,
        apache::thrift::transport::TMemoryBuffer::OBSERVE);

      auto page_header_protocol =
        thrift_compact_protocol_factory.getProtocol(page_header_transport);

      parquet::PageHeader page_header;
      auto const page_header_size = page_header.read(page_header_protocol.get());
      page_offset += page_header_size;

      if (page_header.type != parquet::PageType::DATA_PAGE) {
        throw unsupported_error("Customized Parquet reader: Only supports data pages v1");
      }
      auto const data_page_header = page_header.data_page_header;

      if (data_page_header.encoding != parquet::Encoding::PLAIN ||
          data_page_header.definition_level_encoding != parquet::Encoding::RLE) {
        throw unsupported_error("Customized Parquet reader: Unsupported encoding");
      }

      // Store the page info in the header into column chunk info
      page_info info;
      info.compressed_page_size   = page_header.compressed_page_size;
      info.uncompressed_page_size = page_header.uncompressed_page_size;
      info.column_chunk_offset    = page_offset;
      info.uncompressed_ptr       = nullptr;
      info.row_idx                = column_chunk.row_idx + num_rows_processed;

      assert(info.column_chunk_offset >= 0);

      column_chunk.pages.push_back(std::move(info));

      num_rows_processed += data_page_header.num_values;
      page_offset += page_header.compressed_page_size;
    }
  }
}

// Helper function for copying a host vector to the device memory.
// This function is asynchronous to the host.
template <typename T>
std::unique_ptr<rmm::device_uvector<T>> copy_to_device(std::vector<T> const& host_vec,
                                                       rmm::cuda_stream_view stream)
{
  auto const size = host_vec.size();
  auto device_vec = std::make_unique<rmm::device_uvector<T>>(size, stream);
  GQE_CUDA_TRY(cudaMemcpyAsync(
    device_vec->data(), host_vec.data(), sizeof(T) * size, cudaMemcpyHostToDevice, stream.value()));
  return device_vec;
}

/**
 * @brief Batch copy the data unmodified from sources to targets.
 *
 * This kernel copies the data from compressed_ptrs[idx] to uncompressed_ptrs[idx] with size
 * compressed_bytes[idx] with 0 <= idx < batch_size. It is helpful when the compression format is
 * UNCOMPRESSED.
 *
 * TODO: Use CUB's DeviceBatchMemcpy if CUB 2.1.0 is available (CUDA >=12.2).
 */
__global__ void copy_uncompressed_buffers(void const* const* compressed_ptrs,
                                          std::size_t const* compressed_bytes,
                                          void* const* uncompressed_ptrs,
                                          std::size_t const* uncompressed_bytes,
                                          std::size_t batch_size)
{
  // Assign each buffer to a thread block
  for (std::size_t buffer_idx = blockIdx.x; buffer_idx < batch_size; buffer_idx += gridDim.x) {
    auto in_ptr        = static_cast<std::byte const*>(compressed_ptrs[buffer_idx]);
    auto const in_size = compressed_bytes[buffer_idx];
    auto out_ptr       = static_cast<std::byte*>(uncompressed_ptrs[buffer_idx]);
    [[maybe_unused]] auto const out_size = uncompressed_bytes[buffer_idx];

    assert(in_size == out_size);

    // Assign each byte to a thread
    for (std::size_t byte_idx = threadIdx.x; byte_idx < in_size; byte_idx += blockDim.x) {
      out_ptr[byte_idx] = in_ptr[byte_idx];
    }
  }
}

void io_batch::decompress(gqe::utility::bandwidth_timer& decomp_timer, rmm::cuda_stream_view stream)
{
  utility::nvtx_scoped_range range("io::decompress");
  decomp_timer.start();
  // First pass: calculate the total decompressed size
  int64_t decompressed_size = 0, compressed_size = 0, num_pages = 0;
  for (column_chunk_info const& column_chunk : _column_chunks) {
    for (auto const& page : column_chunk.pages) {
      decompressed_size += page.uncompressed_page_size;
      compressed_size += page.compressed_page_size;
    }
  }

  // Allocate the decompressed buffer
  _decompressed.resize(decompressed_size, stream);

  // Second pass: get the necessary information for nvcomp
  struct compression_info {
    int64_t max_chunk_bytes = 0;
    std::vector<void const*> compressed_ptrs;
    std::vector<std::size_t> compressed_bytes;
    std::vector<void*> uncompressed_ptrs;
    std::vector<std::size_t> uncompressed_bytes;
  };

  // Map from the compression codec to the buffer descriptions
  std::map<parquet::CompressionCodec::type, compression_info> codec_name_to_info;

  int64_t decompressed_offset = 0;

  for (column_chunk_info& column_chunk : _column_chunks) {
    auto [it, success] =
      codec_name_to_info.insert(std::make_pair(column_chunk.compression_codec, compression_info{}));
    auto& comp_info = it->second;

    auto const compressed_base_ptr = column_chunk.device_ptr;
    num_pages += column_chunk.pages.size();
    for (auto& page : column_chunk.pages) {
      auto const page_size = page.uncompressed_page_size;
      if (page_size > comp_info.max_chunk_bytes) comp_info.max_chunk_bytes = page_size;

      auto uncompressed_ptr = static_cast<std::byte*>(_decompressed.data()) + decompressed_offset;

      comp_info.compressed_ptrs.push_back(static_cast<std::byte*>(compressed_base_ptr) +
                                          page.column_chunk_offset);
      comp_info.compressed_bytes.push_back(page.compressed_page_size);
      comp_info.uncompressed_ptrs.push_back(uncompressed_ptr);
      comp_info.uncompressed_bytes.push_back(page_size);

      page.uncompressed_ptr = uncompressed_ptr;
      decompressed_offset += page_size;
      decomp_timer.add(page.compressed_page_size, page.uncompressed_page_size);
    }
  }

  // Launch decompression
  for (auto const& [codec, comp_info] : codec_name_to_info) {
    auto const batch_size = comp_info.compressed_ptrs.size();
    // Copy the pointers and the sizes to the device memory
    auto device_compressed_ptrs    = copy_to_device(comp_info.compressed_ptrs, stream);
    auto device_compressed_bytes   = copy_to_device(comp_info.compressed_bytes, stream);
    auto device_uncompressed_ptrs  = copy_to_device(comp_info.uncompressed_ptrs, stream);
    auto device_uncompressed_bytes = copy_to_device(comp_info.uncompressed_bytes, stream);

    switch (codec) {
      case parquet::CompressionCodec::SNAPPY: {
        // Allocate the temporary workspace used by nvcomp
        std::size_t temp_bytes;
        if (nvcompBatchedSnappyDecompressGetTempSize(
              batch_size, comp_info.max_chunk_bytes, &temp_bytes) != nvcompSuccess) {
          throw std::runtime_error("Customized Parquet reader: Cannot get nvcomp temp size");
        }

        rmm::device_buffer nvcomp_temp_workspace(temp_bytes, stream);

        if (nvcompBatchedSnappyDecompressAsync(device_compressed_ptrs->data(),
                                               device_compressed_bytes->data(),
                                               device_uncompressed_bytes->data(),
                                               nullptr,
                                               batch_size,
                                               nvcomp_temp_workspace.data(),
                                               temp_bytes,
                                               device_uncompressed_ptrs->data(),
                                               nullptr,
                                               stream.value()) != nvcompSuccess) {
          throw std::runtime_error("Customized Parquet reader: Snappy decompression error");
        }

        stream.synchronize();
        break;
      }
      case parquet::CompressionCodec::UNCOMPRESSED: {
        constexpr int32_t block_size = 128;
        copy_uncompressed_buffers<<<batch_size, block_size, 0, stream.value()>>>(
          device_compressed_ptrs->data(),
          device_compressed_bytes->data(),
          device_uncompressed_ptrs->data(),
          device_uncompressed_bytes->data(),
          batch_size);

        stream.synchronize();
        break;
      }
      default: throw unsupported_error("Customized Parquet reader: Unsupported compression format");
    }
  }
  decomp_timer.end();
}

/**
 * @brief Load a value from memory encoded as little endian and interpreted as `T`.
 *
 * @param[in] start Start address of the value. The address can be unaligned.
 * @param[in] length Size of the value. Must be a positive integer (i.e., cannot be zero).
 */
template <typename T>
__device__ T load_little_endian(void const* start, std::size_t length = sizeof(T))
{
  // Avoid using the signed type because bitshift to the signed bit has undefined behavior
  using unsigned_type = gqe::utility::make_unsigned_t<T>;

  unsigned_type rtv = 0;
  auto ptr          = static_cast<uint8_t const*>(start);

  for (std::size_t byte = 0; byte < length; byte++) {
    rtv |= (static_cast<unsigned_type>(ptr[byte]) << (byte * 8));
  }

  if constexpr (std::is_signed_v<T>) {
    // Fill the rest of the value with the leading (sign) bit, to be in accordance with the two's
    // complement representation.
    auto const sign_bits =
      ptr[length - 1] & 0x80 ? static_cast<unsigned_type>(-1) : static_cast<unsigned_type>(0);
    rtv |= (sign_bits << length * 8);
  }

  return *reinterpret_cast<T*>(&rtv);
}

/**
 * @brief Load a value from memory encoded as big endian and interpreted as `T`.
 *
 * @param[in] start Start address of the value. The address can be unaligned.
 * @param[in] length Size of the value. Must be a positive integer (i.e., cannot be zero).
 */
template <typename T>
__device__ T load_big_endian(void const* start, std::size_t length = sizeof(T))
{
  // Avoid using the signed type because bitshift to the signed bit has undefined behavior
  using unsigned_type = gqe::utility::make_unsigned_t<T>;

  unsigned_type rtv = 0;
  auto ptr          = static_cast<uint8_t const*>(start);

  for (std::size_t byte = 0; byte < length; byte++) {
    rtv |= (static_cast<unsigned_type>(ptr[byte]) << ((length - byte - 1) * 8));
  }

  if constexpr (std::is_signed_v<T>) {
    // Fill the rest of the value with the leading (sign) bit, to be in accordance with the two's
    // complement representation.
    auto const sign_bits =
      ptr[0] & 0x80 ? static_cast<unsigned_type>(-1) : static_cast<unsigned_type>(0);
    rtv |= (sign_bits << length * 8);
  }

  return *reinterpret_cast<T*>(&rtv);
}

/**
 * @brief Calculates the length of a bitpacked run (<bit-pack-scaled-run-len>) or a RLE run
 * (<rle-run-len>) from the header.
 *
 * The length is encoded using ULEB128 (https://en.wikipedia.org/wiki/LEB128).
 *
 * @param[in, out] ptr Pointer to the start of the either <bit-packed-header> or <rle-header>. Once
 * this function finishes, the pointer will be moved to the end of the header.
 */
__device__ int32_t calculate_run_length(uint8_t const*& ptr)
{
  uint32_t result = 0;
  uint32_t shift  = 0;
  while (true) {
    uint8_t byte = *ptr;  // next byte in input
    ptr++;
    result |= static_cast<uint32_t>(byte & 0x7F) << shift;  // (low-order 7 bits of byte) << shift
    if (!(byte & 0x80)) break;                              // if (high-order bit of byte == 0)
    shift += 7;
  }
  return static_cast<int32_t>(result >> 1);
}

/**
 * @brief Set the value of a row in a cuDF column from a Parquet page.
 *
 * @param[in] type Data type of the column.
 * @param[in] out_ptr Base pointer of the column.
 * @param[in] row_idx Row index to be set.
 * @param[in] value Location of the value in the Parquet page.
 * @param[in] length Size in byte of the value in the Parquet page.
 */
__device__ void set_value(
  cudf::type_id type, void* out_ptr, cudf::size_type row_idx, void const* value, int32_t length)
{
  switch (type) {
    case cudf::type_id::INT32:
      *(static_cast<int32_t*>(out_ptr) + row_idx) = load_little_endian<int32_t>(value, length);
      break;
    case cudf::type_id::INT64:
      *(static_cast<int64_t*>(out_ptr) + row_idx) = load_little_endian<int64_t>(value, length);
      break;
    case cudf::type_id::FLOAT32:
      *(static_cast<float*>(out_ptr) + row_idx) = load_little_endian<float>(value, length);
      break;
    case cudf::type_id::FLOAT64:
      *(static_cast<double*>(out_ptr) + row_idx) = load_little_endian<double>(value, length);
      break;
    case cudf::type_id::DECIMAL32:
      // For decimal types with byte arrays, Parquet encodes the unscaled number as two's complement
      // using the big-endian byte order.
      // FIXME: Is this true for other primitive types as well?
      *(static_cast<int32_t*>(out_ptr) + row_idx) = load_big_endian<int32_t>(value, length);
      break;
    case cudf::type_id::DECIMAL64:
      *(static_cast<int64_t*>(out_ptr) + row_idx) = load_big_endian<int64_t>(value, length);
      break;
  }
}

/**
 * @brief Copy a range of bits.
 *
 * This function copies bits in [src, src + num_bytes] to bitmask `dst` starting at bit offset
 * `start_bit_offset`. Assume `dst` is initialized to 0.
 *
 * This is a collective function by all threads in the coopertive group `group`.
 *
 * @param[in] group Cooperative group.
 * @param[in] dst Base pointer of the bitmask.
 * @param[in] start_bit_offset Offset to `dst` of the first bit to be copied. Note that this
 * argument indicates the number of bits instead of the number of bytes.
 * @param[in] src Source pointer of the bits to be copied.
 * @param[in] num_bytes Number of bytes to be copied.
 */
template <typename cg_type>
__device__ void copy_bits(cg_type const& group,
                          cudf::bitmask_type* dst,
                          cudf::size_type start_bit_offset,
                          uint8_t const* src,
                          int32_t num_bytes)
{
  static_assert(std::is_unsigned_v<cudf::bitmask_type>);

  constexpr int32_t bitmask_size = sizeof(cudf::bitmask_type);
  constexpr int32_t word_width   = sizeof(cudf::bitmask_type) * 8;  // number of bits per word

  // last bit that needs update in `dst`
  auto const end_bit_offset = start_bit_offset + (num_bytes * 8) - 1;
  // first word that needs update in `dst`
  auto const start_offset = start_bit_offset / word_width;
  // last word that needs update in `dst`
  auto const end_offset = end_bit_offset / word_width;
  auto const bit_shift  = start_bit_offset - start_offset * word_width;

  // Assign each word in `dst` to a thread
  // Note that despite `src` may not be memory aligned, we cannot assign each byte to a thread
  // because atomicOr does not support 1B input.
  for (int32_t out_idx = start_offset + group.thread_rank(); out_idx <= end_offset;
       out_idx += group.num_threads()) {
    cudf::bitmask_type out_word = 0;
    auto const src_idx          = out_idx - start_offset;

    if (src_idx > 0) {
      auto const src_byte_offset = (src_idx - 1) * bitmask_size;
      auto const src_addr        = src + src_byte_offset;
      auto const length          = min(bitmask_size, num_bytes - src_byte_offset);
      out_word |=
        load_little_endian<cudf::bitmask_type>(src_addr, length) >> (word_width - bit_shift);
    }

    if (src_idx * bitmask_size < num_bytes) {
      auto const src_byte_offset = src_idx * bitmask_size;
      auto const src_addr        = src + src_byte_offset;
      auto const length          = min(bitmask_size, num_bytes - src_byte_offset);

      out_word |= load_little_endian<cudf::bitmask_type>(src_addr, length) << bit_shift;
    }

    if (out_idx == start_offset || out_idx == end_offset) {
      // Since bits are initialized to 0, we can use atomicOr to update part of the word.
      // The update needs to be atomic because this word is shared with other runs. It is possible
      // that the word is being updated at the same time by another thread.
      atomicOr(dst + out_idx, out_word);
    } else {
      dst[out_idx] = out_word;
    }
  }
}

/**
 * @brief Set a range of bits in a bitmask to 1.
 *
 * This function sets `num_bits` starting from `start_bit_offset` to 1. Assume the bitmask is
 * initialized with 0.
 *
 * @param[in] group Cooperative group.
 * @param[in] dst Base pointer of the bitmask.
 * @param[in] start_bit_offset Offset to `dst` of the first bit to be set.
 * @param[in] num_bits Number of bits to be set.
 */
template <typename cg_type>
__device__ void set_bits(cg_type const& group,
                         cudf::bitmask_type* dst,
                         cudf::size_type start_bit_offset,
                         cudf::size_type num_bits)
{
  static_assert(std::is_unsigned_v<cudf::bitmask_type>);

  constexpr int32_t word_width = sizeof(cudf::bitmask_type) * 8;  // number of bits per word

  auto const end_bit_offset = start_bit_offset + num_bits - 1;
  auto const start_idx      = start_bit_offset / word_width;
  auto const end_idx        = end_bit_offset / word_width;

  for (int32_t out_idx = start_idx + group.thread_rank(); out_idx <= end_idx;
       out_idx += group.num_threads()) {
    // A bitmask with all bits set to 1
    auto const all_one_mask = static_cast<cudf::bitmask_type>(-1);

    // We start with a bitmask with all bits set to 1, and unset the bits that are out of the range.
    auto mask = all_one_mask;

    int32_t first_bit = out_idx * word_width;
    int32_t last_bit  = first_bit + word_width - 1;

    // Unset bits before `start_bit_offset`
    if (first_bit < start_bit_offset) mask &= (all_one_mask << (start_bit_offset - first_bit));
    // Unset bits after `end_bit_offset`
    if (last_bit > end_bit_offset) mask &= (all_one_mask >> (last_bit - end_bit_offset));

    if (out_idx == start_idx || out_idx == end_idx) {
      // Since bits are initialized to 0, we can use atomicOr to update part of the word
      atomicOr(dst + out_idx, mask);
    } else {
      dst[out_idx] = mask;
    }
  }
}

/**
 * @brief Decode a batch of Parquet pages to cuDF columns.
 *
 * It is important that the kernel can handle multiple pages from different column chunks (possibly
 * with different data types) to improve GPU utilization.
 *
 * Currently, this kernel only works with nullable columns. I.e., it assumes `output_bitmask` has
 * been properly allocated, and the Parquet pages contain the definition level section.
 *
 * @param[in] page_data Array of size `num_pages`, where each element stores the pointer to the data
 * page in the Parquet file.
 * @param[in] output_data Array of size `num_pages`, where each element stores the base pointer to
 * the output data column for the page.
 * @param[in] output_bitmask Array of size `num_pages`, where each element stores the base pointer
 * to the output bitmask for the page.
 * @param[in] physical_type_sizes Array of size `num_pages`, where each element stores the size of
 * the physical type of the page.
 * @param[in] logical_type_ids Array of size `num_pages`, where each element stores the logical type
 * of the page.
 * @param[in] row_idx Array of size `num_pages`, where each element stores the start row index into
 * `output_data` and `output_bitmask` of the page.
 * @param[in] num_pages Number of pages to be decoded. This is the batch size.
 */
template <int32_t block_size>
__global__ void decode_pages_kernel(void const* const* page_data,
                                    void* const* output_data,
                                    cudf::bitmask_type* const* output_bitmask,
                                    int32_t const* physical_type_sizes,
                                    cudf::type_id const* logical_type_ids,
                                    cudf::size_type const* row_idx,
                                    int32_t num_pages)
{
  constexpr int32_t warp_size        = 32;
  constexpr auto num_warps_per_block = block_size / warp_size;
  auto group                         = cg::tiled_partition<warp_size>(cg::this_thread_block());

  auto const local_warp_id  = threadIdx.x / warp_size;  // warp id within the threadblock
  auto const global_warp_id = blockIdx.x * num_warps_per_block + local_warp_id;
  auto const num_warps      = gridDim.x * num_warps_per_block;
  auto const warp_lane      = threadIdx.x % 32;

  // Allocate WarpScan shared memory
  typedef cub::WarpScan<uint8_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[num_warps_per_block];

  // Assign each page to a warp
  for (auto page_idx = global_warp_id; page_idx < num_pages; page_idx += num_warps) {
    auto const type_size = physical_type_sizes[page_idx];
    auto const type_id   = logical_type_ids[page_idx];

    // The first 4B is the size of the definition level section in little endian
    auto const definition_level_size = load_little_endian<uint32_t>(page_data[page_idx]);

    // pointer to the current definition level being decoded
    auto level_ptr = static_cast<uint8_t const*>(page_data[page_idx]) + 4;
    // pointer to the start of the value section
    auto start_value_ptr = level_ptr + definition_level_size;
    // pointer to the current value being decoded
    auto value_ptr = start_value_ptr;
    // Row index from the start of the output column
    cudf::size_type row_offset = row_idx[page_idx];

    // Keep decoding until the definition levels have been completely parsed.
    // Note that the end of the definition level section is the same as the start of the values
    // section.
    while (reinterpret_cast<uintptr_t>(level_ptr) < reinterpret_cast<uintptr_t>(start_value_ptr)) {
      auto const tag    = *level_ptr;
      auto const length = calculate_run_length(level_ptr);

      if (tag & 1) {
        // The current run is a bit-packed run
        // For a flat data type, each value has 1 bit stored in the definition level: 0 for NULL,
        // and 1 for not NULL. In the value section, Parquet does not reserve space for the NULL
        // values, but Arrow does. We assign each thread for an output location, and use a warp
        // scan to calculate the corresponding input location.

        // Parquet always bit-packs a multiple of 8 values at a time, so the actual number of
        // values <bit-packed-run-len> = <bit-pack-scaled-run-len> * 8
        auto const bitpacked_length = length * 8;

        // Since we use whole warp scan, we need to make sure all threads in the warp participate,
        // so we divide the run into stages.
        auto const num_rounds = gqe::utility::divide_round_up(bitpacked_length, warp_size);
        for (int32_t round_idx = 0; round_idx < num_rounds; round_idx++) {
          // The bit offset in <bit-packed-values> the current thread is decoding
          // Equivalently, this is the output offset for the current bitpacked run
          auto const bit_idx         = round_idx * warp_size + warp_lane;
          auto const byte_idx        = bit_idx / 8;
          auto const bit_idx_in_byte = bit_idx - byte_idx * 8;

          // `mask` will be either 0 or 1, indicating whether the current value is NULL
          uint8_t mask = 0;
          if (byte_idx < length) { mask = (level_ptr[byte_idx] >> bit_idx_in_byte) & 0x1; }

          // Calculate the input location by a prefix sum on `mask`
          uint8_t input_offset;
          uint8_t input_total;
          WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, input_offset, input_total);

          // Copy from the value section in the Parquet page to the output column
          if (mask) {
            set_value(type_id,
                      output_data[page_idx],
                      row_offset + bit_idx,
                      value_ptr + input_offset * type_size,
                      type_size);
          }

          value_ptr += input_total * type_size;
        }

        // For a flat data type, the NULL mask in cuDF is the same as the bitpacked definition
        // level, so we can directly copy the bits.
        copy_bits(group, output_bitmask[page_idx], row_offset, level_ptr, length);

        level_ptr += length;
        row_offset += bitpacked_length;
      } else {
        // The current run is a RLE run
        // For a flat data type, each value has 1 bit stored in the definition level: 0 or 1. So the
        // repeated value must be either 0 or 1. If the repeated value is 0, it means we have
        // repeated NULLs in the column. Since the NULL mask is initialized to 0, we do not need to
        // do anything. If the repeated value is 1, we need to copy the data, and set the null mask
        // to 1.
        auto const repeated_value = *level_ptr;

        if (repeated_value) {
          for (int32_t out_idx = group.thread_rank(); out_idx < length;
               out_idx += group.num_threads()) {
            set_value(type_id,
                      output_data[page_idx],
                      row_offset + out_idx,
                      value_ptr + out_idx * type_size,
                      type_size);
          }

          set_bits(group, output_bitmask[page_idx], row_offset, length);

          value_ptr += length * type_size;
        }

        level_ptr++;
        row_offset += length;
      }
    }
  }
}

void io_batch::decode(gqe::utility::bandwidth_timer& decode_timer, rmm::cuda_stream_view stream)
{
  utility::nvtx_scoped_range range("io::decode");
  decode_timer.start();
  std::vector<void const*> page_data;
  std::vector<void*> output_data;
  std::vector<cudf::bitmask_type*> output_bitmask;
  std::vector<int32_t> physical_type_sizes;
  std::vector<cudf::type_id> logical_type_ids;
  std::vector<cudf::size_type> row_idx;
  int32_t num_pages          = 0;
  int64_t total_encoded_size = 0;

  for (column_chunk_info& column_chunk : _column_chunks) {
    auto out_column_base_ptr  = column_chunk.out_column_base_ptr;
    auto out_bitmask_base_ptr = column_chunk.out_bitmask_base_ptr;
    auto const type           = column_chunk.type;

    for (auto const& page : column_chunk.pages) {
      page_data.push_back(page.uncompressed_ptr);
      output_data.push_back(out_column_base_ptr);
      output_bitmask.push_back(out_bitmask_base_ptr);
      physical_type_sizes.push_back(type.physical_size());
      logical_type_ids.push_back(type.logical_type.id());
      row_idx.push_back(page.row_idx);
      num_pages++;
      total_encoded_size += page.uncompressed_page_size;
    }
  }

  auto page_data_device           = copy_to_device(page_data, stream);
  auto output_data_device         = copy_to_device(output_data, stream);
  auto output_bitmask_device      = copy_to_device(output_bitmask, stream);
  auto physical_type_sizes_device = copy_to_device(physical_type_sizes, stream);
  auto logical_type_ids_device    = copy_to_device(logical_type_ids, stream);
  auto row_idx_device             = copy_to_device(row_idx, stream);

  constexpr int32_t block_size = 128;
  constexpr int32_t warp_size  = 32;
  auto const num_blocks        = gqe::utility::divide_round_up(num_pages, block_size / warp_size);

  decode_pages_kernel<block_size>
    <<<num_blocks, block_size, 0, stream.value()>>>(page_data_device->data(),
                                                    output_data_device->data(),
                                                    output_bitmask_device->data(),
                                                    physical_type_sizes_device->data(),
                                                    logical_type_ids_device->data(),
                                                    row_idx_device->data(),
                                                    num_pages);

  stream.synchronize();
  decode_timer.add(total_encoded_size, num_blocks);
  decode_timer.end();
}

}  // namespace

table_with_metadata read_parquet_custom(std::vector<std::string> file_paths,
                                        std::vector<std::string> column_names,
                                        void* bounce_buffer,
                                        int64_t bounce_buffer_size,
                                        std::size_t num_auxiliary_threads,
                                        std::size_t block_size,
                                        io_engine_type engine,
                                        bool pipelining,
                                        std::size_t alignment,
                                        gqe::utility::bandwidth_timer& disk_timer,
                                        gqe::utility::bandwidth_timer& h2d_timer,
                                        gqe::utility::bandwidth_timer& decomp_timer,
                                        gqe::utility::bandwidth_timer& decode_timer,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  const auto num_out_columns = column_names.size();

  // Allocate device bounce buffer
  rmm::device_buffer device_bounce_buffer(bounce_buffer_size, stream, mr);
  stream.synchronize();

  // Keep track of the opened file descriptors
  std::vector<int> fds, direct_fds;

  // Factory for thrift compact protocol used for decoding file metadata
  apache::thrift::protocol::TCompactProtocolFactoryT<apache::thrift::transport::TMemoryBuffer>
    thrift_compact_protocol_factory;

  // List of column chunks in the Parquet files
  std::vector<column_chunk_info> column_chunks_info;

  // Column types corresponding to `column_names`
  std::vector<std::optional<column_data_type>> column_types(num_out_columns);

  // Start row index of the row group being processed, invariant in the following loop
  cudf::size_type row_group_row_index = 0;

  // Number of rows in each file
  std::vector<cudf::size_type> rows_per_file;
  rows_per_file.reserve(file_paths.size());

  for (auto const& file_path : file_paths) {
    // Open the Parquet file and get the file descriptor
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Cannot open fd of column chunk: " + std::string(strerror(errno)));
    }
    fds.push_back(fd);

    int direct_fd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (direct_fd == -1) {
      throw std::runtime_error("Cannot open direct fd of column chunk: " +
                               std::string(strerror(errno)));
    }
    direct_fds.push_back(direct_fd);

    // Retrieve the footer length, which is a 4B integer at 8B before the end of the file
    uint32_t footer_length;
    if (lseek(fd, -8, SEEK_END) == -1) {
      throw std::runtime_error("Cannot seek FD: " + std::string(strerror(errno)));
    }
    if (read(fd, &footer_length, 4) == -1) {
      throw std::runtime_error(std::string("Cannot get Parquet footer length: ") + strerror(errno));
    }

    // Copy the footer from file to memory
    std::vector<uint8_t> footer(footer_length);
    if (lseek(fd, -8 - static_cast<int64_t>(footer_length), SEEK_END) == -1) {
      throw std::runtime_error("Cannot seek FD: " + std::string(strerror(errno)));
    }
    if (read(fd, footer.data(), footer_length) != footer_length) {
      throw std::runtime_error("Cannot read footer: " + std::string(strerror(errno)));
    }

    lseek(direct_fd, lseek(fd, 0, SEEK_CUR), SEEK_SET);

    // GQE_LOG_INFO("fd off={} direct off={}", lseek(fd, 0, SEEK_CUR), lseek(direct_fd, 0,
    // SEEK_CUR));
    assert(lseek(fd, 0, SEEK_CUR) == lseek(direct_fd, 0, SEEK_CUR));

    // Decode the footer to get the file metadata
    auto footer_transport = std::make_shared<apache::thrift::transport::TMemoryBuffer>(
      footer.data(), footer_length, apache::thrift::transport::TMemoryBuffer::OBSERVE);
    auto footer_protocol = thrift_compact_protocol_factory.getProtocol(footer_transport);

    parquet::FileMetaData file_metadata;
    file_metadata.read(footer_protocol.get());

    rows_per_file.push_back(file_metadata.num_rows);

    // Map from the column name to column index
    std::map<std::string, std::size_t> column_name_to_idx;

    // Skip the first element since it is always the root.
    for (std::size_t column_idx = 1; column_idx < file_metadata.schema.size(); column_idx++) {
      if (!file_metadata.schema[column_idx].__isset.type ||
          file_metadata.schema[column_idx].__isset.num_children) {
        throw unsupported_error("Customized Parquet reader: Only flat types are supported");
      }
      auto const column_name = file_metadata.schema[column_idx].name;
      if (column_name_to_idx.find(column_name) != column_name_to_idx.end()) {
        throw std::logic_error(
          "Customized Parquet reader: Duplicated column names in a Parquet file");
      }
      // Warning: Does the index in row_group.columns start from 0 (i.e, excludes the root), or
      // start from 1 (i.e. includes the root)? Assume it excludes the root for now, so we shift the
      // index by 1.
      column_name_to_idx[column_name] = column_idx - 1;
    }

    // Get the column index in the Parquet file corresponding to each column in `column_names`
    std::vector<int64_t> column_indices_parquet;
    for (auto const& column_name : column_names) {
      auto const idx_iter = column_name_to_idx.find(column_name);
      if (idx_iter == column_name_to_idx.end()) {
        throw std::runtime_error("Customized Parquet reader: Cannot find column " + column_name +
                                 " in Parquet file " + file_path);
      }
      column_indices_parquet.push_back(idx_iter->second);
    }

    // Get the column data types recorded in the Parquet file corresponding to `column_names`
    for (std::size_t out_column_idx = 0; out_column_idx < num_out_columns; out_column_idx++) {
      // Add 1 to account for the root schema element
      auto const& schema_element = file_metadata.schema[column_indices_parquet[out_column_idx] + 1];

      column_data_type column_type;
      column_type.physical_type = schema_element.type;
      column_type.type_length   = schema_element.type_length;

      if (schema_element.__isset.logicalType) {
        column_type.logical_type =
          parquet_to_logical_type(schema_element.logicalType, schema_element.type);
      } else {
        // The customized Parquet reader does not support Parquet files with converted_type but not
        // logicalType. converted_type is deprecated by the Parquet format.
        if (schema_element.__isset.converted_type) {
          throw unsupported_error(
            "Customized Parquet reader: Only supports logicalType instead of converted_type");
        }
        // If the logicalType is not set, we use the physical type as the logical type
        column_type.logical_type = physical_to_logical_type(column_type.physical_type);
      }

      switch (schema_element.repetition_type) {
        case parquet::FieldRepetitionType::REQUIRED:
          column_type.nullable = false;
          throw unsupported_error("Non-nullable column is not supported");
          break;
        case parquet::FieldRepetitionType::OPTIONAL: column_type.nullable = true; break;
        default: throw unsupported_error("Customized Parquet reader: Unsupported repetition type");
      }

      if (column_types[out_column_idx].has_value()) {
        if (column_types[out_column_idx] != column_type) {
          throw std::logic_error("Customized Parquet reader: Inconsistent data types among files");
        }
      } else {
        column_types[out_column_idx] = std::move(column_type);
      }
    }

    for (auto const& row_group : file_metadata.row_groups) {
      for (std::size_t out_column_idx = 0; out_column_idx < num_out_columns; out_column_idx++) {
        auto const& column_chunk = row_group.columns[column_indices_parquet[out_column_idx]];
        auto const& meta_data    = column_chunk.meta_data;

        column_chunk_info column_info;
        column_info.fd                   = fd;
        column_info.direct_fd            = direct_fd;
        column_info.file_offset          = meta_data.__isset.dictionary_page_offset
                                             ? meta_data.dictionary_page_offset
                                             : meta_data.data_page_offset;
        column_info.compressed_size      = meta_data.total_compressed_size;
        column_info.compression_codec    = meta_data.codec;
        column_info.host_ptr             = nullptr;
        column_info.device_ptr           = nullptr;
        column_info.out_column_idx       = out_column_idx;
        column_info.out_column_base_ptr  = nullptr;
        column_info.out_bitmask_base_ptr = nullptr;
        column_info.type                 = column_types[out_column_idx].value();
        column_info.row_idx              = row_group_row_index;
        column_info.num_rows             = row_group.num_rows;

        column_chunks_info.push_back(std::move(column_info));
      }

      row_group_row_index += row_group.num_rows;
    }
  }

  auto const result_num_rows = row_group_row_index;

  // Allocate the output columns in the device memory
  std::vector<std::unique_ptr<cudf::column>> out_columns;
  for (auto const& column_type : column_types) {
    assert(column_type.has_value());

    out_columns.push_back(cudf::make_fixed_width_column(
      column_type->logical_type,
      result_num_rows,
      column_type->nullable ? cudf::mask_state::ALL_NULL : cudf::mask_state::UNALLOCATED,
      stream,
      mr));
  }

  for (auto& column_chunk : column_chunks_info) {
    auto const mutable_view = out_columns[column_chunk.out_column_idx]->mutable_view();

    column_chunk.out_column_base_ptr  = mutable_view.head();
    column_chunk.out_bitmask_base_ptr = mutable_view.null_mask();
  }

  // Group column chunks into batches
  std::vector<io_batch> batches;
  batches.emplace_back(bounce_buffer, device_bounce_buffer.data(), bounce_buffer_size);

  for (auto& column_chunk : column_chunks_info) {
    if (column_chunk.compressed_size > bounce_buffer_size) {
      throw unsupported_error(
        "Customized Parquet reader: Bounce buffer is too small for the column chunk");
    }

    if (!batches.back().try_add(column_chunk)) {
      // If the GPU-accessible bounce buffer does not have enough space remaining, we will place the
      // current column chunk into the next batch.
      batches.emplace_back(bounce_buffer, device_bounce_buffer.data(), bounce_buffer_size);
      [[maybe_unused]] auto const status = batches.back().try_add(column_chunk);
      assert(status);
    }
  }

  for (auto& batch : batches) {
    batch.execute(num_auxiliary_threads,
                  block_size,
                  engine,
                  pipelining,
                  alignment,
                  disk_timer,
                  h2d_timer,
                  decomp_timer,
                  decode_timer,
                  stream);
  }

  // Since we change the bitmask during the page decoding, we need to rebuild the null count
  for (auto& column : out_columns) {
    auto const num_nulls = column->view().null_count(0, result_num_rows);
    column->set_null_count(num_nulls);
  }

  // Close all opened file descriptors
  for (auto const& fd : fds) {
    if (close(fd) == -1) {
      throw std::runtime_error("Cannot close fd of column chunk: " + std::string(strerror(errno)));
    }
  }
  for (auto const& direct_fd : direct_fds) {
    if (close(direct_fd) == -1) {
      throw std::runtime_error("Cannot close direct fd of column chunk: " +
                               std::string(strerror(errno)));
    }
  }

  table_with_metadata result;
  result.table         = std::make_unique<cudf::table>(std::move(out_columns));
  result.rows_per_file = std::move(rows_per_file);

  return result;
}

}  // namespace gqe::storage

#endif
