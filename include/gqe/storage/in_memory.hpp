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

#pragma once

#include <gqe/context_reference.hpp>
#include <gqe/executor/read.hpp>
#include <gqe/executor/write.hpp>
#include <gqe/optimizer/statistics.hpp>
#include <gqe/storage/compression.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/sliced_columns.cuh>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/storage/zone_map.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/types.hpp>

#include <boost/container/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <deque>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace gqe {

namespace storage {
class in_memory_readable_view;
class in_memory_writeable_view;
class row_group;

/**
 * @brief Indicate which partitions of an in-memory table remain after partition pruning and which
 * can be skipped.
 *
 * Depending on the column type, the projected columns of an in_memory_read_task compute the same
 * information over the list of partitions returned by the zone map evaluation. For example,
 * contiguous columns consolidate individual partitions into maximally large partitions, whereas
 * compressed sliced columns require individual partition indices. All column classes require the
 * number of rows remaining after partition pruning.
 *
 * This class precomputes these values, so that they are only computed once when accessed by
 * multiple columns.
 */
class pruning_result_t {
 public:
  explicit pruning_result_t(const std::vector<zone_map::partition>& partitions,
                            cudf::size_type partition_size);

  cudf::size_type partition_size() const;

  /// @brief Return all partitions which are not pruned.
  const std::vector<zone_map::partition>& candidate_partitions() const;

  /// @brief Return maximally consolidated partitions.
  const std::vector<zone_map::partition>& consolidated_partitions() const;

  /// @brief Return partition ids
  const std::vector<size_t>& partition_indexes() const;

  /// @brief Return the number of rows after pruning.
  cudf::size_type num_rows() const;

 private:
  std::vector<zone_map::partition> _partitions;
  cudf::size_type _partition_size;
  std::vector<zone_map::partition> _candidate_partitions;
  std::vector<zone_map::partition> _consolidated_partitions;
  std::vector<size_t> _partition_indexes;
  cudf::size_type _num_rows;
};

/// Pair a row group with its partition pruning result
using row_group_with_pruning_result_t = std::pair<const row_group*, pruning_result_t>;

/// Pruning results of all row groups of an in-memory table
using pruning_results_t = std::vector<row_group_with_pruning_result_t>;

enum class in_memory_column_type {
  CONTIGUOUS,
  COMPRESSED,
  COMPRESSED_SLICED,
  SHARED_CONTIGUOUS,
  SHARED_COMPRESSED
};

/**
 * @brief Column with boost data structures that can be safely allocated and
 * accessed on inter-process CPU shared memory.
 *
 * It currently only supports fixed width types and strings.
 */
class shared_column {
 public:
  explicit shared_column(cudf::column_view col,
                         boost::interprocess::managed_shared_memory& segment,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr);

  cudf::column_view view() const;

  ~shared_column();

  int64_t get_data_size() const;

  int64_t get_offsets_size() const;

  cudf::data_type get_type() const;

 private:
  using SharedColumnAllocator =
    boost::interprocess::allocator<shared_column,
                                   boost::interprocess::managed_shared_memory::segment_manager>;
  cudf::data_type _type{cudf::type_id::EMPTY};
  cudf::size_type _size{};
  boost::interprocess::offset_ptr<std::byte> _data{};                // pointer to device data
  boost::interprocess::offset_ptr<cudf::bitmask_type> _null_mask{};  // pointer to device null mask
  mutable cudf::size_type _null_count{};
  boost::container::vector<shared_column, SharedColumnAllocator> _children;
  rmm::device_async_resource_ref _mr;
  int64_t _data_size{};
  int64_t _offsets_size{};
  int64_t _null_mask_size{};
};

/**
 * @brief Table with boost data structures that can be safely allocated and
 * accessed on inter-process CPU shared memory.
 */
class shared_table {
 public:
  explicit shared_table(cudf::table_view table,
                        boost::interprocess::managed_shared_memory& segment,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

  cudf::table_view view() const;

  std::unique_ptr<cudf::table> copy_to_device(rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr);

 private:
  using SharedColumnAllocator =
    boost::interprocess::allocator<shared_column,
                                   boost::interprocess::managed_shared_memory::segment_manager>;

  boost::container::vector<shared_column, SharedColumnAllocator> _columns;
  size_t _size;
};

/**
 * @brief Abstract base column of an in-memory table.
 *
 * No guarantees are provided about the memory layout. Concrete implementations are free to choose
 * their own memory layout.
 */
class column_base {
 public:
  column_base()          = default;
  virtual ~column_base() = default;

  column_base(const column_base& other)      = delete;
  column_base& operator=(const column_base&) = delete;

  /**
   * @brief Return the type of the in-memory column.
   */
  [[nodiscard]] virtual in_memory_column_type type() const = 0;

  /**
   * @brief Return the column size.
   */
  [[nodiscard]] virtual int64_t size() const = 0;

  /**
   * @brief Return the null count of the column.
   */
  [[nodiscard]] virtual cudf::size_type null_count() const = 0;

  /**
   * @brief Return whether the column is compressed or not.
   */
  [[nodiscard]] virtual bool is_compressed() const = 0;

  /**
   * @brief Return the compression ratio of the column if it is compressed, otherwise return 1.0.
   */
  [[nodiscard]] virtual double get_compression_ratio() const
  {
    // Guard against division by zero.
    if (get_compressed_size() == 0 || !is_compressed()) { return 1.0; }
    return static_cast<double>(get_uncompressed_size()) /
           static_cast<double>(get_compressed_size());
  }

  /**
   * @brief Return the compressed size of the column in bytes if it is compressed, otherwise return
   * the uncompressed size.
   */
  [[nodiscard]] virtual int64_t get_compressed_size() const = 0;

  /**
   * @brief Return the uncompressed size of the column in bytes.
   */
  [[nodiscard]] virtual int64_t get_uncompressed_size() const = 0;

  /**
   * @brief Return the column compression statistics.
   */
  [[nodiscard]] virtual column_compression_statistics get_compression_stats() const = 0;
};

/**
 * @brief In-memory column with a contiguous memory layout.
 */
class contiguous_column : public column_base {
 public:
  explicit contiguous_column(cudf::column&& cudf_column);
  ~contiguous_column() override = default;

  contiguous_column(const contiguous_column&)            = delete;
  contiguous_column& operator=(const contiguous_column&) = delete;

  /**
   * @copydoc gqe::storage::type()
   */
  in_memory_column_type type() const override { return in_memory_column_type::CONTIGUOUS; }

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  int64_t size() const override;

  /**
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] virtual cudf::size_type null_count() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  [[nodiscard]] bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   */
  [[nodiscard]] int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   */
  [[nodiscard]] int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;

  /**
   * @brief Return a cuDF compatible column view.
   */
  cudf::column_view view() const;

  /**
   * @brief Return a cuDF compatible mutable column view.
   */
  cudf::mutable_column_view mutable_view();

 private:
  cudf::column _data;
};

/**
 * @brief Contiguous column in inter-process CPU shared memory.
 */
class shared_contiguous_column : public column_base {
 public:
  shared_contiguous_column(std::string column_name,
                           boost::interprocess::managed_shared_memory& segment)
    : _column_name(column_name), _segment(segment)
  {
  }

  ~shared_contiguous_column();

  shared_contiguous_column(const shared_contiguous_column&)            = delete;
  shared_contiguous_column& operator=(const shared_contiguous_column&) = delete;

  /**
   * @copydoc gqe::storage::type()
   */
  in_memory_column_type type() const override { return in_memory_column_type::SHARED_CONTIGUOUS; }

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  int64_t size() const override { return view().size(); }

  /**
   * @brief Return a cuDF compatible column view.
   */
  cudf::column_view view() const;

  /*
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] virtual cudf::size_type null_count() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  [[nodiscard]] bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   */
  [[nodiscard]] int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   */
  [[nodiscard]] int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;

 private:
  std::string _column_name;
  boost::interprocess::managed_shared_memory& _segment;
};

/**
 * @brief Compressed in-memory table.
 */
class compressed_column : public column_base {
  friend class shared_compressed_column_base;

 public:
  compressed_column(cudf::column&& cudf_column,
                    compression_format comp_format,
                    decompression_backend decompress_backend,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr,
                    int compression_chunk_size,
                    double compression_ratio_threshold,
                    bool use_cpu_compression  = false,
                    int compression_level     = 10,
                    std::string column_name   = "",
                    cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY});

  ~compressed_column() override = default;

  compressed_column(const compressed_column&)            = delete;
  compressed_column& operator=(const compressed_column&) = delete;

  /**
   * @copydoc gqe::storage::type()
   */
  in_memory_column_type type() const override { return in_memory_column_type::COMPRESSED; }

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  int64_t size() const override;

  /**
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] virtual cudf::size_type null_count() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  [[nodiscard]] bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   */
  [[nodiscard]] int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   */
  [[nodiscard]] int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;

  /**
   * @brief Compress the column.
   *
   * @param[in] stream CUDA stream used for the decompression.
   * @param[in] mr Memory resource used for allocating the decompressed column.
   */
  std::unique_ptr<rmm::device_buffer> compress(
    rmm::device_buffer const* input,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource(),
    bool is_null_mask                 = false);

  /**
   * @brief Decompress and construct an uncompressed version of the column.
   *
   * @param[in] stream CUDA stream used for the decompression.
   * @param[in] mr Memory resource used for allocating the decompressed column.
   */
  std::unique_ptr<cudf::column> decompress(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

 private:
  int64_t _size;
  int64_t _compressed_size;
  int64_t _uncompressed_size;
  int64_t _null_mask_compressed_size;
  int64_t _null_mask_uncompressed_size;
  cudf::data_type _cudf_type;
  cudf::size_type _null_count;
  compression_format _comp_format;
  bool _is_compressed;
  bool _is_null_mask_compressed;
  compression_manager _nvcomp_manager;

  std::unique_ptr<rmm::device_buffer> _compressed_data;
  std::unique_ptr<rmm::device_buffer> _compressed_null_mask;
  std::vector<std::unique_ptr<compressed_column>> _compressed_children;
};

// compressed_sliced_output_column_helper and the string version need to create temporary buffers
// into which the compressed data of each source column is copied. These are stored in a map.
using compression_buffer_map =
  std::unordered_map<const column_base*, std::unique_ptr<rmm::device_buffer>>;

/**
 * @brief Compressed (and sliced)in-memory table.
 */
class compressed_sliced_column : public column_base {
 public:
  compressed_sliced_column(cudf::column&& cudf_column,
                           int partition_size,
                           memory_kind::type memory_kind,
                           compression_format comp_format,
                           compression_format secondary_compression_format,
                           decompression_backend decompress_backend,
                           int compression_chunk_size,
                           double compression_ratio_threshold,
                           double secondary_compression_ratio_threshold,
                           double secondary_compression_multiplier_threshold,
                           bool use_cpu_compression,
                           int compression_level,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr,
                           std::string column_name   = "",
                           cudf::data_type cudf_type = cudf::data_type{cudf::type_id::EMPTY});

  ~compressed_sliced_column() override = default;

  compressed_sliced_column(const compressed_sliced_column&)            = delete;
  compressed_sliced_column& operator=(const compressed_sliced_column&) = delete;

  /**
   * @copydoc gqe::storage::column_base::type()
   */
  in_memory_column_type type() const override { return in_memory_column_type::COMPRESSED_SLICED; }

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  int64_t size() const override;

  /**
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] virtual cudf::size_type null_count() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  [[nodiscard]] bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   * Compute the compressed size for the column, considering the data and null mask.
   */
  [[nodiscard]] int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   * Compute the uncompressed size for the column, considering the data and null mask.
   */
  [[nodiscard]] int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;

  /**
   * @brief Fill the provided pointer and size arrays with the information of the copied buffers as
   * required by cudaMemcpyBatchAsync.
   *
   * If the column data is compressed, this method allocates a temporary decompression buffers and
   * sets up cudaMemcpyBatchAsync to copy data into the temporary buffer. Otherwise, data is copied
   * directly into the target buffer for the final cuDF column.
   *
   * @param[out] dst_ptrs The destination pointers.
   * @param[out] src_ptrs The source pointers.
   * @param[out] sizes The sizes of each copied buffer.
   * @param[out] compression_buffers Map holding temporary compression buffers of this output
   * column.
   * @param[in] pruning_result Indicates which partitions are pruned.
   * @param[in] target_ptr Pointer into the target buffer of the final cuDF column.
   * @param[in] stream CUDA stream on which to allocate temporary buffers.
   * @param[in] mr RMM resource from which to allocate temporary buffers.
   */
  void prepare_batched_memcpy(
    std::byte** dst_ptrs,
    std::byte** src_ptrs,
    size_t* sizes,
    compression_buffer_map& compression_buffers,
    const pruning_result_t& pruning_result,
    std::byte* target_ptr,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Decompress data into the target buffer of the final cuDF column.
   *
   * If the column is not compressed, this function does nothing.
   *
   * Otherwise, it decompresses the data contained in compression_buffer to target_ptr.
   *
   * @param[out] target_ptr Pointer into the target buffer of the final cuDF column.
   * @param[in] compression_buffer Buffer containing compressed data.
   * @param[in] pruning_result Indicates which partitions are pruned.
   * @param[in] stream CUDA stream on which to allocate temporary buffers.
   * @param[in] mr RMM resource from which to allocate temporary buffers.
   * @return True, if the column was compressed; false, otherwise.
   */
  bool decompress(std::byte* target_ptr,
                  rmm::device_buffer* compression_buffer,
                  const pruning_result_t& pruning_result,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource()) const;

 protected:
  size_t _size;
  size_t _compressed_size;
  size_t _uncompressed_size;
  size_t _null_mask_compressed_size;
  size_t _null_mask_uncompressed_size;
  size_t _partition_size;
  cudf::size_type _null_count;

  cudf::data_type _cudf_type;
  std::vector<cudf::size_type> _null_counts;
  compression_format _comp_format;
  bool _is_compressed;
  bool _is_secondary_compressed;
  bool _is_null_mask_compressed;
  compression_manager _nvcomp_manager;
  compression_manager _nvcomp_null_manager;

  // 1 compressed buffer and 1 compressed null mask per partition
  std::vector<std::unique_ptr<rmm::device_buffer>> _compressed_data_buffers;
  std::vector<std::unique_ptr<rmm::device_buffer>> _compressed_null_masks;
  std::vector<cudf::size_type> _compressed_data_sizes;
  std::vector<cudf::size_type> _compressed_null_mask_sizes;

  const compression_format _secondary_compression_format;
  const double _secondary_compression_ratio_threshold;
  const double _secondary_compression_multiplier_threshold;

  size_t _primary_compressed_size;
  size_t _secondary_compressed_size;
  size_t _null_mask_primary_compressed_size;
  size_t _null_mask_secondary_compressed_size;

  // Protected constructor -- this is only called by derived classes (the string sliced column)
  // We fill the base members but don't do compression
  compressed_sliced_column(const cudf::column& cudf_column,
                           int partition_size,
                           memory_kind::type memory_kind,
                           compression_format comp_format,
                           compression_format secondary_compression_format,
                           decompression_backend decompress_backend,
                           int compression_chunk_size,
                           double compression_ratio_threshold,
                           double secondary_compression_ratio_threshold,
                           double secondary_compression_multiplier_threshold,
                           bool use_cpu_compression,
                           int compression_level,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr,
                           std::string column_name,
                           cudf::data_type cudf_type);

  void compress(cudf::column&& cudf_column,
                memory_kind::type memory_kind,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * @brief Compress one buffer in the column (i.e. data or null mask)
   *
   * @param[in] input The input buffer to compress
   * @param[out] compressed_data_buffers The compressed data buffers
   * @param[out] compressed_sizes The compressed sizes
   * @param[in] num_rows The number of rows in the column
   * @param[in] num_partitions The number of partitions in the column
   * @param[out] is_compressed Whether the column is compressed
   * @param[out] compressed_size The compressed size in bytes
   * @param[out] uncompressed_size The uncompressed size in bytes
   * @param[out] primary_compressed_size The compressed size in bytes if primary compression is
   * applied
   * @param[out] secondary_compressed_size The compressed size in bytes if secondary compression is
   * applied
   * @param[in] is_null_mask Whether the column is a null mask
   * @param[in] stream The CUDA stream to use
   * @param[in] mr The memory resource to use
   */
  void do_compress(rmm::device_buffer const* input,
                   std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data_buffers,
                   std::vector<cudf::size_type>& compressed_sizes,
                   size_t num_rows,
                   size_t num_partitions,
                   bool& is_compressed,
                   size_t& compressed_size,
                   size_t& uncompressed_size,
                   size_t& primary_compressed_size,
                   size_t& secondary_compressed_size,
                   bool is_null_mask,
                   memory_kind::type memory_kind,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr);

  /**
   * @brief Fill the pointer and size arrays for a cudaMemcpyBatchAsync call.
   *
   * This method advances the pointers dst_ptrs, src_ptrs, and sizes.
   *
   * If the column is not compressed, the destination pointers are based off of target_ptr.
   * Otherwise, a new RMM buffer is created and inserted into compression_buffers and the
   * destination pointers are based off of this buffer.
   *
   * @param[in, out] dst_ptrs Pointer to memory block holding destination pointers.
   * @param[in, out] src_ptrs Pointer to memory block holding source pointers.
   * @param[in, out] sizes Pointer to memory block holding copy sizes.
   * @param[out] compression_buffers Map in which the temporary target buffer holding the compressed
   * data is stored.
   * @param[in] pruning_result Indicates which partitions are pruned.
   * @param[in] target_ptr Pointer into the final cuDF output column buffer.
   * @param[in] is_compressed Flag indicating whether the source buffers are compressed.
   * @param[in] compressed_data The source buffers.
   * @param[in] partition_sizes The size of the source buffers.
   * @param[in] stream CUDA stream on which to allocate temporary buffers.
   * @param[in] mr RMM resource from which to allocate temporary buffers.
   * @return The size of copied data in bytes.
   */
  size_t do_fill_memcpy_buffers(
    std::byte**& dst_ptrs,
    std::byte**& src_ptrs,
    size_t*& sizes,
    compression_buffer_map& compression_buffers,
    const pruning_result_t& pruning_result,
    std::byte* target_ptr,
    bool is_compressed,
    const std::vector<std::unique_ptr<rmm::device_buffer>>& compressed_data,
    const std::vector<cudf::size_type>& partition_sizes,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const;
};

/**
 * @brief Base class for @ref string_compressed_sliced_column_base.
 *
 * This class defines the method @ref is_large_string to determine at runtime, if the offset type of
 * the string column is 32-bit or 64-bit.
 */
class string_compressed_sliced_column_base : public compressed_sliced_column {
 public:
  using compressed_sliced_column::compressed_sliced_column;
  /**
   * @brief Determine if the offset type is 64-bit or 32-bit.
   * @return True, if the offset type is 64-bit; false, otherwise.
   */
  virtual bool is_large_string() const = 0;
};

/**
 * @brief Compressed (and sliced) in-memory string column.
 *
 * This is a compressed (and sliced) in-memory column that contains a string column.
 * It is used to store string columns in a compressed and sliced format.
 *
 * This is different because the char (data) array / offset need to be compressed separately
 * Under decompression the offsets also need to be adjusted to point correctly to the reduced char
 * array
 *
 * Partition size must be a multiple of 32
 *
 * @tparam large_string_mode Whether the string column is large (i.e. int64_t offsets)
 */
template <bool large_string_mode>
class string_compressed_sliced_column : public string_compressed_sliced_column_base {
 public:
  string_compressed_sliced_column(cudf::column&& cudf_column,
                                  int partition_size,
                                  memory_kind::type memory_kind,
                                  compression_format comp_format,
                                  compression_format secondary_compression_format,
                                  decompression_backend decompress_backend,
                                  int compression_chunk_size,
                                  double compression_ratio_threshold,
                                  double secondary_compression_ratio_threshold,
                                  double secondary_compression_multiplier_threshold,
                                  bool use_cpu_compression,
                                  int compression_level,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr,
                                  std::string column_name);

  /// @copydoc string_compressed_sliced_column_base::is_large_string
  virtual bool is_large_string() const override;

  /// Represent the size of the character offsets.
  using offsets_type = std::conditional_t<large_string_mode, int64_t, int32_t>;
  static constexpr cudf::data_type offset_element_type = large_string_mode
                                                           ? cudf::data_type(cudf::type_id::INT64)
                                                           : cudf::data_type(cudf::type_id::INT32);

  void compress(cudf::column&& cudf_column,
                memory_kind::type memory_kind,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * Determine the size of the character and offset buffers required by this column.
   * @param pruning_result Indicates which partitions are pruned.
   * @return A pair containing the size of the character buffer and the offset buffer.
   */
  std::pair<size_t, size_t> buffer_sizes(const pruning_result_t& pruning_result) const;

  /**
   * @brief Fill the provided pointer and size arrays with the information of the copied buffers as
   * required by cudaMemcpyBatchAsync.
   *
   * If the character data is compressed, this method allocates a temporary decompression buffers
   * and sets up cudaMemcpyBatchAsync to copy data into the temporary buffer. Otherwise, data is
   * copied directly into the target buffer for the final cuDF column.
   *
   * The offset buffer is handled in the same way.
   *
   * @param[out] dst_ptrs The destination pointers.
   * @param[out] src_ptrs The source pointers.
   * @param[out] sizes The sizes of each copied buffer.
   * @param[out] char_compression_buffers Map holding temporary compression buffers for the char
   * column.
   * @param[out] offset_compression_buffers Map holding temporary compression buffers for the offset
   * column.
   * @param[in] pruning_result Indicates which partitions are pruned.
   * @param[in] char_target_ptr Pointer into the character buffer of the final cuDF column.
   * @param[in] offset_target_ptr Pointer into the offset buffer of the final cuDF column.
   * @param[in] stream CUDA stream on which to allocate temporary buffers.
   * @param[in] mr RMM resource from which to allocate temporary buffers.
   */
  std::tuple<size_t, size_t, size_t> prepare_batched_memcpy(
    std::byte** dst_ptrs,
    std::byte** src_ptrs,
    size_t* sizes,
    compression_buffer_map& char_compression_buffers,
    compression_buffer_map& offset_compression_buffers,
    const pruning_result_t& pruning_result,
    std::byte* char_target_ptr,
    std::byte* offset_target_ptr,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Decompress data into the target buffer of the final cuDF column.
   *
   * If the column is not compressed, this function does nothing. Otherwise, it decompresses the
   * data contained in compression_buffer to target_ptr. The offset buffer is handled in the same
   * way.
   *
   * @param[out] char_target_ptr Pointer into the character buffer of the final cuDF column.
   * @param[out] offset_target_ptr Pointer into the offset buffer of the final cuDF column.
   * @param[in] char_compression_buffer Buffer containing compressed character data.
   * @param[in] offset_compression_buffer Buffer containing compressed offset data.
   * @param[in] pruning_result Indicates which partitions are pruned.
   * @param[in] stream CUDA stream on which to allocate temporary buffers.
   * @param[in] mr RMM resource from which to allocate temporary buffers.
   * @return A std::pair indicating whether the column was decompressed or not, and the size of the
   * decompressed char buffer.
   */
  std::pair<bool, size_t> decompress(
    std::byte* char_target_ptr,
    std::byte* offset_target_ptr,
    rmm::device_buffer* char_compression_buffer,
    rmm::device_buffer* offset_compression_buffer,
    const pruning_result_t& pruning_result,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource()) const;

  /**
   * @brief Determine the row and char offsets of each copied partition.
   * @param[out] partition_char_offsets Character-buffer base offset of each copied partition.
   * @param[out] partition_row_offsets Row-offset base of each copied partition in the concatenated
   * output.
   * @param[in,out] char_offset The current character-buffer offset.
   * @param[in,out] row_offset The current row offset in the concatenated output.
   * @param[in] pruning_result Indicates which partitions are pruned.
   * @param[in] partition_offset_idx The index at which to start writing partition metadata.
   */
  void fill_partition_offsets(offsets_type* partition_char_offsets,
                              cudf::size_type* partition_row_offsets,
                              offsets_type& char_offset,
                              cudf::size_type& row_offset,
                              const pruning_result_t& pruning_result,
                              size_t partition_offset_idx) const;
  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   * Compute the compressed size for the column, considering the data, offsets, and null mask.
   */
  [[nodiscard]] int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   * Compute the uncompressed size for the column, considering the data, offsets, and null mask.
   */
  [[nodiscard]] int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   * Return true if either the data or offsets are compressed.
   */
  [[nodiscard]] bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  [[nodiscard]] column_compression_statistics get_compression_stats() const override;

 private:
  std::vector<std::unique_ptr<rmm::device_buffer>> _compressed_offset_partitions;
  std::vector<cudf::size_type> _compressed_offset_sizes;
  std::vector<cudf::size_type> _partition_char_array_sizes;
  std::vector<cudf::size_type> _partition_row_counts;

  bool _offsets_are_compressed;
  size_t _offsets_compressed_size;
  size_t _offsets_uncompressed_size;
  bool _offsets_are_secondary_compressed;
  double _offsets_compression_ratio;

  size_t _offsets_primary_compressed_size;
  size_t _offsets_secondary_compressed_size;

  static constexpr nvcompType_t offset_nvcomp_data_type =
    large_string_mode ? NVCOMP_TYPE_LONGLONG : NVCOMP_TYPE_INT;
};

extern template class string_compressed_sliced_column<false>;
extern template class string_compressed_sliced_column<true>;

/**
 * @brief Compressed column with boost data structures that can be safely
 * allocated and accessed on inter-process CPU shared memory.
 */
class shared_compressed_column_base {
 public:
  shared_compressed_column_base(gqe::storage::compressed_column&& compressed_column,
                                boost::interprocess::managed_shared_memory& segment,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr);

  std::unique_ptr<cudf::column> decompress(rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr);

  shared_compressed_column_base(shared_compressed_column_base&&) noexcept            = default;
  shared_compressed_column_base& operator=(shared_compressed_column_base&&) noexcept = default;
  shared_compressed_column_base(const shared_compressed_column_base&)                = delete;
  shared_compressed_column_base& operator=(const shared_compressed_column_base&)     = delete;

  ~shared_compressed_column_base();

 public:
  using SharedColumnAllocator =
    boost::interprocess::allocator<shared_compressed_column_base,
                                   boost::interprocess::managed_shared_memory::segment_manager>;
  int64_t _size;
  int64_t _compressed_size;
  int64_t _uncompressed_size;
  int64_t _null_mask_compressed_size;
  int64_t _null_mask_uncompressed_size;
  cudf::data_type _cudf_type;
  cudf::size_type _null_count;
  compression_format _comp_format;
  bool _is_compressed;
  bool _is_null_mask_compressed;
  compression_manager _nvcomp_manager;

  boost::interprocess::offset_ptr<void> _compressed_data;
  boost::interprocess::offset_ptr<void> _compressed_null_mask;
  boost::container::vector<shared_compressed_column_base, SharedColumnAllocator>
    _compressed_children;

  rmm::device_async_resource_ref _mr;
};

/**
 * @brief Compressed in-memory table in inter-process CPU shared memory.
 */
class shared_compressed_column : public column_base {
 public:
  explicit shared_compressed_column(std::string name,
                                    boost::interprocess::managed_shared_memory& segment);

  ~shared_compressed_column() override;

  shared_compressed_column(const shared_compressed_column&)            = delete;
  shared_compressed_column& operator=(const shared_compressed_column&) = delete;

  /**
   * @copydoc gqe::storage::column_base::type()
   */
  in_memory_column_type type() const override { return in_memory_column_type::SHARED_COMPRESSED; }

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  int64_t size() const override;

  /**
   * @copydoc gqe::storage::column_base::null_count()
   */
  [[nodiscard]] virtual cudf::size_type null_count() const override;

  /**
   * @copydoc gqe::storage::column_base::is_compressed()
   */
  bool is_compressed() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compressed_size()
   */
  int64_t get_compressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_uncompressed_size()
   */
  int64_t get_uncompressed_size() const override;

  /**
   * @copydoc gqe::storage::column_base::get_compression_stats()
   */
  column_compression_statistics get_compression_stats() const override;

  /**
   * @brief Decompress and construct an uncompressed version of the column.
   *
   * @param[in] stream CUDA stream used for the decompression.
   * @param[in] mr Memory resource used for allocating the decompressed column.
   */
  std::unique_ptr<cudf::column> decompress(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

 private:
  std::string _name;
  boost::interprocess::managed_shared_memory& _segment;
};

/**
 * In-memory row group.
 */
class row_group {
 public:
  row_group()                  = default;
  row_group(row_group&& other) = default;

  explicit row_group(std::vector<std::unique_ptr<column_base>>&& columns);
  explicit row_group(std::vector<std::unique_ptr<column_base>>&& columns,
                     std::unique_ptr<gqe::zone_map> zone_map);

  row_group(const row_group& other)      = delete;
  row_group& operator=(const row_group&) = delete;

  virtual ~row_group() = default;

  /**
   * @brief Return the number of rows in the row group.
   */
  [[nodiscard]] int64_t size() const;

  [[nodiscard]] int64_t num_columns() const;

  /**
   * @brief Lookup a column by index.
   */
  [[nodiscard]] column_base& get_column(cudf::size_type column_index) const;

  gqe::zone_map* zone_map() const;

 private:
  std::vector<std::unique_ptr<column_base>> _columns;
  std::unique_ptr<gqe::zone_map> _zone_map;
};

/**
 * @brief A table that stores data in memory.
 *
 * The data in this table are represented as a memory location. This includes
 * GPU memory, CPU memory, and memory-mapped storage.
 *
 * == Memory Layout ==
 *
 * The table consists of multiple row groups, each of which contains a column
 * per table attribute. The table does not define the column's memory layout,
 * this is defined by the concrete column type.
 *
 * Row groups are stored in an extensible data structure that maintains reference
 * validity during append operations, e.g., a `std::deque`.
 *
 * == Memory Kinds ==
 *
 * The table provides a memory resource for new memory allocations. The memory
 * kind determines the type of memory resource used. Examples include a CUDA
 * device memory allocator and a NUMA-aware host memory allocator.
 *
 * Some memory kinds are directly accessible by GPU kernels, others are not.
 * Thus, the executor may need to, e.g., copy data to device memory before the
 * kernel reads the data. The data access method must be determined at runtime
 * depending on the memory kind and hardware coherency support.
 *
 * == Thread Safety ==
 *
 * The table guarantees atomicity of appends, and reference validity during
 * appends. The table ensures this by a RW latch on the row group data
 * structure, that allows either multiple readers or a single writer.
 */
class in_memory_table : public table {
 public:
  friend in_memory_readable_view;
  friend in_memory_writeable_view;

  /**
   * @brief A functor to append row groups.
   *
   * The row group appender is a separation of concerns: it provides a
   * thread-safe append operation, but not full access to the table.
   *
   * == Thread Safety ==
   *
   * The appender acquires exclusive access to the row group data structure for
   * the duration of the append operation.
   */
  class row_group_appender {
    friend class in_memory_table;

   public:
    /**
     * @brief Append a row group to the table.
     */
    void operator()(row_group&& new_row_group);

    /**
     * @brief Append multiple row groups to the table.
     *
     * This method avoids acquiring the exclusive access multiple times.
     */
    void operator()(std::vector<row_group>&& new_row_groups);

   private:
    row_group_appender(std::deque<row_group>* non_owning_row_groups,
                       std::shared_mutex* non_owning_row_group_latch);

    std::deque<row_group>* _non_owning_row_groups;
    std::shared_mutex* _non_owning_row_group_latch;
  };

  in_memory_table()                             = delete;
  in_memory_table(const in_memory_table& other) = delete;

  /**
   * @brief Create an in-memory table.
   *
   * @param[in] memory_kind The memory kind to allocate.
   * @param[in] column_names The name per column contained in the table.
   * @param[in] column_types The data type of each column.
   * @param[in] ctx Non-owning pointer to task manager context for accessing centralized
   *                memory resources. Required for boost_shared, numa, and numa_pinned memory kinds.
   */
  in_memory_table(memory_kind::type memory_kind,
                  std::vector<std::string> const& column_names,
                  std::vector<cudf::data_type> const& column_types,
                  task_manager_context* ctx);

  /**
   * @copydoc gqe::storage::table::is_readable()
   */
  [[nodiscard]] bool is_readable() const override;

  /**
   * @copydoc gqe::storage::table::is_writeable()
   */
  [[nodiscard]] bool is_writeable() const override;

  /**
   * @copydoc gqe::storage::table::max_concurrent_readers()
   */
  [[nodiscard]] int32_t max_concurrent_readers() const override;

  /**
   * @copydoc gqe::storage::table::max_concurrent_writers()
   */
  [[nodiscard]] int32_t max_concurrent_writers() const override;

  /**
   * @copydoc gqe::storage::table::readable_view()
   */
  std::unique_ptr<storage::readable_view> readable_view() override;

  /**
   * @copydoc gqe::storage::table::writeable_view()
   */
  std::unique_ptr<storage::writeable_view> writeable_view() override;

  /**
   * @brief Return the numeric index of a column referenced by its column name.
   *
   * @throw std::out_of_range exception if the column name does not exist.
   */
  cudf::size_type get_column_index(std::string const& column_name) const;

  row_group_appender get_row_group_appender();

 private:
  task_manager_context* _task_manager_context; /**< Non-owning pointer to the task manager context.
                                                    Used to access memory resources. */
  memory_kind::type _memory_kind;              /**< Memory kind of this table. */
  std::unordered_map<std::string, cudf::size_type> _column_name_to_index;
  std::vector<cudf::data_type> _column_types;
  std::deque<row_group>
    _row_groups; /** Non-owning references to row_group instances require a container that does not
                  * invalidate references to elements when resizing. `std::deque` guarantees
                  * reference validity when resizing, `std::vector` does not. */
  std::shared_mutex _row_group_latch; /**< An exclusive latch must be held while resizing the row
                                       * group vector. A shared latch must be held for read-only
                                       * container operations (e.g., while getting a reference to
                                       * an element, but not for accessing the element itself). */
};

/**
 * @brief Construct a cudf::column output column for the in_memory_read_task from the columns
 * contained in multiple row groups.
 *
 * This is a base class which defines the interface that is used in @ref
 * in_memory_read_task::emit_copied_result(). Subclasses implement this interface for specific
 * in-memory table column types (e.g., CONTIGUOUS or COMPRESSED_SLICED) or cuDF types (e.g.,
 * fixed-width or string).
 */
class output_column_helper {
 public:
  /**
   * @brief Create a helper to construct a CUDF column from the columns of multiple row groups.
   * @param cudf_type The cuDF type of the column.
   * @param pruning_results Partition pruning results
   * @param column_idx The index of the output column in the base table schema
   */
  output_column_helper(cudf::data_type cudf_type,
                       std::shared_ptr<pruning_results_t> pruning_results,
                       cudf::size_type column_idx);

  /// Virtual destructor
  virtual ~output_column_helper() = default;

  /**
   * @brief Determine the number of buffers copied by cudaMemcpyBatchAsync.
   * @param partition_size The number of rows in a partition used for pruning.
   * @return The number of copied buffers.
   */
  [[nodiscard]] virtual size_t num_copied_buffers([[maybe_unused]] size_t partition_size) = 0;

  /**
   * @brief Fill the provided pointer and size arrays with the information of the copied buffers as
   * required by cudaMemcpyBatchAsync.
   *
   * This method sets up the destination pointer so that they point into either (a) the target
   * buffer of the final cuDF column or (b) temporary decompression buffers.
   *
   * @param[out] dst_ptrs The destination pointers.
   * @param[out] src_ptrs The source pointers.
   * @param[out] sizes The sizes of each copied buffer.
   * @param[in] stream CUDA stream on which to allocate temporary buffers.
   */
  virtual void prepare_batched_memcpy(std::byte** dst_ptrs,
                                      std::byte** src_ptrs,
                                      size_t* sizes,
                                      rmm::cuda_stream_view stream) = 0;

  /**
   * @brief Decompress compressed columns, if necessary.
   *
   * Calls to nvCOMP should only occur in this method.
   *
   * @param decompression_stream The CUDA stream on which to execute operations.
   * @return True, if the column of any row group was compressed.
   */
  [[nodiscard]] virtual bool decompress_row_group_columns(
    rmm::cuda_stream_view decompression_stream) = 0;

  /**
   * @brief Construct the output CUDF column.
   *
   * Calls to kernels should only occur in this method.
   *
   * @param concatenation_stream The CUDA stream on which to execute kernels.
   * @return A CUDF column representing the output column of the read task.
   */
  [[nodiscard]] virtual std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) = 0;

 protected:
  const cudf::data_type _cudf_type;
  std::shared_ptr<pruning_results_t> _pruning_results;
  const cudf::size_type _column_idx;
  const size_t _num_rows;  //< Number of rows in the output column
};

/**
 * @brief Construct a cudf::column output column for the in_memory_read_task from the columns
 * contained in multiple row groups using @ref cudf::concatenate.
 *
 * Note: This class serves as a legacy implementation for column types for which there does not yet
 * exist a dedicated @ref output_column_helper.
 */
class concatenating_output_column_helper : public output_column_helper {
 public:
  using output_column_helper::output_column_helper;
  using output_column_helper::operator=;

  /// @copydoc gqe::storage::output_column_helper::num_copied_buffers
  [[nodiscard]] size_t num_copied_buffers(size_t partition_size) override;

  /// @copydoc gqe::storage::output_column_helper::prepare_batched_memcpy
  void prepare_batched_memcpy(std::byte** dst_ptrs,
                              std::byte** src_ptrs,
                              size_t* sizes,
                              rmm::cuda_stream_view stream) override;

  /// @copydoc gqe::storage::output_column_helper::decompress_row_group_columns
  [[nodiscard]] bool decompress_row_group_columns(
    rmm::cuda_stream_view decompression_stream) override;

  /// @copydoc gqe::storage::output_column_helper::make_cudf_column
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  /// Store decompressed columns between decompression and creation of the output CUDF column
  std::unordered_map<const column_base*, std::unique_ptr<cudf::column>> _decompressed_columns;
};

// TODO Do I really need the template? Not supporting SHARED_CONTIGUOUS for now.
template <typename T>
class contiguous_output_column_helper : public output_column_helper {
 public:
  using output_column_helper::output_column_helper;
  using output_column_helper::operator=;

  /// @copydoc gqe::storage::output_column_helper::num_copied_buffers
  [[nodiscard]] size_t num_copied_buffers(size_t partition_size) override;

  /// @copydoc gqe::storage::output_column_helper::prepare_batched_memcpy
  void prepare_batched_memcpy(std::byte** dst_ptrs,
                              std::byte** src_ptrs,
                              size_t* sizes,
                              rmm::cuda_stream_view stream) override;

  /// @copydoc gqe::storage::output_column_helper::decompress_row_group_columns
  [[nodiscard]] bool decompress_row_group_columns(
    rmm::cuda_stream_view decompression_stream) override;

  /// @copydoc gqe::storage::output_column_helper::make_cudf_column
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  std::unique_ptr<rmm::device_buffer> _output_buffer = nullptr;
};

class compressed_sliced_output_column_helper : public output_column_helper {
 public:
  using output_column_helper::output_column_helper;
  using output_column_helper::operator=;

  /// @copydoc gqe::storage::output_column_helper::num_copied_buffers
  [[nodiscard]] size_t num_copied_buffers(size_t partition_size) override;

  /// @copydoc gqe::storage::output_column_helper::prepare_batched_memcpy
  void prepare_batched_memcpy(std::byte** dst_ptrs,
                              std::byte** src_ptrs,
                              size_t* sizes,
                              rmm::cuda_stream_view stream) override;

  /// @copydoc gqe::storage::output_column_helper::decompress_row_group_columns
  [[nodiscard]] bool decompress_row_group_columns(
    rmm::cuda_stream_view decompression_stream) override;

  /// @copydoc gqe::storage::output_column_helper::make_cudf_column
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  std::unique_ptr<rmm::device_buffer> _output_buffer = nullptr;
  compression_buffer_map _compression_buffers;
};

template <bool large_string_mode>
class string_compressed_sliced_output_column_helper : public output_column_helper {
 public:
  using output_column_helper::output_column_helper;
  using output_column_helper::operator=;

  /// @copydoc gqe::storage::output_column_helper::num_copied_buffers
  [[nodiscard]] size_t num_copied_buffers(size_t partition_size) override;

  /// @copydoc gqe::storage::output_column_helper::prepare_batched_memcpy
  void prepare_batched_memcpy(std::byte** dst_ptrs,
                              std::byte** src_ptrs,
                              size_t* sizes,
                              rmm::cuda_stream_view stream) override;

  /// @copydoc gqe::storage::output_column_helper::decompress_row_group_columns
  [[nodiscard]] bool decompress_row_group_columns(
    rmm::cuda_stream_view decompression_stream) override;

  /// @copydoc gqe::storage::output_column_helper::make_cudf_column
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  static constexpr cudf::data_type offset_element_type = large_string_mode
                                                           ? cudf::data_type(cudf::type_id::INT64)
                                                           : cudf::data_type(cudf::type_id::INT32);

  using offsets_type = std::conditional_t<large_string_mode, int64_t, int32_t>;

  std::unique_ptr<rmm::device_buffer> _char_buffer   = nullptr;
  std::unique_ptr<rmm::device_buffer> _offset_buffer = nullptr;
  compression_buffer_map _char_compression_buffers;
  compression_buffer_map _offset_compression_buffers;
};

extern template class string_compressed_sliced_output_column_helper<false>;
extern template class string_compressed_sliced_output_column_helper<true>;

/**
 * @brief A read task for loading data from memory.
 */
class in_memory_read_task : public read_task_base {
 public:
  /**
   * @brief Construct an in-memory read task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] row_groups The row groups assigned to this task.
   * @param[in] column_indexes Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_names`.
   * @param[in] memory_kind The memory kind used by the input table.
   * @param[in] partial_filter Used to support predicate pushdown. Note that a row that satisfies
   * the predicate is guaranteed to be included in the loaded table, but a row that does not satisfy
   * the predicate may or may not be excluded. If such exclusion needs to be guaranteed, an extra
   * filter task is needed. If this argument is nullptr, no rows will be filtered out.
   * @param[in] subquery_tasks Subquery tasks that may be referenced by a subquery expression. A
   * relation index `i` in a subquery expression refers to `subquery_expressions[i]`.
   * @param[in] force_zero_copy_disable Override the zero-copy optimization parameter with a disable
   * command. Used, e.g., for unit testing or when zero-copy is unsafe due to the system
   * configuration.
   */
  in_memory_read_task(context_reference ctx_ref,
                      int32_t task_id,
                      int32_t stage_id,
                      std::vector<const row_group*> row_groups,
                      std::vector<cudf::size_type> column_indexes,
                      std::vector<cudf::data_type> data_types,
                      memory_kind::type memory_kind,
                      std::unique_ptr<gqe::expression> partial_filter   = nullptr,
                      std::vector<std::shared_ptr<task>> subquery_tasks = {},
                      bool force_zero_copy_disable                      = false);

  in_memory_read_task(const in_memory_read_task&)            = delete;
  in_memory_read_task& operator=(const in_memory_read_task&) = delete;

  void execute() override;

 private:
  /// Determine if partial pruning is supported. Returns true if partial pruning is enabled and a
  /// partial filter exists.
  bool can_prune_partitions() const;

  /// Determine the partitions of the table that should be emitted. If a partial filter exists,
  /// evaluate it on the zone map of on all row groups, and pair each row group with the evaluation
  /// result. If no partial filter exists, pair each row group with a partition which covers the
  /// entire row group.
  std::unique_ptr<pruning_results_t> evaluate_partial_filter();

  /// Determine if a zero-copy read is possible, based on the number of row group partitions and
  /// environment parameters.
  bool is_zero_copy_possible(const pruning_results_t& pruning_results) const;

  /// Construct an empty table with the schema of the read task and emit it as an owned result.
  void emit_empty_table();

  /// Emit a single row group partition as a borrowed result, which is accessed via zero-copy.
  void emit_zero_copy_result(std::unique_ptr<pruning_results_t> pruning_results);

  /// Copy one or more row groups to GPU memory and emit an owned result.
  void emit_copied_result(std::unique_ptr<pruning_results_t> pruning_results);

  std::vector<const row_group*>
    _row_groups; /**< Non-owning references to the row groups assigned to this task. */
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
  memory_kind::type _memory_kind;
  std::unique_ptr<gqe::expression> _partial_filter;
  bool _force_zero_copy_disable;
};

/**
 * @brief A write task for storing data to memory.
 */
class in_memory_write_task : public write_task_base {
 public:
  /**
   * @brief Construct an in-memory write task.
   *
   * @param[in] ctx_ref The context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input The input task emitting new data.
   * @param[in] non_owned_memory_resource A memory resource for allocating memory.
   * @param[in] appender An appender functor for adding row groups to the table.
   * @param[in] column_indexes Columns to be loaded.
   * @param[in] column_names Names of the columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_names`.
   * @param[in] statistics Statistics manager of the in-memory table
   */
  in_memory_write_task(context_reference ctx_ref,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<task> input,
                       rmm::mr::device_memory_resource* non_owned_memory_resource,
                       in_memory_table::row_group_appender appender,
                       memory_kind::type memory_kind,
                       std::vector<cudf::size_type> column_indexes,
                       std::vector<std::string> column_names,
                       std::vector<cudf::data_type> data_types,
                       table_statistics_manager* statistics);

  in_memory_write_task(const in_memory_write_task&)            = delete;
  in_memory_write_task& operator=(const in_memory_write_task&) = delete;

  void execute() override;
  void execute_default();
  void execute_shared_memory();

 private:
  rmm::mr::device_memory_resource*
    _non_owned_memory_resource; /**< Non-owning reference to a memory
                                   resource owned by in_memory_table. */
  in_memory_table::row_group_appender
    _appender; /**< Implicitly holds a non-owning reference to an in_memory_table */
  std::vector<cudf::size_type> _column_indexes;
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
  memory_kind::type _memory_kind;
  table_statistics_manager* _statistics;
};

/**
 * @brief Data access method to read an in-memory table.
 *
 * `get_read_tasks` acquires a shared read-only latch on the row groups
 * collection in the table for the duration of creating new read tasks.
 */
class in_memory_readable_view : public readable_view {
 public:
  friend in_memory_table;

  /**
   * @copydoc gqe::storage::readable_view::get_read_tasks()
   */
  std::vector<std::unique_ptr<read_task_base>> get_read_tasks(
    std::vector<readable_view::task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

 private:
  in_memory_readable_view(in_memory_table* non_owning_table);

  /// Transform a filter expression into an equivalent expression for the zone map.
  /// Returns the transformed expression; or nullptr, if:
  /// 1) the filter expression is nullptr
  /// 2) the filter expression cannot be transformed
  std::unique_ptr<gqe::expression> transform_partial_filter(gqe::expression* partial_filter);

  in_memory_table* _non_owning_table;
};

/**
 * @brief Data access method to write an in-memory table.
 */
class in_memory_writeable_view : public writeable_view {
 public:
  friend in_memory_table;

  /**
   * @copydoc gqe::storage::writeable_view::get_write_tasks()
   */
  std::vector<std::unique_ptr<write_task_base>> get_write_tasks(
    std::vector<writeable_view::task_parameters>&& task_parameters,
    context_reference ctx_ref,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types,
    table_statistics_manager* statistics) override;

 private:
  in_memory_writeable_view(in_memory_table* non_owning_table);

  in_memory_table* _non_owning_table;
};

}  // namespace storage

}  // namespace gqe
