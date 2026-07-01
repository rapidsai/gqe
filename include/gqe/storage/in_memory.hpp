/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <gqe/utility/cuda.hpp>

#include <boost/container/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <deque>
#include <filesystem>
#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gqe {

namespace storage {
class in_memory_readable_view;
class in_memory_writeable_view;
class decompression_batch;
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

enum class in_memory_column_type { CONTIGUOUS, COMPRESSED_SLICED, SHARED_CONTIGUOUS };

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
  rmm::cuda_stream_view _allocation_stream;
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

/// Device span containing the bytes for one compressed-sliced partition.
using partition_span = cudf::device_span<uint8_t const>;

template <bool large_string_mode>
class string_compressed_sliced_column;

/**
 * @brief Compressed and partitioned in-memory column.
 */
class compressed_sliced_column : public column_base {
 protected:
  struct buffer_storage;

 public:
  /**
   * @brief Non-owning view over one partitioned buffer of a compressed-sliced column.
   *
   * The view can reference either the data buffer or an auxiliary buffer, such as string offsets.
   * It preserves the buffer's compression metadata so callers can copy uncompressed partitions
   * directly or schedule decompression of compressed partitions.
   */
  class buffer_view {
   public:
    /**
     * @brief Return true if the referenced partition buffer is compressed.
     */
    [[nodiscard]] bool is_compressed() const;

    /**
     * @brief Return the compression format used by the referenced partition buffer.
     */
    [[nodiscard]] gqe::compression_format compression_format() const;

    /**
     * @brief Return the element type represented by the referenced partition buffer.
     */
    [[nodiscard]] cudf::data_type element_type() const;

    /**
     * @brief Return a device span for one partition in the referenced buffer.
     *
     * @param[in] partition_idx Index of the partition to access.
     */
    [[nodiscard]] partition_span get_partition(size_t partition_idx) const;

   private:
    friend class compressed_sliced_column;
    template <bool>
    friend class string_compressed_sliced_column;

    /**
     * @brief Construct a view over one of a compressed-sliced column's partition buffers.
     *
     * @param[in] buffer Partition buffer storage to view.
     */
    explicit buffer_view(const buffer_storage* buffer) : _buffer(buffer) {}

    const buffer_storage* _buffer{nullptr};
  };

  compressed_sliced_column(cudf::column&& cudf_column,
                           int partition_size,
                           memory_kind::type memory_kind,
                           compression_configuration compression_config,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);

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
   * @brief Return a non-owning view of the partitioned data buffer.
   */
  [[nodiscard]] buffer_view data_buffer_view() const { return buffer_view(&_data_buffer); }

  /**
   * @brief Return the compression manager used by this column's partition buffers.
   */
  [[nodiscard]] compression_manager const& get_compression_manager() const
  {
    return _nvcomp_manager;
  }

  /**
   * @brief Write this fixed-width compressed sliced column to @p file_path.
   *
   * @param[in] column_name Column name used for logging.
   * @param[in] file_path Destination file path.
   * @param[in] row_group_index Row group index this column belongs to.
   * @param[in] stream CUDA stream for serialization work.
   * @throw std::runtime_error Serialization is not yet implemented.
   */
  void serialize_to_disk(std::string const& column_name,
                         std::filesystem::path const& file_path,
                         size_t row_group_index,
                         rmm::cuda_stream_view stream) const;

  /**
   * @brief Read a fixed-width compressed sliced column from disk.
   *
   * @param[in] file_path Path to the serialized column file.
   * @param[in] column_name Column name used for logging.
   * @param[in] row_group_index Row group index being loaded.
   * @param[in] stream CUDA stream for deserialization work.
   * @param[in] mr Device memory resource for allocations.
   */
  [[nodiscard]] static std::unique_ptr<compressed_sliced_column> deserialize_from_disk(
    std::filesystem::path const& file_path,
    std::string const& column_name,
    size_t row_group_index,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

 protected:
  /**
   * @brief Owning storage for the per-partition buffers and their compression metadata.
   */
  struct buffer_storage {
    /**
     * @brief Construct empty storage for an untyped buffer.
     */
    buffer_storage() = default;

    /**
     * @brief Construct empty storage for a typed buffer.
     *
     * @param[in] element_type Type of the values represented by this buffer.
     */
    explicit buffer_storage(cudf::data_type element_type) : element_type(element_type) {}

    /**
     * @brief Return true if partitions in this buffer are compressed.
     */
    [[nodiscard]] bool is_compressed() const;

    /**
     * @brief Return a device span for one partition in this buffer.
     *
     * @param[in] partition_idx Index of the partition to access.
     */
    [[nodiscard]] partition_span get_partition(size_t partition_idx) const;

    cudf::data_type element_type{cudf::data_type(cudf::type_id::EMPTY)};
    std::vector<std::unique_ptr<rmm::device_buffer>> buffers;
    gqe::compression_format compression_format{gqe::compression_format::none};
    size_t primary_compressed_size{0};
    size_t secondary_compressed_size{0};
    size_t uncompressed_size{0};
    size_t compressed_size{0};
  };

  size_t _size;
  size_t _partition_size;
  cudf::data_type _cudf_type;

  std::vector<cudf::size_type> _null_counts;
  compression_manager _nvcomp_manager;
  buffer_storage _data_buffer;
  buffer_storage _null_mask_buffer;

  // Protected constructor -- this is only called by derived classes (the string sliced column)
  // We fill the base members but don't do compression
  compressed_sliced_column(const cudf::column& cudf_column,
                           int partition_size,
                           compression_configuration compression_config);

  void compress(cudf::column&& cudf_column,
                memory_kind::type memory_kind,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * @brief Compress one buffer in the column (i.e. data or null mask)
   *
   * @param[in] input The input buffer to compress
   * @param[out] output The compressed buffer
   * @param[in] num_rows The number of rows in the column
   * @param[in] num_partitions The number of partitions in the column
   * @param[in] is_null_mask Whether the column is a null mask
   * @param[in] memory_kind Memory kind used for the compressed output buffers.
   * @param[in] stream The CUDA stream to use
   * @param[in] mr The memory resource to use
   */
  void do_compress(rmm::device_buffer const* input,
                   buffer_storage& output,
                   size_t num_rows,
                   size_t num_partitions,
                   bool is_null_mask,
                   memory_kind::type memory_kind,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr);

 private:
  /**
   * @brief Reconstruct a column from on-disk (used by @ref deserialize_from_disk).
   *
   * Config fields come from @p compression_config; per-buffer effective codecs come from the
   * `is_*_compressed` flags.
   */
  compressed_sliced_column(
    size_t size,
    size_t partition_size,
    cudf::data_type cudf_type,
    compression_configuration compression_config,
    bool is_compressed,
    bool is_null_mask_compressed,
    bool is_secondary_compressed,
    std::vector<cudf::size_type>&& compressed_data_sizes,
    std::vector<cudf::size_type>&& compressed_null_mask_sizes,
    std::vector<cudf::size_type>&& null_counts,
    std::vector<std::unique_ptr<rmm::device_buffer>>&& compressed_data_buffers,
    std::vector<std::unique_ptr<rmm::device_buffer>>&& compressed_null_masks);
};

/**
 * @brief Collects partition decompression requests and enqueues them sequentially.
 *
 * This is paired with @ref gqe::utility::copy_batch: compressed partition bytes are first copied
 * into staging buffers, then the decompression batch expands those staged bytes into final output
 * buffers.
 */
class decompression_batch {
 public:
  /**
   * @brief Add a decompression request to the batch.
   *
   * Empty partition lists are ignored.
   *
   * @param[in] host_source_buffer Source partition buffer view.
   * @param[in] manager Compression manager owned by the source column.
   * @param[in] partition_indexes Partition indices represented by @p device_staging_buffer.
   * @param[in] device_staging_buffer Temporary buffer containing compressed partition bytes.
   * @param[out] output_ptr Destination pointer for decompressed bytes.
   */
  void add(compressed_sliced_column::buffer_view host_source_buffer,
           compression_manager const* manager,
           std::vector<size_t> partition_indexes,
           rmm::device_buffer* device_staging_buffer,
           std::byte* output_ptr);

  /**
   * @brief Reserve storage for at least @p num_requests decompression requests.
   */
  void reserve(size_t num_requests);

  /**
   * @brief Return true if the batch contains no decompression requests.
   */
  [[nodiscard]] bool empty() const;

  /**
   * @brief Return the number of decompression requests in the batch.
   */
  [[nodiscard]] size_t size() const;

  /**
   * @brief Enqueue all queued decompressions asynchronously on a CUDA stream.
   *
   * This does not synchronize @p stream before returning; callers must synchronize or otherwise
   * order subsequent work before consuming the decompressed outputs.
   *
   * @param[in] stream CUDA stream used for decompression.
   */
  void execute_async(rmm::cuda_stream_view stream) const;

 private:
  /**
   * @brief Non-owning metadata for one partition decompression operation.
   */
  struct decompression_request {
    compressed_sliced_column::buffer_view host_source_buffer;
    compression_manager const* manager;
    std::vector<size_t> partition_indexes;
    rmm::device_buffer* device_staging_buffer;
    std::byte* output_ptr;
  };

  std::vector<decompression_request> _requests;
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
                                  compression_configuration compression_config,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

  /// @copydoc string_compressed_sliced_column_base::is_large_string
  virtual bool is_large_string() const override;

  /// Represent the size of the character offsets.
  using offsets_type = std::conditional_t<large_string_mode, int64_t, int32_t>;
  static constexpr cudf::data_type offset_element_type = large_string_mode
                                                           ? cudf::data_type(cudf::type_id::INT64)
                                                           : cudf::data_type(cudf::type_id::INT32);

  /**
   * @brief Compress the string column's character and offset buffers independently.
   *
   * @param[in] cudf_column String column to compress.
   * @param[in] memory_kind Memory kind used for compressed buffers.
   * @param[in] stream CUDA stream used for compression.
   * @param[in] mr Memory resource used for temporary and output allocations.
   */
  void compress(cudf::column&& cudf_column,
                memory_kind::type memory_kind,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

  /**
   * @brief Return a non-owning view of the partitioned character data buffer.
   */
  [[nodiscard]] buffer_view char_buffer_view() const { return data_buffer_view(); }

  /**
   * @brief Return the decompressed character byte count for one partition.
   *
   * @param[in] partition_idx Partition index to query.
   */
  [[nodiscard]] size_t get_char_partition_output_size(size_t partition_idx) const;

  /**
   * @brief Return a non-owning view of the partitioned string offset buffer.
   */
  [[nodiscard]] buffer_view offset_buffer_view() const { return buffer_view(&_offset_buffer); }

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
  buffer_storage _offset_buffer;
  std::vector<cudf::size_type> _partition_char_array_sizes;
};

extern template class string_compressed_sliced_column<false>;
extern template class string_compressed_sliced_column<true>;

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
   * @brief Create an in-memory table (empty row groups).
   *
   * To load serialized row-group snapshots from disk, call @ref
   * gqe::catalog::deserialize_table after registering the table.
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
   * @brief Load serialized row groups from disk under @p table_serialized_data_root (`zmps-*`).
   *
   * For each `rg-{i}/`, zone maps under `zone_maps/` are loaded before column snapshots under
   * `{compression}/{chunk}/`. Column order and names come from @ref _column_names. Replaces any
   * existing row groups.
   *
   * @throw std::runtime_error if loading any row group or column fails.
   */
  void deserialize_table_from_disk(std::string const& table_serialized_data_root,
                                   std::string const& table_name,
                                   rmm::cuda_stream_view stream);

  /**
   * @brief Serialize all row groups to disk under @p table_serialized_data_root (`zmps-*`).
   *
   * For each `rg-{i}/`, writes `zone_maps/` then column `.bin` / `.json` under
   * `{compression}/{chunk}/`. Column order and names come from @ref _column_names. Call after write
   * work on @p stream is complete. Uses a shared latch on the row-group container for the duration
   * of the pass.
   *
   * @throw std::runtime_error if any row group fails to serialize.
   */
  void serialize_table_to_disk(std::string const& table_serialized_data_root,
                               std::string const& table_name,
                               rmm::cuda_stream_view stream);

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
  std::vector<std::string> _column_names;
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
 * @brief Builds one output cuDF column for an in-memory read task.
 *
 * This is an abstract interface. Concrete subclasses implement layout-specific materialization by
 * overriding make_cudf_column() and submit_materialization_requests().
 *
 * Implementations queue any required partition copies and decompression requests, then construct a
 * cuDF column from the selected partitions.
 */
class output_column_builder {
 public:
  /**
   * @brief Construct a builder for one projected output column.
   *
   * @param[in] cudf_type Expected output cuDF data type.
   * @param[in] pruning_results Row groups and partitions selected by predicate pruning.
   * @param[in] column_idx Index of the source column within each row group.
   */
  output_column_builder(cudf::data_type cudf_type,
                        std::shared_ptr<pruning_results_t> pruning_results,
                        cudf::size_type column_idx);

  /**
   * @brief Destroy an output column builder.
   */
  virtual ~output_column_builder() = default;

  /**
   * @brief Queue materialization work required before constructing the output column.
   *
   * Builders for partitioned layouts append copy and decompression requests here. The default
   * implementation does nothing for builders that materialize data through another path.
   *
   * @param[in,out] c_batch Batched copy requests to append to.
   * @param[in,out] d_batch Batched decompression requests to append to.
   * @param[in] stream CUDA stream used for any allocations made while preparing materialization.
   */
  virtual void submit_materialization_requests(utility::copy_batch& c_batch,
                                               decompression_batch& d_batch,
                                               rmm::cuda_stream_view stream);

  /**
   * @brief Construct the final cuDF column for this output column.
   *
   * @param[in] concatenation_stream CUDA stream used for cuDF column construction or concatenation.
   * @return The materialized output column.
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
 * @brief Output column builder that constructs results with @ref cudf::concatenate.
 *
 * This fallback path handles layouts without a specialized partition-copy builder.
 */
class concatenating_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;
};

/**
 * @brief Output column builder for fixed-width contiguous columns.
 *
 * Selected partitions are copied directly into one contiguous output data buffer.
 *
 * @tparam T Source column type exposing a cuDF-compatible @c view() method.
 */
template <typename T>
class contiguous_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  /**
   * @copydoc gqe::storage::output_column_builder::submit_materialization_requests()
   */
  void submit_materialization_requests(utility::copy_batch& c_batch,
                                       decompression_batch& d_batch,
                                       rmm::cuda_stream_view stream) override;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  /**
   * @brief Allocate the output data buffer.
   *
   * @param[in] stream CUDA stream used for allocation.
   * @return The output data buffer.
   */
  [[nodiscard]] rmm::device_buffer allocate_output_buffer(rmm::cuda_stream_view stream);

  rmm::device_buffer _output_buffer;
};

/**
 * @brief Output column builder for fixed-width compressed-sliced columns.
 *
 * Selected uncompressed partitions are copied directly. Selected compressed partitions are copied
 * into staging buffers and decompressed into the final output buffer.
 */
class compressed_sliced_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  /**
   * @copydoc gqe::storage::output_column_builder::submit_materialization_requests()
   */
  void submit_materialization_requests(utility::copy_batch& c_batch,
                                       decompression_batch& d_batch,
                                       rmm::cuda_stream_view stream) override;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  /**
   * @brief Allocate the output data buffer.
   *
   * @param[in] stream CUDA stream used for allocation.
   * @return The output data buffer.
   */
  [[nodiscard]] rmm::device_buffer allocate_output_buffer(rmm::cuda_stream_view stream);

  rmm::device_buffer _output_buffer;
  std::deque<rmm::device_buffer> _staging_buffers;
};

/**
 * @brief Output column builder for compressed-sliced string columns.
 *
 * Character data and string offsets are materialized through separate partitioned buffers. The
 * builder then adjusts offsets so that the selected character partitions form one valid cuDF string
 * column.
 *
 * @tparam large_string_mode Whether the source string column uses 64-bit offsets.
 */
template <bool large_string_mode>
class string_compressed_sliced_output_column_builder : public output_column_builder {
 public:
  using output_column_builder::output_column_builder;
  using output_column_builder::operator=;

  static constexpr cudf::data_type offset_element_type = large_string_mode
                                                           ? cudf::data_type(cudf::type_id::INT64)
                                                           : cudf::data_type(cudf::type_id::INT32);

  using offsets_type = std::conditional_t<large_string_mode, int64_t, int32_t>;

  /**
   * @copydoc gqe::storage::output_column_builder::submit_materialization_requests()
   */
  void submit_materialization_requests(utility::copy_batch& c_batch,
                                       decompression_batch& d_batch,
                                       rmm::cuda_stream_view stream) override;

  /**
   * @copydoc gqe::storage::output_column_builder::make_cudf_column()
   */
  [[nodiscard]] std::unique_ptr<cudf::column> make_cudf_column(
    rmm::cuda_stream_view concatenation_stream) override;

 private:
  /**
   * @brief Allocate character and offset output buffers.
   *
   * @param[in] char_buffer_size Number of bytes in the output character buffer.
   * @param[in] offset_buffer_size Number of bytes in the output offset buffer.
   * @param[in] stream CUDA stream used for allocation.
   * @return Character and offset output buffers.
   */
  [[nodiscard]] std::pair<rmm::device_buffer, rmm::device_buffer> allocate_output_buffers(
    size_t char_buffer_size, size_t offset_buffer_size, rmm::cuda_stream_view stream);

  rmm::device_buffer _char_buffer;
  rmm::device_buffer _offset_buffer;
  std::deque<rmm::device_buffer> _char_staging_buffers;
  std::deque<rmm::device_buffer> _offset_staging_buffers;
};

extern template class string_compressed_sliced_output_column_builder<false>;
extern template class string_compressed_sliced_output_column_builder<true>;

/**
 * @brief Builds a cuDF table from row groups selected by an in-memory read task.
 *
 * The builder chooses a concrete implementation of the abstract output_column_builder interface per
 * projected column, batches partition copies, runs required decompression, and constructs the final
 * output table.
 */
class output_table_builder {
 public:
  /**
   * @brief Construct an output table builder.
   *
   * @param[in] ctx_ref Execution context used for streams, memory resources, and semaphores.
   * @param[in] column_indexes Projected source column indices.
   * @param[in] data_types Expected cuDF data types for the projected columns.
   * @param[in] pruning_results Row groups and partitions selected by predicate pruning.
   */
  output_table_builder(context_reference ctx_ref,
                       std::vector<cudf::size_type> column_indexes,
                       std::vector<cudf::data_type> data_types,
                       std::unique_ptr<pruning_results_t> pruning_results);

  /**
   * @brief Materialize the selected row-group partitions as a cuDF table.
   *
   * @return The output table containing the projected columns and selected rows.
   */
  [[nodiscard]] std::unique_ptr<cudf::table> build();

 private:
  context_reference _ctx_ref;
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
  std::shared_ptr<pruning_results_t> _pruning_results;
  std::vector<std::unique_ptr<output_column_builder>> _output_columns;
};

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
                      std::optional<arrow::compute::Expression> partial_filter = std::nullopt,
                      std::vector<std::shared_ptr<task>> subquery_tasks        = {},
                      bool force_zero_copy_disable                             = false);

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
  std::optional<arrow::compute::Expression> _partial_filter;
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
                       rmm::device_async_resource_ref non_owned_memory_resource,
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
  rmm::device_async_resource_ref _non_owned_memory_resource; /**< Non-owning reference to a memory
                                                                resource owned by
                                                                in_memory_table. */
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
  /// Returns the transformed expression; or std::nullopt, if:
  /// 1) the filter expression is nullptr
  /// 2) the filter expression cannot be transformed
  std::optional<arrow::compute::Expression> transform_partial_filter(
    gqe::expression* partial_filter);

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
