/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <gqe/executor/read.hpp>
#include <gqe/executor/write.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/table.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/types.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
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

enum class in_memory_column_type { CONTIGUOUS, COMPRESSED };

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

  column_base(const column_base& other) = delete;
  column_base& operator=(const column_base&) = delete;

  /**
   * @brief Return the type of the in-memory column.
   */
  [[nodiscard]] virtual in_memory_column_type type() const = 0;

  /**
   * @brief Return the column size.
   */
  [[nodiscard]] virtual int64_t size() const = 0;
};

/**
 * @brief In-memory column with a contiguous memory layout.
 */
class contiguous_column : public column_base {
 public:
  explicit contiguous_column(cudf::column&& cudf_column);
  ~contiguous_column() override = default;

  contiguous_column(const contiguous_column&) = delete;
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
 * @brief Abstract base class representing a compressed buffer.
 */
class compressed_buffer {
 public:
  virtual ~compressed_buffer() = default;

  /**
   * @brief Decompress the data into a buffer in device-accessible memory.
   *
   * @param[in] stream CUDA stream used for the decompression.
   * @param[in] mr Memory resource used to allocate the decompressed buffer.
   */
  virtual std::unique_ptr<rmm::device_buffer> decompress(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;

  /**
   * @brief Return the compressed size of the buffer.
   */
  virtual std::size_t compressed_size() const = 0;
};

/**
 * @brief An implementation of `compressed_buffer` interface with plain encoding (i.e.,
 * uncompressed).
 */
class plain_buffer : public compressed_buffer {
 public:
  /**
   * @brief Construct a compressed buffer using plain encoding.
   *
   * @param[in] input Input buffer to be compressed.
   * @param[in] stream CUDA stream used for compression.
   * @param[in] mr Memory resource used to allocate the compressed buffer.
   */
  plain_buffer(rmm::device_buffer const* input,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr);

  /**
   * @copydoc gqe::storage::compressed_buffer::decompress(rmm::cuda_stream_view,
   * rmm::device_async_resource_ref)
   */
  std::unique_ptr<rmm::device_buffer> decompress(rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) const override;

  /**
   * @copydoc gqe::storage::compressed_size()
   */
  std::size_t compressed_size() const override { return _buffer->size(); }

 private:
  std::unique_ptr<rmm::device_buffer> _buffer;
};

/**
 * @brief A buffer compressed using Asymmetric Numeral Systems (ANS).
 */
class ans_compressed_buffer : public compressed_buffer {
 public:
  /**
   * @brief Construct a compressed buffer using ANS.
   *
   * @param[in] input Input buffer to be compressed.
   * @param[in] stream CUDA stream used for compression.
   * @param[in] mr Memory resource used to allocate the compressed buffer.
   */
  ans_compressed_buffer(rmm::device_buffer const* input,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr);

  /**
   * @copydoc gqe::storage::compressed_buffer::decompress(rmm::cuda_stream_view,
   * rmm::device_async_resource_ref)
   */
  std::unique_ptr<rmm::device_buffer> decompress(rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr) const override;

  /**
   * @copydoc gqe::storage::compressed_size()
   */
  std::size_t compressed_size() const override { return _compressed_buffer->size(); }

 private:
  std::unique_ptr<rmm::device_buffer> _compressed_buffer;
};

/**
 * @brief Compressed in-memory table.
 */
class compressed_column : public column_base {
 public:
  explicit compressed_column(cudf::column&& cudf_column,
                             compression_format comp_format,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr);

  ~compressed_column() override = default;

  compressed_column(const compressed_column&) = delete;
  compressed_column& operator=(const compressed_column&) = delete;

  /**
   * @copydoc gqe::storage::type()
   */
  in_memory_column_type type() const { return in_memory_column_type::COMPRESSED; }

  /**
   * @copydoc gqe::storage::column_base::size()
   */
  int64_t size() const override;

  /**
   * @brief Decompress and construct an uncompressed version of the column.
   *
   * @param[in] stream CUDA stream used for the decompression.
   * @param[in] mr Memory resource used for allocating the decompressed column.
   */
  std::unique_ptr<cudf::column> decompress(
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource()) const;

 private:
  int64_t _size;
  cudf::data_type _dtype;
  cudf::size_type _null_count;
  compression_format _comp_format;

  std::unique_ptr<compressed_buffer> _compressed_data;
  std::unique_ptr<compressed_buffer> _compressed_null_mask;
  std::vector<std::unique_ptr<compressed_column>> _compressed_children;
};

/**
 * In-memory row group.
 */
class row_group {
 public:
  row_group()                  = default;
  row_group(row_group&& other) = default;

  explicit row_group(std::vector<std::unique_ptr<column_base>>&& columns);

  row_group(const row_group& other) = delete;
  row_group& operator=(const row_group&) = delete;

  virtual ~row_group() = default;

  /**
   * @brief Return the number of rows in the row group.
   */
  [[nodiscard]] int64_t size() const;

  /**
   * @brief Lookup a column by index.
   */
  [[nodiscard]] column_base& get_column(cudf::size_type column_index) const;

 private:
  std::vector<std::unique_ptr<column_base>> _columns;
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
 * Row groups are stored in a extensible data structure that maintains reference
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
   */
  in_memory_table(memory_kind::type memory_kind,
                  std::vector<std::string> const& column_names,
                  std::vector<cudf::data_type> const& column_types);

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
  memory_kind::type _memory_kind; /**< Memory kind of this table. */
  std::unique_ptr<rmm::mr::device_memory_resource>
    _memory_resource; /**< Memory resource for allocating memory of the specified kind. */
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
 * @brief A read task for loading data from memory.
 */
class in_memory_read_task : public read_task_base {
 public:
  /**
   * @brief Construct an in-memory read task.
   *
   * @param[in] query_context The query context in which the current task is running in.
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
  in_memory_read_task(query_context* query_context,
                      int32_t task_id,
                      int32_t stage_id,
                      std::vector<const row_group*> row_groups,
                      std::vector<cudf::size_type> column_indexes,
                      std::vector<cudf::data_type> data_types,
                      memory_kind::type memory_kind,
                      std::unique_ptr<gqe::expression> partial_filter   = nullptr,
                      std::vector<std::shared_ptr<task>> subquery_tasks = {},
                      bool force_zero_copy_disable                      = false);

  in_memory_read_task(const in_memory_read_task&) = delete;
  in_memory_read_task& operator=(const in_memory_read_task&) = delete;

  void execute() override;

 private:
  void execute_read_by_value();
  void execute_read_by_reference();

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
   * @param[in] query_context The query context in which the current task is running in.
   * @param[in] task_id Globally unique identifier of the task.
   * @param[in] stage_id Stage of the current task.
   * @param[in] input The input task emitting new data.
   * @param[in] non_owned_memory_resource A memory resource for allocating memory.
   * @param[in] appender An appender functor for adding row groups to the table.
   * @param[in] column_indexes Columns to be loaded.
   * @param[in] data_types Expected data types of each column. If the actual data type of a loaded
   * column is different from expected, the column will be casted to the data type specified. Must
   * have the same length as `column_names`.
   */
  in_memory_write_task(query_context* query_context,
                       int32_t task_id,
                       int32_t stage_id,
                       std::shared_ptr<task> input,
                       rmm::mr::device_memory_resource* non_owned_memory_resource,
                       in_memory_table::row_group_appender appender,
                       std::vector<cudf::size_type> column_indexes,
                       std::vector<cudf::data_type> data_types);

  in_memory_write_task(const in_memory_write_task&) = delete;
  in_memory_write_task& operator=(const in_memory_write_task&) = delete;

  void execute() override;

 private:
  rmm::mr::device_memory_resource*
    _non_owned_memory_resource; /**< Non-owning reference to a memory
                                   resource owned by in_memory_table. */
  in_memory_table::row_group_appender
    _appender; /**< Implicitly holds a non-owning reference to an in_memory_table */
  std::vector<cudf::size_type> _column_indexes;
  std::vector<cudf::data_type> _data_types;
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
    query_context* query_context,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

 private:
  in_memory_readable_view(in_memory_table* non_owning_table);

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
    query_context* query_context,
    int32_t stage_id,
    std::vector<std::string> column_names,
    std::vector<cudf::data_type> data_types) override;

 private:
  in_memory_writeable_view(in_memory_table* non_owning_table);

  in_memory_table* _non_owning_table;
};

}  // namespace storage

}  // namespace gqe
