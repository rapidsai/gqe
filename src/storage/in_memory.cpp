/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/task.hpp>
#include <gqe/memory_resource/numa_memory_resource.hpp>
#include <gqe/memory_resource/pinned_memory_resource.hpp>
#include <gqe/memory_resource/system_memory_resource.hpp>
#include <gqe/storage/in_memory.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/types.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>  // std::unique_lock
#include <numeric>
#include <shared_mutex>  // std::shared_lock
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

class nvcomp_allocator {
 public:
  nvcomp_allocator(rmm::cuda_stream_view stream) : _stream(stream) {}

  void* allocate(std::size_t bytes)
  {
    return rmm::mr::get_current_device_resource()->allocate(bytes, _stream);
  }

  void deallocate(void* ptr, std::size_t bytes)
  {
    return rmm::mr::get_current_device_resource()->deallocate(ptr, bytes, _stream);
  }

 private:
  rmm::cuda_stream_view _stream;
};

std::unique_ptr<rmm::device_buffer> ans_compress(rmm::device_buffer const* uncompressed,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr,
                                                 nvcomp_allocator const& allocator)
{
  if (uncompressed->size() <= 0) { return std::make_unique<rmm::device_buffer>(0, stream, mr); }

  // TODO: More experiment and discussion are needed to choose the best chunk size for our purpose.
  size_t constexpr chunk_size = nvcompANSCompressionMaxAllowedChunkSize;
  nvcomp::ANSManager nvcomp_manager(chunk_size, nvcompBatchedANSDefaultOpts, stream);

  auto comp_config = nvcomp_manager.configure_compression(uncompressed->size());

  auto compressed_buffer =
    std::make_unique<rmm::device_buffer>(comp_config.max_compressed_buffer_size, stream, mr);

  // FIXME: in newer verions of nvcomp, we can pass an additional argument to get comp_size
  nvcomp_manager.compress(static_cast<uint8_t const*>(uncompressed->data()),
                          static_cast<uint8_t*>(compressed_buffer->data()),
                          comp_config);

  // FIXME: Use `set_scratch_allocators` to let nvcomp use the device memory pool.
  // The following code is commented out due to bugs in nvcomp.
  /*
  nvcomp_manager.set_scratch_allocators(
    std::bind(&nvcomp_allocator::allocate, allocator, std::placeholders::_1),
    std::bind(
      &nvcomp_allocator::deallocate, allocator, std::placeholders::_1, std::placeholders::_2));
  */

  auto const comp_size =
    nvcomp_manager.get_compressed_output_size(static_cast<uint8_t*>(compressed_buffer->data()));
  compressed_buffer->resize(comp_size, stream);
  compressed_buffer->shrink_to_fit(stream);

  return compressed_buffer;
}

std::unique_ptr<rmm::device_buffer> ans_decompress(rmm::device_buffer const* compressed,
                                                   rmm::cuda_stream_view stream,
                                                   nvcomp_allocator const& allocator,
                                                   rmm::device_async_resource_ref mr)
{
  // Since ANS is a multi-pass algorithm, copy the data into device memory before
  // decompression.
  rmm::device_buffer device_memory_compressed{compressed->data(), compressed->size(), stream};
  auto comp_buffer = static_cast<uint8_t const*>(device_memory_compressed.data());

  auto nvcomp_manager = nvcomp::create_manager(comp_buffer, stream);

  // FIXME: Use `set_scratch_allocators` to let nvcomp use the device memory pool.
  // The following code is commented out due to bugs in nvcomp.
  /*
  nvcomp_manager->set_scratch_allocators(
    std::bind(&nvcomp_allocator::allocate, allocator, std::placeholders::_1),
    std::bind(
      &nvcomp_allocator::deallocate, allocator, std::placeholders::_1, std::placeholders::_2));
  */

  auto decomp_config = nvcomp_manager->configure_decompression(comp_buffer);

  auto decompressed_buffer =
    std::make_unique<rmm::device_buffer>(decomp_config.decomp_data_size, stream, mr);
  nvcomp_manager->decompress(
    static_cast<uint8_t*>(decompressed_buffer->data()), comp_buffer, decomp_config);

  return decompressed_buffer;
}

}  // namespace

namespace gqe {

namespace storage {

// Function to map cudf::type_id to nvcomp_data_format
nvcompType_t get_optimal_nvcomp_data_type(cudf::type_id dtype)
{
  switch (dtype) {
    case cudf::type_id::INT8: return NVCOMP_TYPE_CHAR;
    case cudf::type_id::INT16: return NVCOMP_TYPE_SHORT;
    case cudf::type_id::INT32: return NVCOMP_TYPE_INT;
    case cudf::type_id::INT64: return NVCOMP_TYPE_LONGLONG;
    case cudf::type_id::UINT8: return NVCOMP_TYPE_UCHAR;
    case cudf::type_id::UINT16: return NVCOMP_TYPE_USHORT;
    case cudf::type_id::UINT32: return NVCOMP_TYPE_UINT;
    case cudf::type_id::UINT64: return NVCOMP_TYPE_ULONGLONG;
    case cudf::type_id::BOOL8: return NVCOMP_TYPE_CHAR;
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_DAYS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
    case cudf::type_id::DECIMAL32: return NVCOMP_TYPE_INT;
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return NVCOMP_TYPE_LONGLONG;
    default: return NVCOMP_TYPE_CHAR;
  }
}

contiguous_column::contiguous_column(cudf::column&& cudf_column)
  : column_base(), _data(std::move(cudf_column))
{
}

int64_t contiguous_column::size() const
{
  assert(static_cast<std::size_t>(_data.size()) <= std::numeric_limits<int64_t>::max());
  return static_cast<int64_t>(_data.size());
}

cudf::column_view contiguous_column::view() const { return _data.view(); }

cudf::mutable_column_view contiguous_column::mutable_view() { return _data.mutable_view(); }

plain_buffer::plain_buffer(rmm::device_buffer const* input,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  _buffer = std::make_unique<rmm::device_buffer>(*input, stream, mr);
}

std::unique_ptr<rmm::device_buffer> plain_buffer::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return std::make_unique<rmm::device_buffer>(*_buffer, stream, mr);
}

ans_compressed_buffer::ans_compressed_buffer(rmm::device_buffer const* input,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  nvcomp_allocator allocator(stream);

  _compressed_buffer = ans_compress(input, stream, mr, allocator);
}

std::unique_ptr<rmm::device_buffer> ans_compressed_buffer::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  nvcomp_allocator allocator(stream);

  return ans_decompress(_compressed_buffer.get(), stream, allocator, mr);
}

std::unique_ptr<compressed_buffer> do_compress(rmm::device_buffer const* input,
                                               compression_format comp_format,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::unique_ptr<compressed_buffer> output;

  switch (comp_format) {
    case compression_format::ans: {
      output = std::make_unique<ans_compressed_buffer>(input, stream, mr);
      break;
    }
    default: throw std::logic_error("Unsupported compression format for the in-memory table");
  }

  auto const uncompressed_size = input->size();
  auto const compressed_size   = output->compressed_size();
  auto const compression_ratio = static_cast<double>(uncompressed_size) / compressed_size;

  if (compression_ratio < 1) {
    // If the compression ratio is less than 1, the compressed size is larger than the uncompressed
    // size, so we fall back to use the plain encoding.
    // In the future, we could consider setting the threshold to larger than 1.
    output = std::make_unique<plain_buffer>(input, stream, mr);
  }

  return output;
}

std::unique_ptr<rmm::device_buffer> compressed_column::compress(rmm::device_buffer const* input,
                                                                rmm::device_async_resource_ref mr,
                                                                bool is_null_mask)
{
  bool compression_viable;
  std::unique_ptr<rmm::device_buffer> output =
    _nvcomp_manager.do_compress(input, compression_viable, mr);
  if (!is_null_mask) { is_compressed = compression_viable; }

  return output;
}

compressed_column::compressed_column(cudf::column&& cudf_column,
                                     compression_format comp_format,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr,
                                     nvcompType_t nvcomp_data_format,
                                     int chunk_size)
  : column_base(),
    _comp_format(comp_format),
    compression_ratio(0.0),
    is_compressed(true),
    _nvcomp_manager(stream, comp_format, nvcomp_data_format, chunk_size)
{
  _size       = cudf_column.size();
  _dtype      = cudf_column.type();
  _null_count = cudf_column.null_count();

  auto column_content = cudf_column.release();
  _compressed_data    = compress(column_content.data.get(), mr, false);
  _compressed_size    = _compressed_data->size();
  compression_ratio   = static_cast<float>(_size) / _compressed_size;

  if (_null_count > 0) {
    _compressed_null_mask = compress(column_content.null_mask.get(), mr, true);
  }

  _compressed_children.reserve(column_content.children.size());

  for (auto& child : column_content.children) {
    auto dtype         = child->type().id();
    nvcomp_data_format = get_optimal_nvcomp_data_type(dtype);

    _compressed_children.push_back(std::make_unique<compressed_column>(
      std::move(*child), comp_format, stream, mr, nvcomp_data_format, chunk_size));
  }
}

int64_t compressed_column::size() const { return _size; }

std::unique_ptr<cudf::column> compressed_column::decompress(rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  std::vector<std::unique_ptr<cudf::column>> decompressed_children;
  decompressed_children.reserve(_compressed_children.size());

  for (auto const& compressed_child : _compressed_children) {
    decompressed_children.push_back(compressed_child->decompress(stream, mr));
  }

  // auto decompressed_data  = _nvcomp_manager.do_decompress(_compressed_data.get(), mr);
  std::unique_ptr<rmm::device_buffer> decompressed_data;

  if (is_compressed) {
    decompressed_data = _nvcomp_manager.do_decompress(_compressed_data.get(), mr);
  } else {
    decompressed_data = std::make_unique<rmm::device_buffer>(*_compressed_data, stream, mr);
  }

  std::unique_ptr<rmm::device_buffer> decompressed_null_mask;
  if (_null_count > 0) {
    decompressed_null_mask = _nvcomp_manager.do_decompress(_compressed_null_mask.get(), mr);
  } else {
    decompressed_null_mask = std::make_unique<rmm::device_buffer>();
  }

  return std::make_unique<cudf::column>(_dtype,
                                        _size,
                                        std::move(*decompressed_data),
                                        std::move(*decompressed_null_mask),
                                        _null_count,
                                        std::move(decompressed_children));
}

row_group::row_group(std::vector<std::unique_ptr<column_base>>&& columns)
  : _columns(std::move(columns))
{
}

int64_t row_group::size() const
{
  if (_columns.empty()) {
    return 0;
  } else {
    return _columns.front()->size();
  }
}

column_base& row_group::get_column(cudf::size_type column_index) const
{
  return *_columns.at(column_index);
}

in_memory_table::in_memory_table(memory_kind::type memory_kind,
                                 std::vector<std::string> const& column_names,
                                 std::vector<cudf::data_type> const& column_types)
  : table(),
    _memory_kind(memory_kind),
    _column_types(std::move(column_types)),
    _row_groups(),
    _row_group_latch()
{
  // Populate column name-to-index map
  _column_name_to_index.reserve(column_names.size());
  for (decltype(column_names.size()) idx = 0; idx < column_names.size(); ++idx) {
    _column_name_to_index[column_names[idx]] = idx;
  }

  // Create memory resource based on memory kind. The allocators must be thread-safe.
  std::unique_ptr<rmm::mr::device_memory_resource> mr = std::visit(
    utility::overloaded{
      [](memory_kind::system) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::make_unique<memory_resource::system_memory_resource>();
      },
      [](const memory_kind::numa& numa) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::make_unique<memory_resource::numa_memory_resource>(numa.numa_node_set,
                                                                       numa.page_kind);
      },
      [](const memory_kind::pinned& numa) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::make_unique<memory_resource::pinned_memory_resource>();
      },
      [](const memory_kind::device& device) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        // FIXME: specify device instead of allocating on default CUDA device
        return std::make_unique<rmm::mr::cuda_memory_resource>();
      },
      [](const memory_kind::managed) -> std::unique_ptr<rmm::mr::device_memory_resource> {
        return std::unique_ptr<rmm::mr::managed_memory_resource>();
      }},
    memory_kind);

  _memory_resource = std::move(mr);
}

bool in_memory_table::is_readable() const { return true; }

bool in_memory_table::is_writeable() const { return true; }

int32_t in_memory_table::max_concurrent_readers() const
{
  auto const nrow_groups = _row_groups.size();
  assert(nrow_groups <= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()));

  return static_cast<int32_t>(nrow_groups);
}

int32_t in_memory_table::max_concurrent_writers() const
{
  // Each writer appends a new row group to the table. Thus, the number of
  // writers is theoretically unlimited.
  return std::numeric_limits<int32_t>::max();
}

std::unique_ptr<readable_view> in_memory_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new in_memory_readable_view(this));
}

std::unique_ptr<writeable_view> in_memory_table::writeable_view()
{
  return std::unique_ptr<storage::writeable_view>(new in_memory_writeable_view(this));
}

cudf::size_type in_memory_table::get_column_index(std::string const& column_name) const
{
  return _column_name_to_index.at(column_name);
}

in_memory_table::row_group_appender in_memory_table::get_row_group_appender()
{
  return {&_row_groups, &_row_group_latch};
}

in_memory_table::row_group_appender::row_group_appender(
  std::deque<row_group>* non_owning_row_groups, std::shared_mutex* non_owning_row_group_latch)
  : _non_owning_row_groups(non_owning_row_groups),
    _non_owning_row_group_latch(non_owning_row_group_latch)
{
}

void in_memory_table::row_group_appender::operator()(row_group&& new_row_group)
{
  std::unique_lock latch_guard(*_non_owning_row_group_latch);
  _non_owning_row_groups->push_back(std::move(new_row_group));
}

void in_memory_table::row_group_appender::operator()(std::vector<row_group>&& new_row_groups)
{
  std::unique_lock latch_guard(*_non_owning_row_group_latch);

  for (auto&& rg : new_row_groups) {
    _non_owning_row_groups->push_back(std::move(rg));
  }
}

in_memory_read_task::in_memory_read_task(query_context* query_context,
                                         int32_t task_id,
                                         int32_t stage_id,
                                         std::vector<const row_group*> row_groups,
                                         std::vector<cudf::size_type> column_indexes,
                                         std::vector<cudf::data_type> data_types,
                                         memory_kind::type memory_kind,
                                         std::unique_ptr<gqe::expression> partial_filter,
                                         std::vector<std::shared_ptr<task>> subquery_tasks,
                                         bool force_zero_copy_disable)
  : read_task_base(query_context, task_id, stage_id, std::move(subquery_tasks)),
    _row_groups(std::move(row_groups)),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _memory_kind(std::move(memory_kind)),
    _partial_filter(std::move(partial_filter)),
    _force_zero_copy_disable(force_zero_copy_disable)
{
}

void in_memory_read_task::execute_read_by_value()
{
  auto mr     = rmm::mr::get_current_device_resource();
  auto stream = cudf::get_default_stream();

  std::vector<std::unique_ptr<cudf::column>> result_columns;
  result_columns.reserve(_column_indexes.size());

  // For each table column, concatenate the row groups into a single cudf::column
  for (auto const& idx : _column_indexes) {
    std::vector<cudf::column_view> src_views;
    src_views.reserve(_row_groups.size());
    std::vector<std::unique_ptr<cudf::column>>
      src_columns;  // Needed for keeping the decompressed column alive

    for (auto const& row_group : _row_groups) {
      auto& current_column = row_group->get_column(idx);

      switch (current_column.type()) {
        case in_memory_column_type::CONTIGUOUS: {
          auto src_view = dynamic_cast<contiguous_column&>(current_column).view();
          src_views.push_back(src_view);
          break;
        }
        case in_memory_column_type::COMPRESSED: {
          auto src_column = dynamic_cast<compressed_column&>(current_column).decompress(stream);
          src_views.push_back(src_column->view());
          src_columns.push_back(std::move(src_column));
          break;
        }
      }
    }

    auto result_column = cudf::concatenate(src_views, stream, mr);

    result_columns.emplace_back(std::move(result_column));
  }

  // Emit the result
  auto new_table = std::make_unique<cudf::table>(std::move(result_columns));

  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_value, output_size={}.",
    task_id(),
    stage_id(),
    new_table->num_rows());
  emit_result(std::move(new_table));
}

void in_memory_read_task::execute_read_by_reference()
{
  if (_row_groups.size() > 1) {
    throw std::logic_error(
      "Zero-copy read is only possible if columns have a contiguous "
      "memory layout. Cannot handle multiple row groups.");
  }

  std::vector<cudf::column_view> result_columns;
  result_columns.reserve(_column_indexes.size());

  for (auto const& idx : _column_indexes) {
    auto view = dynamic_cast<contiguous_column&>(_row_groups.front()->get_column(idx)).view();
    result_columns.push_back(view);
  }

  auto result = cudf::table_view(std::move(result_columns));

  GQE_LOG_TRACE(
    "Execute in-memory read task: task_id={}, stage_id={}, strategy=by_reference, output_size={}.",
    task_id(),
    stage_id(),
    result.num_rows());
  emit_result(result);
}

void in_memory_read_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range in_memory_read_task_range("in_memory_read_task");

  // Check if zero-copy is legal.
  bool is_gpu_accessible   = memory_kind::is_gpu_accessible(_memory_kind);
  bool is_single_row_group = _row_groups.size() <= 1;
  bool is_compressed =
    get_query_context()->parameters.in_memory_table_compression_format != compression_format::none;

  // Execute read.
  if (get_query_context()->parameters.read_zero_copy_enable) {
    if (_force_zero_copy_disable) {
      GQE_LOG_INFO("Disabling zero-copy read due to override by the executor for performance.");
      execute_read_by_value();
    } else if (!is_gpu_accessible) {
      GQE_LOG_WARN(
        "Disabling zero-copy read because the GPU cannot access pageable memory on this "
        "system.");
      execute_read_by_value();
    } else if (!is_single_row_group) {
      GQE_LOG_WARN("Disabling zero-copy because cannot handle more than one row group per task.");
      execute_read_by_value();
    } else if (is_compressed) {
      GQE_LOG_WARN("Disabling zero-copy because cannot handle compressed tables.");
      execute_read_by_value();
    } else {
      GQE_LOG_DEBUG("Performing zero-copy read.");
      execute_read_by_reference();
    }
  } else {
    execute_read_by_value();
  }

  remove_dependencies();
}

in_memory_write_task::in_memory_write_task(
  query_context* query_context,
  int32_t task_id,
  int32_t stage_id,
  std::shared_ptr<task> input,
  rmm::mr::device_memory_resource* non_owned_memory_resource,
  in_memory_table::row_group_appender appender,
  std::vector<cudf::size_type> column_indexes,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
  : write_task_base(query_context, task_id, stage_id, input),
    _non_owned_memory_resource(non_owned_memory_resource),
    _appender(std::move(appender)),
    _column_indexes(std::move(column_indexes)),
    _data_types(std::move(data_types)),
    _statistics(statistics)
{
}

void in_memory_write_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range in_memory_write_task_range("in_memory_write_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = *dependent_tasks[0]->result();

  // Check if input schema matches output schema
  auto num_columns = _column_indexes.size();
  if (_data_types.size() != num_columns) {
    throw std::length_error("Data types must have the same length as the number of columns");
  }
  if (static_cast<decltype(num_columns)>(input_table.num_columns()) != num_columns) {
    throw std::invalid_argument("Query result schema must match Parquet output schema");
  }

  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    if (_data_types[column_idx] != input_table.column(column_idx).type()) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }
  }

  // Create columns and insert data into them

  // Specify the stream to enqueue copy on
  auto stream = cudf::get_default_stream();

  // FIXME: Maybe we can zero-copy construct (i.e., move to) the new columns if
  // the input_table columns have the same memory type as the result columns.
  // This would require us to destroy the result cache here, but that should be
  // ok because `write` is the root task.
  std::vector<std::unique_ptr<column_base>> new_columns(_column_indexes.size());
  for (decltype(input_table.num_columns()) column_idx = 0; column_idx < input_table.num_columns();
       ++column_idx) {
    auto const& input_column = input_table.column(column_idx);
    auto cudf_column         = cudf::column(input_column, stream, _non_owned_memory_resource);

    auto comp_format = get_query_context()->parameters.in_memory_table_compression_format;
    nvcompType_t nvcomp_data_format =
      get_query_context()->parameters.in_memory_table_compression_data_type;
    auto const chunk_size = get_query_context()->parameters.compression_chunk_size;

    auto dtype = cudf_column.type().id();

    nvcomp_data_format = get_optimal_nvcomp_data_type(dtype);

    if (comp_format == compression_format::none) {
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<contiguous_column>(std::move(cudf_column));
    } else {
      new_columns[_column_indexes[column_idx]] =
        std::make_unique<compressed_column>(std::move(cudf_column),
                                            comp_format,
                                            stream,
                                            _non_owned_memory_resource,
                                            nvcomp_data_format,
                                            chunk_size);
    }
  }

  // Create row group from columns
  auto row_group = storage::row_group(std::move(new_columns));

  // Append row group to table
  _appender(std::move(row_group));
  _statistics->add_rows(input_table.num_rows());

  GQE_LOG_TRACE("Execute in-memory write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
}

in_memory_readable_view::in_memory_readable_view(in_memory_table* non_owning_table)
  : readable_view(), _non_owning_table(non_owning_table)
{
}

std::vector<std::unique_ptr<read_task_base>> in_memory_readable_view::get_read_tasks(
  std::vector<readable_view::task_parameters>&& task_parameters,
  query_context* query_context,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types)
{
  assert(!task_parameters.empty() && "Must have at least one read task");
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0 && "Stage ID must be a positive value");

  assert(task_parameters.size() <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  auto parallelism = static_cast<int64_t>(task_parameters.size());

  // Transform column names into column indexes
  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](const std::string& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  // Acquire shared read latch on row groups and release on function return
  std::shared_lock latch_guard(_non_owning_table->_row_group_latch);

  assert(_non_owning_table->_row_groups.size() <=
         static_cast<decltype(_non_owning_table->_row_groups.size())>(
           std::numeric_limits<int64_t>::max()));
  const auto nrow_groups = static_cast<int64_t>(_non_owning_table->_row_groups.size());

  const auto max_nrow_groups_per_instance = utility::divide_round_up(nrow_groups, parallelism);

  // Create tasks
  std::vector<std::unique_ptr<read_task_base>> read_tasks;
  read_tasks.reserve(task_parameters.size());
  {
    // Iterate over multiple iterators at the same time
    auto task            = task_parameters.begin();
    int64_t begin_offset = 0;

    while (task != task_parameters.end() && begin_offset < nrow_groups) {
      const auto end_offset = std::min(begin_offset + max_nrow_groups_per_instance, nrow_groups);

      // Get a chunk of row groups
      std::vector<const row_group*> row_groups_chunk;
      assert(end_offset >= begin_offset);
      row_groups_chunk.reserve(end_offset - begin_offset);
      std::transform(_non_owning_table->_row_groups.cbegin() + begin_offset,
                     _non_owning_table->_row_groups.cbegin() + end_offset,
                     std::back_inserter(row_groups_chunk),
                     [](const row_group& rg) { return &rg; });

      // Create a new read task
      auto read_task = std::make_unique<in_memory_read_task>(query_context,
                                                             task->task_id,
                                                             stage_id,
                                                             std::move(row_groups_chunk),
                                                             column_indexes,
                                                             data_types,
                                                             _non_owning_table->_memory_kind,
                                                             std::move(task->partial_filter),
                                                             std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      // Advance the iterators
      ++task, begin_offset += max_nrow_groups_per_instance;
    }
  }

  return read_tasks;
}

in_memory_writeable_view::in_memory_writeable_view(in_memory_table* non_owning_table)
  : writeable_view(), _non_owning_table(non_owning_table)
{
}

std::vector<std::unique_ptr<write_task_base>> in_memory_writeable_view::get_write_tasks(
  std::vector<writeable_view::task_parameters>&& task_parameters,
  query_context* query_context,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
{
  assert(!task_parameters.empty() && "Must have at least one write task");
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0 && "Stage ID must be a positive value");

  // Transform column names into column indexes
  std::vector<cudf::size_type> column_indexes;
  column_indexes.reserve(column_names.size());
  std::transform(column_names.cbegin(),
                 column_names.cend(),
                 std::back_inserter(column_indexes),
                 [this](const std::string& column_name) {
                   return _non_owning_table->get_column_index(column_name);
                 });

  std::vector<std::unique_ptr<write_task_base>> write_tasks;
  write_tasks.reserve(task_parameters.size());

  for (auto& task_parameter : task_parameters) {
    // Create appender, which will acquire a write latch to the row groups vector
    auto appender = _non_owning_table->get_row_group_appender();

    // Create a new write task
    auto write_task =
      std::make_unique<in_memory_write_task>(query_context,
                                             task_parameter.task_id,
                                             stage_id,
                                             std::move(task_parameter.input),
                                             _non_owning_table->_memory_resource.get(),
                                             std::move(appender),
                                             column_indexes,
                                             data_types,
                                             statistics);
    write_tasks.push_back(std::move(write_task));
  }

  return write_tasks;
}

}  // namespace storage

}  // namespace gqe
