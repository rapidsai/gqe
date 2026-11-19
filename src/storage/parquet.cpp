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

#include <gqe/storage/parquet.hpp>

#include <gqe/expression/column_reference.hpp>
#include <gqe/query_context.hpp>
#include <gqe/storage/chunked_parquet_reader.hpp>
#include <gqe/storage/chunked_parquet_writer.hpp>
#include <gqe/storage/hive_partitioning.hpp>
#include <gqe/storage/oneshot_parquet_reader.hpp>
#include <gqe/storage/parquet_footer_cache.hpp>
#include <gqe/storage/parquet_reader.hpp>
#include <gqe/storage/parquet_reference_iterate.hpp>
#include <gqe/storage/parquet_resolve.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>
#include <gqe/utility/multi_process_helpers.hpp>

#include <cuda/stream>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <rmm/device_buffer.hpp>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gqe {

namespace storage {

parquet_table::parquet_table(std::vector<std::string> file_paths)
  : table(), _file_paths(std::make_shared<std::vector<std::string>>(std::move(file_paths)))
{
  // File paths need to be sorted to retain the existing clustering on disk. This assumes that the
  // lexicographical sort order corresponds to the sort order of the rows in the table, which means
  // that the row group numbers in the file names have to be padded with leading zeros.
  std::sort(_file_paths->begin(), _file_paths->end());
}

bool parquet_table::is_readable() const { return true; }
bool parquet_table::is_writeable() const { return true; }
int32_t parquet_table::max_concurrent_readers() const { return _file_paths->size(); }
int32_t parquet_table::max_concurrent_writers() const { return _file_paths->size(); }

std::unique_ptr<readable_view> parquet_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new parquet_readable_view(_file_paths.get()));
}

std::unique_ptr<writeable_view> parquet_table::writeable_view()
{
  return std::unique_ptr<parquet_writeable_view>(new parquet_writeable_view(_file_paths.get()));
}

parquet_read_task::parquet_read_task(context_reference ctx_ref,
                                     int32_t task_id,
                                     int32_t stage_id,
                                     std::vector<std::string> file_paths,
                                     std::vector<std::string> column_names,
                                     std::vector<cudf::data_type> data_types,
                                     std::unique_ptr<expression> partial_filter,
                                     std::vector<std::shared_ptr<task>> subquery_tasks)
  : read_task_base(ctx_ref, task_id, stage_id, std::move(subquery_tasks)),
    _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types)),
    _partial_filter(std::move(partial_filter))
{
}

namespace {

// Load a cuDF table from Parquet files listed at `file_paths`, and columns indicated by
// `column_names`.
// Note: the `file_paths` argument must not be empty.
table_with_metadata table_from_parquet(context_reference ctx_ref,
                                       std::vector<std::string> const& file_paths,
                                       std::vector<std::string> const& column_names)
{
  assert(!file_paths.empty());

  table_with_metadata result;

#ifdef ENABLE_CUSTOMIZED_PARQUET
  if (ctx_ref._query_context->parameters.use_customized_io) {
    try {
      auto const bounce_buffer_size = ctx_ref._query_context->io_bounce_buffer_mr->get_block_size();
      // Note that we use `device_buffer` only as a RAII wrapper. `bounce_buffer` is located in the
      // pinned host memory, not device memory.
      rmm::device_buffer bounce_buffer(bounce_buffer_size,
                                       cudf::get_default_stream(),
                                       *ctx_ref._query_context->io_bounce_buffer_mr);

      result =
        gqe::storage::read_parquet_custom(file_paths,
                                          column_names,
                                          bounce_buffer.data(),
                                          bounce_buffer_size,
                                          ctx_ref._query_context->parameters.io_auxiliary_threads,
                                          ctx_ref._query_context->parameters.io_block_size,
                                          ctx_ref._query_context->parameters.io_engine,
                                          ctx_ref._query_context->parameters.io_pipelining,
                                          ctx_ref._query_context->parameters.io_alignment,
                                          ctx_ref._query_context->disk_timer,
                                          ctx_ref._query_context->h2d_timer,
                                          ctx_ref._query_context->decomp_timer,
                                          ctx_ref._query_context->decode_timer);
    } catch (gqe::storage::unsupported_error const& error) {
      GQE_LOG_TRACE(error.what());
      GQE_LOG_TRACE("Fallback to cuDF's Parquet reader when loading: " + file_paths[0]);

      result = read_parquet_cudf(file_paths, column_names);
    }
  } else {
    result = read_parquet_cudf(file_paths, column_names);
  }
#else
  result = read_parquet_cudf(file_paths, column_names);
#endif

  assert(result.table->num_columns() == static_cast<cudf::size_type>(column_names.size()));
  return result;
}

// Convert the read columns into the specified types if necessary
std::unique_ptr<cudf::table> enforce_data_types(std::unique_ptr<cudf::table> input,
                                                std::vector<cudf::data_type> const& data_type)
{
  auto input_columns     = input->release();
  auto const num_columns = input_columns.size();

  std::vector<std::unique_ptr<cudf::column>> converted_columns;
  converted_columns.reserve(num_columns);
  for (std::size_t column_idx = 0; column_idx < num_columns; column_idx++) {
    auto const column_view   = input_columns[column_idx]->view();
    auto const expected_type = data_type[column_idx];

    if (column_view.type() == expected_type) {
      converted_columns.push_back(std::move(input_columns[column_idx]));
    } else {
      converted_columns.push_back(gqe::storage::cast(column_view, expected_type));
    }
  }
  assert(converted_columns.size() == num_columns);

  return std::make_unique<cudf::table>(std::move(converted_columns));
}

}  // namespace

std::string parquet_read_task::print_column_names() const
{
  std::string output = "[";
  for (std::size_t column_idx = 0; column_idx < _column_names.size(); column_idx++) {
    output += _column_names[column_idx];
    if (column_idx + 1 < _column_names.size()) output += ",";
  }
  output += "]";
  return output;
}

detail::hive_file_filter_info parquet_read_task::parse_partial_filter() const
{
  if (!_partial_filter || _partial_filter->type() != expression::expression_type::subquery ||
      static_cast<subquery_expression const*>(_partial_filter.get())->subquery_type() !=
        subquery_expression::subquery_type_type::in_predicate) {
    return detail::all_files_non_partitioned(_file_paths.size());
  }

  auto const partial_filter_subquery =
    static_cast<subquery_expression const*>(_partial_filter.get());
  auto const expressions = partial_filter_subquery->children();

  // FIXME: Assuming there is one child expressions
  if (expressions.size() != 1) {
    throw std::runtime_error("Currently only one expression per subquery");
  }

  // FIXME: Assuming all queried expressions are column references
  if (expressions[0]->type() != expression::expression_type::column_reference) {
    throw std::runtime_error(
      "Currently only column_reference type expressions are supported for the read task "
      "partial filter");
  }

  auto const column_idx =
    dynamic_cast<gqe::column_reference_expression const*>(expressions[0])->column_idx();

  assert(static_cast<size_t>(column_idx) < _column_names.size());
  auto const column_name = _column_names[column_idx];

  auto const subquery_tasks = subqueries();
  auto haystack_table = subquery_tasks[partial_filter_subquery->relation_index()]->result().value();

  return detail::filter_hive_files_by_haystack(
    _file_paths, column_idx, column_name, haystack_table);
}

void parquet_read_task::read()
{
  std::unique_ptr<cudf::table> result_table;
  auto info = parse_partial_filter();

  auto const has_non_partitioned_files = !info.non_partitioned_files.empty();
  auto const has_partitioned_files     = !info.partitioned_files.empty();

  // Load the table from non-partitioned files
  std::unique_ptr<cudf::table> non_partitioned_table;
  if (has_non_partitioned_files) {
    std::vector<std::string> file_paths_to_load;

    for (auto const& file_idx : info.non_partitioned_files) {
      file_paths_to_load.push_back(_file_paths[file_idx]);
    }
    assert(!file_paths_to_load.empty());

    auto table_with_metadata =
      table_from_parquet(get_context_reference(), file_paths_to_load, _column_names);
    if (_column_names.empty()) {
      // cudf::read_parquet with no projected columns returns a 0-row 0-col table, losing the
      // actual row count. Recover it from the per-file metadata and add a single phantom column
      // of type cudf::type_id::EMPTY with size `total_rows` and no data buffer, so
      // `table->num_rows()` and downstream COUNT(*) (via `column_view::size()`) see the right N
      // without allocating ~1 byte/row. (`make_fixed_width_column(EMPTY, n)` is disallowed in
      // libcudf; the public `cudf::column` constructor is the supported way to size an EMPTY
      // column while keeping a zero-size data buffer.)
      cudf::size_type total_rows = 0;
      for (auto n : table_with_metadata.rows_per_file)
        total_rows += n;
      std::vector<std::unique_ptr<cudf::column>> phantom;
      phantom.push_back(std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::EMPTY},
                                                       total_rows,
                                                       rmm::device_buffer{},
                                                       rmm::device_buffer{},
                                                       0));
      non_partitioned_table = std::make_unique<cudf::table>(std::move(phantom));
    } else {
      non_partitioned_table = enforce_data_types(std::move(table_with_metadata.table), _data_types);
    }
  }

  // Load the table from partitioned files
  std::unique_ptr<cudf::table> partitioned_table;
  if (has_partitioned_files) {
    std::vector<std::string> file_paths_to_load;
    std::vector<int64_t> partition_keys;

    for (auto const& [file_idx, partition_key] : info.partitioned_files) {
      file_paths_to_load.push_back(_file_paths[file_idx]);
      partition_keys.push_back(*std::static_pointer_cast<int64_t>(partition_key));
    }

    std::vector<std::string> columns_to_load(_column_names);
    auto const partition_column_idx = info.partition_column_idx;
    columns_to_load.erase(columns_to_load.begin() + partition_column_idx);

    auto loaded_table =
      table_from_parquet(get_context_reference(), file_paths_to_load, std::move(columns_to_load));

    auto loaded_columns = loaded_table.table->release();
    loaded_columns.insert(
      loaded_columns.begin() + partition_column_idx,
      construct_partition_key_column(
        _data_types[partition_column_idx], std::move(partition_keys), loaded_table.rows_per_file));

    partitioned_table = std::make_unique<cudf::table>(std::move(loaded_columns));
    partitioned_table = enforce_data_types(std::move(partitioned_table), _data_types);
  }

  if (!has_non_partitioned_files && !has_partitioned_files) {
    // Construct an empty table
    std::vector<std::unique_ptr<cudf::column>> empty_columns;
    for (auto const& column_type : _data_types)
      empty_columns.push_back(cudf::make_empty_column(column_type));
    result_table = std::make_unique<cudf::table>(std::move(empty_columns));

  } else if (has_non_partitioned_files && !has_partitioned_files) {
    result_table = std::move(non_partitioned_table);

  } else if (!has_non_partitioned_files && has_partitioned_files) {
    result_table = std::move(partitioned_table);

  } else {
    // There are both partitioned and unpartitioned files, so we need to merge the partitioned
    // table with the unpartitioned table. This should be uncommon.
    GQE_LOG_WARN(
      "Both the partitioned and unpartitioned files are present. This usually means the dataset "
      "is "
      "not partitioned properly.");

    std::vector<cudf::table_view> tables_to_concat = {non_partitioned_table->view(),
                                                      partitioned_table->view()};
    result_table                                   = cudf::concatenate(tables_to_concat);
  }

  GQE_LOG_TRACE("Execute Parquet read task: task_id={}, stage_id={}, columns={}, output_size={}.",
                task_id(),
                stage_id(),
                print_column_names(),
                result_table->num_rows());
  emit_result(std::move(result_table));
}

void parquet_read_task::read_nothing(int mpi_rank)
{
  GQE_LOG_TRACE(
    "Execute Parquet read task: task_id={}, stage_id={}, Rank {} does not read and emits an "
    "empty table",
    task_id(),
    stage_id(),
    mpi_rank);
  emit_result(std::make_unique<cudf::table>());
}

void parquet_read_task::execute()
{
  prepare_dependencies();
  utility::nvtx_scoped_range parquet_read_task_range("parquet_read_task");

  if (get_query_context()->parameters.use_in_memory_table_multigpu) {
    // Only a single rank reads the data
    if (gqe::utility::multi_process::nvshmem_rank_zero()) {
      read();
    } else {
      read_nothing(gqe::utility::multi_process::nvshmem_rank());
    }
  } else {
    read();
  }

  remove_dependencies();
}

parquet_write_task::parquet_write_task(context_reference ctx_ref,
                                       int32_t task_id,
                                       int32_t stage_id,
                                       std::shared_ptr<task> input,
                                       std::vector<std::string> file_paths,
                                       std::vector<std::string> column_names,
                                       std::vector<cudf::data_type> data_types)
  : write_task_base(ctx_ref, task_id, stage_id, input),
    _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types))
{
}

void parquet_write_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range parquet_write_task_range("parquet_write_task");

  auto const dependent_tasks = dependencies();
  assert(dependent_tasks.size() == 1);
  auto input_table = *dependent_tasks[0]->result();

  // check if input schema matches output schema
  auto num_columns = _column_names.size();
  if (_data_types.size() != num_columns) {
    throw std::length_error("Data type list must have the same length as the number of columns");
  }
  if (static_cast<std::size_t>(input_table.num_columns()) != num_columns) {
    throw std::invalid_argument("Query result schema must match Parquet output schema");
  }

  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    if (_data_types[column_idx] != input_table.column(column_idx).type()) {
      throw std::invalid_argument("Query result schema must match Parquet output schema");
    }
  }

  // set column names
  auto metadata = cudf::io::table_input_metadata(input_table);
  for (decltype(num_columns) column_idx = 0; column_idx < num_columns; ++column_idx) {
    metadata.column_metadata[column_idx].set_name(_column_names[column_idx]);
  }

  // write to Parquet file
  auto sink    = cudf::io::sink_info(std::move(_file_paths));
  auto builder = cudf::io::chunked_parquet_writer_options::builder(sink).metadata(metadata);

  cudf::io::chunked_parquet_writer(builder).write(input_table).close();

  GQE_LOG_TRACE("Execute Parquet write task: task_id={}, stage_id={}, input_size={}.",
                task_id(),
                stage_id(),
                input_table.num_rows());
  remove_dependencies();
}

parquet_readable_view::parquet_readable_view(std::vector<std::string>* non_owning_file_paths)
  : readable_view(), _non_owning_file_paths(non_owning_file_paths)
{
}

std::vector<std::unique_ptr<read_task_base>> parquet_readable_view::get_read_tasks(
  std::vector<readable_view::task_parameters>&& task_parameters,
  context_reference ctx_ref,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types)
{
  assert(!task_parameters.empty());
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0);

  GQE_EXPECTS(data_types.size() == column_names.size(),
              "data_types must have the same length as the number of columns",
              std::length_error);

  assert(task_parameters.size() <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  auto parallelism = static_cast<int64_t>(task_parameters.size());

  assert(_non_owning_file_paths->size() <=
         static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  const auto nfiles = static_cast<int64_t>(_non_owning_file_paths->size());

  auto const max_nfiles_per_instance = utility::divide_round_up(nfiles, parallelism);

  std::vector<std::unique_ptr<read_task_base>> read_tasks;
  read_tasks.reserve(task_parameters.size());
  {
    auto task            = task_parameters.begin();
    int64_t begin_offset = 0;

    while (task != task_parameters.end() && begin_offset < nfiles) {
      const auto end_offset = std::min(begin_offset + max_nfiles_per_instance, nfiles);

      assert(end_offset >= begin_offset);
      std::vector<std::string> file_paths_task{_non_owning_file_paths->begin() + begin_offset,
                                               _non_owning_file_paths->begin() + end_offset};

      auto read_task = std::make_unique<parquet_read_task>(ctx_ref,
                                                           task->task_id,
                                                           stage_id,
                                                           std::move(file_paths_task),
                                                           column_names,
                                                           data_types,
                                                           std::move(task->partial_filter),
                                                           std::move(task->subquery_tasks));
      read_tasks.push_back(std::move(read_task));

      ++task, begin_offset += max_nfiles_per_instance;
    }
  }

  return read_tasks;
}

qep_read_pipeline parquet_readable_view::get_read_pipeline(
  source_reader_type reader,
  std::string /*table_name*/,
  table_provider const* /*provider*/,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  std::optional<cudf::size_type> partition_column_idx,
  std::unique_ptr<gqe::expression> partial_filter)
{
  GQE_EXPECTS(data_types.size() == column_names.size(),
              "data_types must have the same length as the number of columns",
              std::length_error);
  if (partial_filter != nullptr) {
    GQE_LOG_INFO(
      "parquet_readable_view::get_read_pipeline: ignoring partial_filter pushdown hint; "
      "Parquet read tasks do not skip data by this predicate");
  }
  std::optional<hive_pruner_options> pruner;
  if (partition_column_idx) { pruner = hive_pruner_options{*partition_column_idx}; }
  switch (reader) {
    case source_reader_type::oneshot:
      return {.source = std::make_unique<oneshot_parquet_read_task>(
                *_non_owning_file_paths, std::move(column_names), std::move(data_types), pruner),
              .resolve = nullptr};
    case source_reader_type::chunked:
      return {.source = std::make_unique<chunked_parquet_read_task>(
                *_non_owning_file_paths, std::move(column_names), std::move(data_types), pruner),
              .resolve = nullptr};
    case source_reader_type::deferred: {
      auto cache = std::make_shared<parquet_footer_cache>();
      auto resolve =
        std::make_unique<parquet_resolve_task>(cache, column_names, data_types, pruner);
      auto source = std::make_unique<parquet_reference_iterate_task>(*_non_owning_file_paths,
                                                                     std::move(column_names),
                                                                     std::move(data_types),
                                                                     pruner,
                                                                     std::move(cache));
      return {.source = std::move(source), .resolve = std::move(resolve)};
    }
  }
  throw std::invalid_argument("parquet_readable_view: invalid source reader");
}

parquet_writeable_view::parquet_writeable_view(std::vector<std::string>* non_owning_file_paths)
  : writeable_view(), _non_owning_file_paths(non_owning_file_paths)
{
}

std::vector<std::unique_ptr<write_task_base>> parquet_writeable_view::get_write_tasks(
  std::vector<writeable_view::task_parameters>&& task_parameters,
  context_reference ctx_ref,
  int32_t stage_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  table_statistics_manager* statistics)
{
  assert(!task_parameters.empty());
  assert(std::all_of(task_parameters.cbegin(),
                     task_parameters.cend(),
                     [](auto& tp) { return tp.task_id >= 0; }) &&
         "Task ID must be a positive value");
  assert(stage_id >= 0);

  assert(task_parameters.size() <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  auto parallelism = static_cast<int64_t>(task_parameters.size());

  if (_non_owning_file_paths->size() < static_cast<std::size_t>(parallelism)) {
    throw std::logic_error("Less than one Parquet output file per worker");
  }

  assert(_non_owning_file_paths->size() <=
         static_cast<std::size_t>(std::numeric_limits<int64_t>::max()));
  const auto nfiles = static_cast<int64_t>(_non_owning_file_paths->size());

  int64_t const max_nfiles_per_instance = utility::divide_round_up(nfiles, parallelism);

  std::vector<std::unique_ptr<write_task_base>> write_tasks;
  write_tasks.reserve(task_parameters.size());
  {
    auto task            = task_parameters.begin();
    int64_t begin_offset = 0;

    while (task != task_parameters.end() && begin_offset < nfiles) {
      const auto end_offset = std::min(begin_offset + max_nfiles_per_instance, nfiles);

      assert(end_offset >= begin_offset);
      std::vector<std::string> file_paths_task{_non_owning_file_paths->begin() + begin_offset,
                                               _non_owning_file_paths->begin() + end_offset};
      auto write_task = std::make_unique<parquet_write_task>(ctx_ref,
                                                             task->task_id,
                                                             stage_id,
                                                             std::move(task->input),
                                                             std::move(file_paths_task),
                                                             column_names,
                                                             data_types);

      write_tasks.push_back(std::move(write_task));

      ++task, begin_offset += max_nfiles_per_instance;
    }
  }

  return write_tasks;
}

std::unique_ptr<qep::fold_task> parquet_writeable_view::get_fold_task(
  std::vector<std::string> column_names, std::vector<cudf::data_type> data_types)
{
  GQE_EXPECTS(data_types.size() == column_names.size(),
              "data_types must have the same length as the number of columns",
              std::length_error);
  GQE_EXPECTS(_non_owning_file_paths->size() == 1,
              "parquet_writeable_view::get_fold_task: parquet_table must have exactly one file "
              "path for the QEP write path",
              std::invalid_argument);
  return std::make_unique<chunked_parquet_write_task>(
    _non_owning_file_paths->front(), std::move(column_names), std::move(data_types));
}

};  // namespace storage
};  // namespace gqe
