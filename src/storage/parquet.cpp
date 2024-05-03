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

#include <gqe/expression/column_reference.hpp>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/parquet_reader.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>
#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace gqe {

namespace storage {

parquet_table::parquet_table(std::vector<std::string> file_paths)
  : table(), _file_paths(std::make_shared<std::vector<std::string>>(std::move(file_paths)))
{
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

parquet_read_task::parquet_read_task(query_context* query_context,
                                     int32_t task_id,
                                     int32_t stage_id,
                                     std::vector<std::string> file_paths,
                                     std::vector<std::string> column_names,
                                     std::vector<cudf::data_type> data_types,
                                     std::unique_ptr<expression> partial_filter,
                                     std::vector<std::shared_ptr<task>> subquery_tasks)
  : read_task_base(query_context, task_id, stage_id, std::move(subquery_tasks)),
    _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types)),
    _partial_filter(std::move(partial_filter))
{
}

struct file_filter_functor {
  /**
   * @brief Check the predicate to filter in files.
   *
   * @param[in] needles (file_idx, value) pairs to be tested.
   * @param[in] acceptable_value_col Files should be included if the value is in this list.
   *
   * @return A subset of `needles` where the predicate evaluates to `true`.
   */
  template <typename T>
  std::vector<std::pair<size_t, std::shared_ptr<void>>> operator()(
    std::vector<std::pair<size_t, int64_t>> const& needles,
    cudf::column_view const& acceptable_value_col)
  {
    std::vector<std::pair<size_t, std::shared_ptr<void>>> filtered_needles;

    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      filtered_needles.reserve(needles.size());

      std::vector<T> haystack(acceptable_value_col.size());
      GQE_CUDA_TRY(cudaMemcpy(haystack.data(),
                              acceptable_value_col.data<T>(),
                              acceptable_value_col.size() * sizeof(T),
                              cudaMemcpyDefault));

      // construct a hash set to easily search the needles. Make it 2x oversized to decrease
      // construction time
      std::unordered_set<int64_t> haystack_set(
        haystack.begin(), haystack.end(), 2 * haystack.size());

      // Filter out files that are needles and are not present in the haystack
      for (auto const& needle : needles) {
        if (haystack_set.find(needle.second) != haystack_set.end()) {
          filtered_needles.emplace_back(needle.first, std::make_shared<int64_t>(needle.second));
        }
      }
    }

    return filtered_needles;
  }
};

namespace {

// Load a cuDF table from Parquet files listed at `file_paths`, and columns indicated by
// `column_names`.
// Note: the `file_paths` argument must not be empty.
table_with_metadata table_from_parquet(query_context* qctx,
                                       std::vector<std::string> const& file_paths,
                                       std::vector<std::string> const& column_names)
{
  assert(!file_paths.empty());

  table_with_metadata result;

#ifdef ENABLE_CUSTOMIZED_PARQUET
  if (qctx->parameters.use_customized_io) {
    try {
      auto const bounce_buffer_size = qctx->io_bounce_buffer_mr->get_block_size();
      // Note that we use `device_buffer` only as a RAII wrapper. `bounce_buffer` is located in the
      // pinned host memory, not device memory.
      rmm::device_buffer bounce_buffer(
        bounce_buffer_size, rmm::cuda_stream_default, qctx->io_bounce_buffer_mr.get());

      auto const num_auxiliary_threads = qctx->parameters.io_auxiliary_threads;

      result = gqe::storage::read_parquet_custom(
        file_paths, column_names, bounce_buffer.data(), bounce_buffer_size, num_auxiliary_threads);
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
      converted_columns.push_back(cudf::cast(column_view, expected_type));
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

parquet_read_task::partial_filter_info parquet_read_task::parse_partial_filter() const
{
  parquet_read_task::partial_filter_info info;

  if (!_partial_filter) {
    info.non_partitioned_files.resize(_file_paths.size());
    std::iota(info.non_partitioned_files.begin(), info.non_partitioned_files.end(), 0);
    return info;
  }

  // FIXME: Only support in_predicate_expression
  if (_partial_filter->type() != expression::expression_type::subquery) {
    throw std::runtime_error("Partial filter expression must be a subquery expression");
  }
  auto const partial_filter_subquery =
    dynamic_cast<subquery_expression const*>(_partial_filter.get());
  if (partial_filter_subquery->subquery_type() !=
      subquery_expression::subquery_type_type::in_predicate) {
    throw std::runtime_error("Partial filter expression must be a in-predicate expression");
  }

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
  info.column_idx = column_idx;

  assert(static_cast<size_t>(column_idx) < _column_names.size());
  auto const column_name = _column_names[column_idx];

  // Get the available values for the column we've partitioned along
  std::vector<std::pair<size_t, int64_t>> needles;
  needles.reserve(_file_paths.size());
  std::regex valid_filename("(" + column_name + "=)([A-Za-z0-9_]+)");

  for (std::size_t file_idx = 0; file_idx < _file_paths.size(); ++file_idx) {
    std::smatch matches;
    std::regex_search(_file_paths[file_idx], matches, valid_filename);
    // make sure both subexpressions (column name and value) are matched
    if (matches.size() == 3) {
      try {
        int64_t value = stoll(matches[2]);
        needles.emplace_back(file_idx, value);
      } catch (const std::invalid_argument&) {
        if (matches[2] != "__HIVE_DEFAULT_PARTITION__") {
          throw std::runtime_error("Cannot parse partition key " + matches.str(2));
        }
      }
    } else {
      GQE_LOG_WARN(
        "Partial filter is supplied in the parquet read task, but cannot parse the file name");
      info.non_partitioned_files.push_back(file_idx);
    }
  }

  auto const subquery_tasks = subqueries();
  auto haystack_table = subquery_tasks[partial_filter_subquery->relation_index()]->result().value();

  if (haystack_table.num_columns() != 1) {
    throw std::runtime_error("Incorrect number of columns in the partial filter haystack");
  }

  if (!cudf::is_integral(haystack_table.column(0).type())) {
    throw std::runtime_error(
      "Partial filter in parquet read task does not support non-integral haystack");
  }

  auto acceptable_value_table     = cudf::drop_nulls(haystack_table, {0});
  auto const acceptable_value_col = acceptable_value_table->view().column(0);

  // Check the predicate to filter in files.
  info.partitioned_files = cudf::type_dispatcher(
    acceptable_value_col.type(), file_filter_functor{}, needles, acceptable_value_col);

  return info;
}

void parquet_read_task::execute()
{
  prepare_dependencies();

  utility::nvtx_scoped_range parquet_read_task_range("parquet_read_task");

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

    non_partitioned_table =
      table_from_parquet(get_query_context(), file_paths_to_load, _column_names).table;
    non_partitioned_table = enforce_data_types(std::move(non_partitioned_table), _data_types);
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
    auto const partition_column_idx = info.column_idx;
    columns_to_load.erase(columns_to_load.begin() + partition_column_idx);

    auto loaded_table =
      table_from_parquet(get_query_context(), file_paths_to_load, std::move(columns_to_load));

    auto loaded_columns = loaded_table.table->release();
    loaded_columns.insert(
      loaded_columns.begin() + partition_column_idx,
      construct_partition_key_column(
        _data_types[partition_column_idx], std::move(partition_keys), loaded_table.rows_per_file));

    partitioned_table = std::make_unique<cudf::table>(std::move(loaded_columns));
    partitioned_table = enforce_data_types(std::move(partitioned_table), _data_types);
  }

  std::unique_ptr<cudf::table> result_table;

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
    // There are both partitioned and unpartitioned files, so we need to merge the partitioned table
    // with the unpartitioned table.
    // This should be uncommon.
    GQE_LOG_WARN(
      "Both the partitioned and unpartitioned files are present. This usually means the dataset is "
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
  remove_dependencies();
}

parquet_write_task::parquet_write_task(query_context* query_context,
                                       int32_t task_id,
                                       int32_t stage_id,
                                       std::shared_ptr<task> input,
                                       std::vector<std::string> file_paths,
                                       std::vector<std::string> column_names,
                                       std::vector<cudf::data_type> data_types)
  : write_task_base(query_context, task_id, stage_id, input),
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

  cudf::io::parquet_chunked_writer(builder).write(input_table).close();

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
  query_context* query_context,
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

      auto read_task = std::make_unique<parquet_read_task>(query_context,
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

parquet_writeable_view::parquet_writeable_view(std::vector<std::string>* non_owning_file_paths)
  : writeable_view(), _non_owning_file_paths(non_owning_file_paths)
{
}

std::vector<std::unique_ptr<write_task_base>> parquet_writeable_view::get_write_tasks(
  std::vector<writeable_view::task_parameters>&& task_parameters,
  query_context* query_context,
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
      auto write_task = std::make_unique<parquet_write_task>(query_context,
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

};  // namespace storage
};  // namespace gqe
