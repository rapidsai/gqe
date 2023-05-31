/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/read.hpp>
#include <gqe/expression/column_reference.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/logger.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <cassert>
#include <filesystem>
#include <memory>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <unordered_set>

namespace gqe {

read_task_base::read_task_base(int32_t task_id,
                               int32_t stage_id,
                               std::vector<std::shared_ptr<task>> subquery_tasks)
  : task(task_id, stage_id, {}, std::move(subquery_tasks))
{
}

parquet_read_task::parquet_read_task(int32_t task_id,
                                     int32_t stage_id,
                                     std::vector<std::string> file_paths,
                                     std::vector<std::string> column_names,
                                     std::vector<cudf::data_type> data_types,
                                     std::unique_ptr<expression> partial_filter,
                                     std::vector<std::shared_ptr<task>> subquery_tasks)
  : read_task_base(task_id, stage_id, std::move(subquery_tasks)),
    _file_paths(std::move(file_paths)),
    _column_names(std::move(column_names)),
    _data_types(std::move(data_types)),
    _partial_filter(std::move(partial_filter))
{
}

struct file_filter_functor {
  /**
   * @brief Filter out irrelevant files.
   *
   * @param[out] file_filter Boolean vector indicating whether each file should be included.
   * @param[in] needles (file_idx, value) pairs to be tested.
   * @param[in] acceptable_value_col Files should be included if the value is in this list.
   */
  template <typename T>
  void operator()(std::vector<bool>& file_filter,
                  std::vector<std::pair<size_t, size_t>> const& needles,
                  cudf::column_view const& acceptable_value_col)
  {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      std::vector<T> haystack(acceptable_value_col.size());
      GQE_CUDA_TRY(cudaMemcpy(haystack.data(),
                              acceptable_value_col.data<T>(),
                              acceptable_value_col.size() * sizeof(T),
                              cudaMemcpyDefault));

      // construct a hash set to easily search the needles. Make it 2x oversized to decrease
      // construction time
      std::unordered_set<size_t> haystack_set(
        haystack.begin(), haystack.end(), 2 * haystack.size());

      // Filter out files that are needles and are not present in the haystack
      for (auto const& needle : needles) {
        file_filter[needle.first] = haystack_set.find(needle.second) != haystack_set.end();
      }
    }
  }
};

std::unique_ptr<cudf::table> parquet_read_task::table_from_parquet(
  std::vector<std::string> const& file_paths) const
{
  if (file_paths.size() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_columns;
    for (auto const& column_type : _data_types)
      empty_columns.push_back(cudf::make_empty_column(column_type));
    return std::make_unique<cudf::table>(std::move(empty_columns));
  }

  auto const num_columns = _column_names.size();
  auto source            = cudf::io::source_info(file_paths);

  auto options = cudf::io::parquet_reader_options::builder(source);
  options.columns(_column_names);

  auto table_with_metadata = cudf::io::read_parquet(options);
  auto read_columns        = table_with_metadata.tbl->release();
  assert(read_columns.size() == num_columns);

  // Convert the read columns into the specified types if necessary
  if (_data_types.size() != num_columns)
    throw std::length_error("data_types must have the same length as the number of columns");

  std::vector<std::unique_ptr<cudf::column>> converted_columns;
  converted_columns.reserve(num_columns);
  for (std::size_t column_idx = 0; column_idx < num_columns; column_idx++) {
    auto const column_view   = read_columns[column_idx]->view();
    auto const expected_type = _data_types[column_idx];

    if (column_view.type() == expected_type) {
      converted_columns.push_back(std::move(read_columns[column_idx]));
    } else {
      converted_columns.push_back(cudf::cast(column_view, expected_type));
    }
  }
  assert(converted_columns.size() == num_columns);

  return std::make_unique<cudf::table>(std::move(converted_columns));
}

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

void parquet_read_task::execute()
{
  auto const partial_filter = _partial_filter.get();

  // Indicates whether each file should be loaded
  std::vector<bool> file_filter(_file_paths.size(), true);

  if (partial_filter != nullptr) {
    // FIXME: Only support in_predicate_expression
    if (partial_filter->type() != expression::expression_type::subquery) {
      throw std::runtime_error("Partial filter expression must be a subquery expression");
    }
    auto const partial_filter_subquery = dynamic_cast<const subquery_expression*>(partial_filter);

    // Execute subquery tasks
    prepare_subqueries();
    auto const subquery_tasks = subqueries();
    auto const expressions    = partial_filter_subquery->children();

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

    // Get the available values for the column we've partitioned along
    std::vector<std::pair<size_t, size_t>> needles;
    needles.reserve(_file_paths.size());
    std::regex valid_filename("(" + column_name + "=)([A-Za-z0-9_]+)");

    for (std::size_t file_idx = 0; file_idx < _file_paths.size(); ++file_idx) {
      std::filesystem::path file_path{_file_paths[file_idx]};
      std::string filename(file_path.filename());

      std::smatch matches;
      std::regex_search(_file_paths[file_idx], matches, valid_filename);
      // make sure both subexpressions (column name and value) are matched
      if (matches.size() == 3) {
        try {
          std::size_t value = stoull(matches[2]);
          needles.emplace_back(file_idx, value);
        } catch (const std::invalid_argument&) {
          if (matches[2] == "__HIVE_DEFAULT_PARTITION__") {
            // Filter out NULLs
            file_filter[file_idx] = false;
          }
        }
      }
    }

    // FIXME: assuming the resulting table has one column
    // retrieve result of subquery task
    auto const acceptable_value_table = cudf::drop_nulls(
      subquery_tasks[partial_filter_subquery->relation_index()]->result().value(), {0});
    auto const acceptable_value_col = acceptable_value_table->view().column(0);

    // FIXME: Currently only filter on a file-by-file basis.
    // Note that if there are no needles (the input tables are not partitioned), `file_filter`
    // contains all `true` so no files are filtered out.
    cudf::type_dispatcher(acceptable_value_col.type(),
                          file_filter_functor{},
                          file_filter,
                          needles,
                          acceptable_value_col);
  }

  std::vector<std::string> filtered_file_paths;
  filtered_file_paths.reserve(_file_paths.size());
  for (size_t file_idx = 0; file_idx < _file_paths.size(); file_idx++) {
    if (file_filter[file_idx]) filtered_file_paths.push_back(_file_paths[file_idx]);
  }

  auto loaded_table = table_from_parquet(filtered_file_paths);

  GQE_LOG_TRACE(
    "Load {0} from files with {1} rows.", print_column_names(), loaded_table->num_rows());

  update_result_cache(std::move(loaded_table));
  remove_dependencies();
  remove_subqueries();
}

}  // namespace gqe
