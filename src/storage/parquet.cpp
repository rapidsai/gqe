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

#include <cstdint>
#include <gqe/storage/parquet.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gqe {

namespace storage {

parquet_table::parquet_table(std::vector<std::string> file_paths)
  : table(), _file_paths(std::make_shared<std::vector<std::string>>(std::move(file_paths)))
{
}

bool parquet_table::is_readable() const { return true; }
bool parquet_table::is_writeable() const { return true; }
int32_t parquet_table::max_concurrent_writers() const { return _file_paths->size(); }

std::unique_ptr<readable_view> parquet_table::readable_view()
{
  return std::unique_ptr<storage::readable_view>(new parquet_readable_view(_file_paths.get()));
}

std::unique_ptr<writeable_view> parquet_table::writeable_view()
{
  return std::unique_ptr<parquet_writeable_view>(new parquet_writeable_view(_file_paths.get()));
}

parquet_readable_view::parquet_readable_view(std::vector<std::string>* non_owning_file_paths)
  : readable_view(), _non_owning_file_paths(non_owning_file_paths)
{
}

std::unique_ptr<read_task_base> parquet_readable_view::get_read_task(
  int32_t task_id,
  int32_t stage_id,
  int32_t parallelism,
  int32_t instance_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  std::unique_ptr<gqe::expression> partial_filter,
  std::vector<std::shared_ptr<task>> subquery_tasks)
{
  assert(task_id >= 0);
  assert(stage_id >= 0);
  assert(parallelism >= 1);
  assert(instance_id >= 0);

  assert(_non_owning_file_paths->size() <= std::numeric_limits<int64_t>::max());
  int64_t const max_num_files_per_instance = utility::divide_round_up(
    static_cast<int64_t>(_non_owning_file_paths->size()), int64_t{parallelism});
  auto begin_offset = std::min(int64_t{instance_id} * max_num_files_per_instance,
                               static_cast<int64_t>(_non_owning_file_paths->size()));
  auto end_offset   = std::min(begin_offset + max_num_files_per_instance,
                             static_cast<int64_t>(_non_owning_file_paths->size()));

  std::vector<std::string> file_paths_task{_non_owning_file_paths->begin() + begin_offset,
                                           _non_owning_file_paths->begin() + end_offset};

  return std::make_unique<parquet_read_task>(task_id,
                                             stage_id,
                                             std::move(file_paths_task),
                                             std::move(column_names),
                                             std::move(data_types),
                                             std::move(partial_filter),
                                             std::move(subquery_tasks));
}

parquet_writeable_view::parquet_writeable_view(std::vector<std::string>* non_owning_file_paths)
  : writeable_view(), _non_owning_file_paths(non_owning_file_paths)
{
}

std::unique_ptr<write_task_base> parquet_writeable_view::get_write_task(
  int32_t task_id,
  int32_t stage_id,
  int32_t parallelism,
  int32_t instance_id,
  std::vector<std::string> column_names,
  std::vector<cudf::data_type> data_types,
  std::shared_ptr<task> input)
{
  assert(task_id >= 0);
  assert(stage_id >= 0);
  assert(parallelism >= 1);
  assert(instance_id >= 0);

  if (_non_owning_file_paths->size() < static_cast<std::size_t>(parallelism)) {
    throw std::logic_error("Less than one Parquet output file per worker");
  }

  assert(_non_owning_file_paths->size() <= std::numeric_limits<int64_t>::max());
  int64_t const max_num_files_per_instance = utility::divide_round_up(
    static_cast<int64_t>(_non_owning_file_paths->size()), int64_t{parallelism});
  auto begin_offset = std::min(int64_t{instance_id} * max_num_files_per_instance,
                               static_cast<int64_t>(_non_owning_file_paths->size()));
  auto end_offset   = std::min(begin_offset + max_num_files_per_instance,
                             static_cast<int64_t>(_non_owning_file_paths->size()));

  std::vector<std::string> file_paths_task{_non_owning_file_paths->begin() + begin_offset,
                                           _non_owning_file_paths->begin() + end_offset};

  return std::make_unique<parquet_write_task>(task_id,
                                              stage_id,
                                              std::move(input),
                                              std::move(file_paths_task),
                                              std::move(column_names),
                                              std::move(data_types));
}

};  // namespace storage

};  // namespace gqe
