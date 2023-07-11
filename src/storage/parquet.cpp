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

#include <gqe/storage/parquet.hpp>
#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>

#include <cassert>
#include <cstdint>
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

std::vector<std::unique_ptr<read_task_base>> parquet_readable_view::get_read_tasks(
  std::vector<readable_view::task_parameters>&& task_parameters,
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

      auto read_task = std::make_unique<parquet_read_task>(task->task_id,
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
      auto write_task = std::make_unique<parquet_write_task>(task->task_id,
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
