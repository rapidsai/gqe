/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/catalog.hpp>

#include <stdexcept>

namespace gqe {

void catalog::register_table(std::string table_name,
                             std::vector<std::pair<std::string, cudf::data_type>> const& columns,
                             std::vector<std::string> const& file_paths,
                             file_format_type file_format)
{
  if (_tables_info.find(table_name) != _tables_info.end())
    throw std::logic_error("table \"" + table_name + "\" is already registered");

  table_info_type table_info;
  for (auto const& column : columns) {
    auto const& column_name = column.first;
    auto const& column_type = column.second;
    if (table_info._column_name_to_type.find(column_name) != table_info._column_name_to_type.end())
      throw std::logic_error("column name already exists when registering table");
    table_info._column_name_to_type[column_name] = column_type;
  }
  table_info._file_paths  = file_paths;
  table_info._file_format = file_format;

  _tables_info[table_name] = std::move(table_info);
}

cudf::data_type catalog::column_type(std::string const& table_name,
                                     std::string const& column_name) const
{
  auto const table_info_iter = _tables_info.find(table_name);

  if (table_info_iter == _tables_info.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");
  auto const& column_name_to_type = table_info_iter->second._column_name_to_type;

  auto const column_type_iter = column_name_to_type.find(column_name);
  if (column_type_iter == column_name_to_type.end())
    throw std::logic_error("cannot find column \"" + column_name + "\" of table \"" + table_name +
                           "\" in the catalog");

  return column_type_iter->second;
}

std::vector<std::string> catalog::file_paths(std::string const& table_name) const
{
  auto const table_info_iter = _tables_info.find(table_name);

  if (table_info_iter == _tables_info.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  return table_info_iter->second._file_paths;
}

file_format_type catalog::file_format(std::string const& table_name) const
{
  auto const table_info_iter = _tables_info.find(table_name);

  if (table_info_iter == _tables_info.end())
    throw std::logic_error("cannot find table \"" + table_name + "\" in the catalog");

  return table_info_iter->second._file_format;
}

}  // namespace gqe
