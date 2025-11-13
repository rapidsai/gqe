/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/optimizer/relation_properties.hpp>

std::string gqe::optimizer::column_property::to_string(
  gqe::optimizer::column_property::property_id prop)
{
  switch (prop) {
    case gqe::optimizer::column_property::property_id::compressed: return "compressed";
    case gqe::optimizer::column_property::property_id::sorted: return "sorted";
    case gqe::optimizer::column_property::property_id::unique: return "unique";
    default: throw std::runtime_error("unsupported column property");
  }
}

void gqe::optimizer::relation_properties::add_column_property(
  cudf::size_type col_idx, column_property::property_id property) noexcept
{
  _columns_properties[property].insert(col_idx);
}

void gqe::optimizer::relation_properties::remove_column_property(
  cudf::size_type col_idx, column_property::property_id property) noexcept
{
  if (!_columns_properties.count(property)) return;
  _columns_properties[property].erase(col_idx);
  if (_columns_properties[property].empty()) _columns_properties.erase(property);
}

std::vector<gqe::optimizer::column_property::property_id>
gqe::optimizer::relation_properties::get_column_properties(cudf::size_type col_idx) const noexcept
{
  std::vector<column_property::property_id> properties;
  for (const auto& [property, indices] : _columns_properties) {
    if (indices.count(col_idx)) { properties.push_back(property); }
  }
  return properties;
}

std::unordered_set<cudf::size_type> gqe::optimizer::relation_properties::get_columns_with_property(
  column_property::property_id property) const noexcept
{
  if (!_columns_properties.count(property)) return {};
  return _columns_properties.at(property);
}

bool gqe::optimizer::relation_properties::check_column_property(
  cudf::size_type col_idx, column_property::property_id property) const noexcept
{
  if (!_columns_properties.count(property)) return false;
  auto& indices = _columns_properties.at(property);
  return indices.count(col_idx);
}

std::string gqe::optimizer::relation_properties::to_string() const
{
  std::string column_properties_str = "{\"Column properties\" : {\n";
  for (auto outer_iter = _columns_properties.begin(); outer_iter != _columns_properties.end();
       ++outer_iter) {
    column_properties_str +=
      "\t\"" + gqe::optimizer::column_property::to_string(outer_iter->first) + "\" : [";
    for (auto inner_iter = outer_iter->second.begin(); inner_iter != outer_iter->second.end();
         ++inner_iter) {
      column_properties_str += "\"" + std::to_string(*inner_iter) + "\"";
      if (std::distance(inner_iter, outer_iter->second.end()) > 1) column_properties_str += ", ";
    }
    column_properties_str += "]";
    if (std::distance(outer_iter, _columns_properties.end()) > 1) column_properties_str += ",\n";
  }
  column_properties_str += "}}";
  return column_properties_str;
}
