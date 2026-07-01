/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/optimizer/relation_properties.hpp>

#include <algorithm>
#include <cassert>

std::string gqe::optimizer::column_property::to_string(
  gqe::optimizer::column_property::property_id prop)
{
  switch (prop) {
    case gqe::optimizer::column_property::property_id::compressed: return "compressed";
    case gqe::optimizer::column_property::property_id::sorted: return "sorted";
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

void gqe::optimizer::relation_properties::add_unique_key(std::vector<cudf::size_type> key)
{
  if (key.empty()) return;
  std::sort(key.begin(), key.end());
  key.erase(std::unique(key.begin(), key.end()), key.end());
  if (std::find(_unique_keys.begin(), _unique_keys.end(), key) != _unique_keys.end()) return;
  _unique_keys.push_back(std::move(key));
}

void gqe::optimizer::relation_properties::remove_unique_key(
  std::vector<cudf::size_type> const& key) noexcept
{
  auto sorted_key = key;
  std::sort(sorted_key.begin(), sorted_key.end());
  auto it = std::find(_unique_keys.begin(), _unique_keys.end(), sorted_key);
  if (it != _unique_keys.end()) _unique_keys.erase(it);
}

bool gqe::optimizer::relation_properties::covers_unique_key(
  std::unordered_set<cudf::size_type> const& cols) const noexcept
{
  return std::any_of(_unique_keys.begin(), _unique_keys.end(), [&](auto const& key) {
    // `add_unique_key` ignores empty keys, so every registered key-set has at least one column.
    // `all_of` would otherwise return true for an empty key-set.
    assert(!key.empty());
    return std::all_of(key.begin(), key.end(), [&](auto c) { return cols.contains(c); });
  });
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
  if (!_unique_keys.empty()) {
    column_properties_str += ",\n\t\"unique_keys\" : [";
    for (std::size_t ki = 0; ki < _unique_keys.size(); ++ki) {
      column_properties_str += '[';
      for (std::size_t ci = 0; ci < _unique_keys[ki].size(); ++ci) {
        column_properties_str += std::to_string(_unique_keys[ki][ci]);
        if (ci + 1 < _unique_keys[ki].size()) column_properties_str += ',';
      }
      column_properties_str += ']';
      if (ki + 1 < _unique_keys.size()) column_properties_str += ',';
    }
    column_properties_str += ']';
  }
  column_properties_str += "}}";
  return column_properties_str;
}
