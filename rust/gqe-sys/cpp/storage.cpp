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

#include <cxx_gqe/storage.hpp>

#include <gqe/logical/read.hpp>
#include <gqe/logical/write.hpp>

#include <cudf/types.hpp>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <memory>
#include <string>
#include <vector>

namespace cxx_gqe {

std::shared_ptr<logical_relation> new_read_relation(
  rust::Slice<const std::shared_ptr<logical_relation>> subquery_relations,
  rust::Slice<const rust::String> column_names,
  rust::Slice<const type_id> column_types,
  const rust::Str table_name)
{
  return std::make_shared<gqe::logical::read_relation>(
    std::move(std::vector<std::shared_ptr<gqe::logical::relation>>(subquery_relations.begin(),
                                                                   subquery_relations.end())),
    std::move(std::vector<std::string>(column_names.begin(), column_names.end())),
    std::move(std::vector<cudf::data_type>(column_types.begin(), column_types.end())),
    std::move(std::string(table_name)),
    nullptr);
}

std::shared_ptr<logical_relation> new_write_relation(
  std::shared_ptr<logical_relation> input_relation,
  rust::Slice<const rust::String> column_names,
  rust::Slice<const type_id> column_types,
  const rust::Str table_name)
{
  return std::make_shared<gqe::logical::write_relation>(
    std::move(input_relation),
    std::move(std::vector<std::string>(column_names.begin(), column_names.end())),
    std::move(std::vector<cudf::data_type>(column_types.begin(), column_types.end())),
    std::move(std::string(table_name)));
}

}  // namespace cxx_gqe
