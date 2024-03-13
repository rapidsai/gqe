/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
