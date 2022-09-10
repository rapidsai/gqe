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

#include <gqe/logical/relation.hpp>

#include <cassert>
#include <utility>

namespace gqe {
namespace logical {

read_relation::read_relation(std::vector<std::shared_ptr<relation>> children,
                             std::vector<std::string> column_names,
                             std::vector<cudf::data_type> column_types,
                             std::variant<std::string, std::vector<file_or_files>> read_location)
  : relation(std::move(children)),
    _column_names(std::move(column_names)),
    _read_location(std::move(read_location)),
    _data_types(std::move(column_types))
{
}

project_relation::project_relation(std::vector<std::shared_ptr<relation>> children,
                                   std::vector<std::shared_ptr<expression>> output_expressions)
  : relation(std::move(children)), output_expressions(std::move(output_expressions))
{
}

}  // namespace logical
}  // namespace gqe