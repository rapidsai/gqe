/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <gqe/types.hpp>

#include <string>
#include <vector>

namespace gqe {
/**
 * @brief Forward declaration
 */
struct column_traits;
}  // namespace gqe

namespace gqe::utility {

/**
 * @brief A single table's schema: columns plus any UNIQUE / PRIMARY KEY constraints.
 *
 * Each inner vector in `unique_keys` is one key-set; size 1 is a single-column key,
 * size >= 2 is a composite key. Matches the shape accepted by `catalog::register_table`.
 */
struct table_definition {
  std::vector<gqe::column_traits> columns;
  std::vector<std::vector<std::string>> unique_keys;
};

}  // namespace gqe::utility
