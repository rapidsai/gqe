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

#pragma once

#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace gqe {
/**
 * @brief Forward declaration
 */
struct column_traits;
}  // namespace gqe

namespace gqe::utility {
namespace tpcds {

/**
 * @brief Type mappings between TPC-DS and GQE
 */
constexpr auto identifier_type = cudf::data_type(cudf::type_id::INT64);
constexpr auto integer_type    = cudf::data_type(cudf::type_id::INT64);
constexpr auto decimal_type    = cudf::data_type(cudf::type_id::FLOAT64);
constexpr auto string_type     = cudf::data_type(cudf::type_id::STRING);
constexpr auto date_type       = cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);

/**
 * @brief Map of the TPC-DS DDL
 *
 * @details See the [TPC-DS
 * reference](https://www.tpc.org/tpc_documents_current_versions/pdf/tpc-ds_v3.2.0.pdf) for details.
 *
 * @return A map, which associates table names to column definition vectors
 */
std::unordered_map<std::string, std::vector<gqe::column_traits>> const&
table_definitions() noexcept;

}  // namespace tpcds
}  // namespace gqe::utility
