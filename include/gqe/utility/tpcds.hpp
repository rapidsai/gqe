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

#pragma once

#include <gqe/types.hpp>

#include <cudf/types.hpp>

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

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
 * @brief Helper type to map column names to data types
 */
using column_definition_type = std::pair<std::string, cudf::data_type>;

/**
 * @brief Map of the TPC-DS DDL
 *
 * @details See the [TPC-DS
 * reference](https://www.tpc.org/tpc_documents_current_versions/pdf/tpc-ds_v3.2.0.pdf) for details.
 *
 * @return A map, which associates table names to column definition vectors
 */
std::unordered_map<std::string, std::vector<column_definition_type>> const&
table_definitions() noexcept;

}  // namespace tpcds
}  // namespace gqe::utility