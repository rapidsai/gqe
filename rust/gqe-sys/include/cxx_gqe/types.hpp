/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/types.hpp>

#include <cudf/types.hpp>

namespace cxx_gqe {

/*
 * ==========================================================
 *   Forward declarations of types exported by Rust to C++.
 * ==========================================================
 */

/**
 * @brief Forward declaration of Rust `OptimizationParameters` exported to C++.
 */
class optimization_parameters;

/*
 * @brief Storage kind wrapper.
 *
 * This wrapper represents the C++ `std::variant` as an `enum`.
 */
enum class storage_kind_type : int32_t;

/*
 * @brief Partitioning schema kind wrapper.
 *
 * This wrapper represents the C++ `std::variant` as an `enum`.
 */
enum class partitioning_schema_kind_type : int32_t;

/*
 * @brief Column schema helper type.
 *
 * This type summarizes the column name and column data type, because there is no shared
 * `std::tuple` type.
 */
struct column_schema;

/*
 * ======================================
 *   Shared types between C++ and Rust.
 * ======================================
 */

/*
 * @brief Void type helper.
 *
 * Shares `void` with Rust.
 *
 * See: https://github.com/dtolnay/cxx/issues/1049#issuecomment-1312854737
 */
using c_void = void;

/*
 * @brief Data type helper.
 *
 * Shares `cudf::type_id` with Rust.
 */
using type_id = cudf::type_id;

/*
 * @brief Page kind helper.
 *
 * Shares `gqe::page_kind::type` with Rust.
 */
using page_kind_type = gqe::page_kind::type;

/*
 * @brief Compression format helper.
 *
 * Shares `gqe::compression_format` with Rust.
 */
using compression_format = gqe::compression_format;

/*
 * @brief IO engine type helper.
 *
 * Shares `gqe::io_engine_type` with Rust.
 */
using io_engine_type = gqe::io_engine_type;

}  // namespace cxx_gqe
