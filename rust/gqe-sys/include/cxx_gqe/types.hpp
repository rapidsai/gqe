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

#pragma once

#include <cudf/types.hpp>

#include <gqe/executor/optimization_parameters.hpp>
#include <gqe/types.hpp>

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

}  // namespace cxx_gqe
