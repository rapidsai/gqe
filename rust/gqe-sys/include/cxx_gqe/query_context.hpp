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

#include <gqe/query_context.hpp>

#include <utility>

namespace cxx_gqe {
/*
 * @brief Query context wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * query context.
 */
class query_context {
 public:
  explicit query_context(gqe::query_context&& context) : _context(std::move(context)) {}

  query_context()                     = delete;
  query_context(query_context const&) = delete;
  query_context& operator=(query_context const&) = delete;

  /*
   * @brief Returns the C++ query context.
   *
   * This is a helper method used in the C++ bindings to convert from the wrapper to the actual
   * object.
   */
  inline gqe::query_context& get() { return _context; }

 private:
  gqe::query_context _context;
};

/*
 * @brief Returns a new query context wrapper.
 */
std::unique_ptr<query_context> new_query_context(
  cxx_gqe::optimization_parameters const& parameters);
}  // namespace cxx_gqe
