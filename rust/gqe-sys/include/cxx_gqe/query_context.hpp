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

  query_context()                                = delete;
  query_context(query_context const&)            = delete;
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
