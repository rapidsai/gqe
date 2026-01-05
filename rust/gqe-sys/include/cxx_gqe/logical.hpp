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

#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/relation.hpp>

#include <cxx_gqe/api.hpp>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <memory>
#include <vector>

namespace cxx_gqe {

/*
 * @brief Directly exposes logical relation as an opaque type to Rust.
 */
using logical_relation = gqe::logical::relation;

/*
 * @brief Workaround for non-implemented `CxxVector<SharedPtr<Type>>`.
 *
 * See https://github.com/dtolnay/cxx/issues/774
 */
using shared_logical_relation = std::shared_ptr<logical_relation>;

/*
 * @brief Substrait parser wrapper.
 *
 * This class is exported to Rust as an opaque type. It exposes an FFI-compatible API of the GQE
 * substrait parser.
 */
class substrait_parser {
  friend std::unique_ptr<substrait_parser> new_substrait_parser(catalog& catalog);

 public:
  explicit substrait_parser(gqe::substrait_parser&& parser) : _parser(std::move(parser)) {}

  substrait_parser()                                   = delete;
  substrait_parser(substrait_parser const&)            = delete;
  substrait_parser& operator=(substrait_parser const&) = delete;

  std::unique_ptr<std::vector<shared_logical_relation>> from_file(const rust::Str substrait_file);

 private:
  gqe::substrait_parser _parser;
};

/*
 * @brief Returns a new substrait parser wrapper.
 */
std::unique_ptr<substrait_parser> new_substrait_parser(catalog& catalog);

}  // namespace cxx_gqe
