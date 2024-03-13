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

#include <cxx_gqe/api.hpp>

#include <gqe/logical/from_substrait.hpp>
#include <gqe/logical/relation.hpp>

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

  substrait_parser()                        = delete;
  substrait_parser(substrait_parser const&) = delete;
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
