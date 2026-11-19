/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <gqe/physical/relation.hpp>

#include <cudf/types.hpp>

#include <cstdint>
#include <memory>

namespace gqe::physical {

/**
 * @brief Physical single-key top-K relation.
 *
 * Complex ordering expressions are projected into columns before this relation. `key_index`
 * identifies the projected ordering column. The relation retains `offset + count` candidates and
 * returns the ordered range `[offset, offset + count)`.
 */
class top_k_relation : public relation {
 public:
  top_k_relation(std::shared_ptr<relation> input,
                 cudf::size_type key_index,
                 cudf::order column_order,
                 cudf::null_order null_precedence,
                 int64_t offset,
                 int64_t count);

  [[nodiscard]] relation_type type() const noexcept override { return relation_type::top_k; }
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  [[nodiscard]] cudf::size_type key_index() const noexcept { return _key_index; }
  [[nodiscard]] cudf::order column_order() const noexcept { return _column_order; }
  [[nodiscard]] cudf::null_order null_precedence() const noexcept { return _null_precedence; }
  [[nodiscard]] int64_t offset() const noexcept { return _offset; }
  [[nodiscard]] int64_t count() const noexcept { return _count; }

  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;
  [[nodiscard]] std::string to_string() const override;

 private:
  cudf::size_type _key_index;
  cudf::order _column_order;
  cudf::null_order _null_precedence;
  int64_t _offset;
  int64_t _count;
};

}  // namespace gqe::physical
