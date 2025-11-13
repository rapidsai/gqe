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

#include <gqe/physical/relation.hpp>

#include <cstdint>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for getting a consecutive subset of rows based on `count` and `offset`.
 */
class fetch_relation : public relation {
 public:
  /**
   * @brief Construct a physical fetch relation.
   *
   * @param[in] input Input table.
   * @param[in] offset Row index from which the fetch starts.
   * @param[in] count Number of rows to retrieve starting from `offset`.
   */
  fetch_relation(std::shared_ptr<relation> input, int64_t offset, int64_t count)
    : relation({std::move(input)}, {}), _offset(offset), _count(count)
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the row index from which the fetch starts.
   */
  [[nodiscard]] int64_t offset() const noexcept { return _offset; }

  /**
   * @brief Return the number of rows to retrieve.
   */
  [[nodiscard]] int64_t count() const noexcept { return _count; }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

 private:
  int64_t _offset;
  int64_t _count;
};

}  // namespace physical
}  // namespace gqe
