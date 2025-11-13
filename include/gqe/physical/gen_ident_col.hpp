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

#include <memory>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for generating a unique row identifier column.
 */
class gen_ident_col_relation : public relation {
 public:
  /**
   * @brief Construct a physical gen_ident_col relation.
   *
   * @param[in] input Input table to which the identifier col will be appended
   */
  gen_ident_col_relation(std::shared_ptr<relation> input) : relation({std::move(input)}, {}) {}

  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @copydoc relation::output_data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> output_data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;
};

}  // namespace physical
}  // namespace gqe