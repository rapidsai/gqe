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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace logical {

/**
 * @brief The fetch relation is used for limiting the number of rows returned.
 *
 * If the offset is specified, it will also return the row starting from the spcified
 * offset.
 */
class fetch_relation : public relation {
 public:
  /**
   * @brief Construct a new fetch relation object
   *
   * @param input_relation Input to the fetch relation
   * @param offset The offset experssed in number of records
   * @param count The number of records to return
   */
  fetch_relation(std::shared_ptr<relation> input_relation, int64_t offset, int64_t count);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::fetch; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the offset for the retrieval records
   *
   * @return The index of the starting row to be returned
   */
  [[nodiscard]] int64_t offset() const noexcept { return _offset; };

  /**
   * @brief Return the count for the retrieval records
   *
   * @return The number of rows to be returned
   */
  [[nodiscard]] int64_t count() const noexcept { return _count; };

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  int64_t _offset;
  int64_t _count;
};

}  // namespace logical
}  // namespace gqe
