/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/storage/table.hpp>

#include <memory>
#include <string_view>

namespace gqe::storage {

/**
 * @brief Interface for retrieving storage tables by name.
 *
 * Concrete implementations include the catalog (for locally registered tables)
 * and the remote table cache (for tables mapped from other task managers).
 *
 * This is a pure interface — implementations must not add state to this class.
 */
class table_provider {
 public:
  virtual ~table_provider() = default;

  /**
   * @brief Retrieve a storage table by name.
   *
   * @param[in] table_name Name of the table.
   * @return A shared pointer to the table.
   * @throws std::logic_error if the table is not found.
   */
  [[nodiscard]] virtual std::shared_ptr<table> get_table(std::string_view table_name) const = 0;
};

}  // namespace gqe::storage
