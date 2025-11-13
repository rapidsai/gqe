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

#include <gqe/storage/readable_view.hpp>
#include <gqe/storage/writeable_view.hpp>

#include <cstdint>
#include <memory>

namespace gqe {

namespace storage {

/**
 * @brief Interface to store and manage the data of a table.
 *
 * Data stored in the database are owned by a table. The table provides a
 * unified interface to manage the data for all table kinds.
 *
 * As table is an abstract class, subtypes of table define the concrete
 * management methods and data access methods.
 */
class table {
 public:
  /**
   * @brief Create a new table.
   */
  table() = default;

  table(const table&)            = delete;
  table& operator=(const table&) = delete;

  virtual ~table() = default;

  /**
   * @brief Return whether the table is readable.
   *
   * If true, a readable view is available.
   */
  [[nodiscard]] virtual bool is_readable() const = 0;

  /**
   * @brief Return if the table is writeable.
   *
   * If true, a writeable view is available.
   */
  [[nodiscard]] virtual bool is_writeable() const = 0;

  /**
   * @brief Return the maximum number of concurrent readers.
   */
  [[nodiscard]] virtual int32_t max_concurrent_readers() const = 0;

  /**
   * @brief Return the maximum number of concurrent writers.
   */
  [[nodiscard]] virtual int32_t max_concurrent_writers() const = 0;

  /**
   * @brief Return a readable view to the table.
   */
  virtual std::unique_ptr<storage::readable_view> readable_view() = 0;

  /**
   * @brief Return a writeable view to the table.
   */
  virtual std::unique_ptr<storage::writeable_view> writeable_view() = 0;
};

};  // namespace storage

};  // namespace gqe
