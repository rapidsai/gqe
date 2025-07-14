/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
