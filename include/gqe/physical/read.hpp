/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/physical/relation.hpp>

#include <string>
#include <vector>

namespace gqe {

namespace physical {

/**
 * @brief Physical relation for loading data from files.
 */
class read_relation : public relation {
 public:
  /**
   * @brief Construct a new physical read relation.
   *
   * @param[in] table_name Name of the table to be loaded.
   * @param[in] column_names Names of the columns to be loaded.
   */
  read_relation(std::string table_name, std::vector<std::string> column_names)
    : relation({}), _table_name(std::move(table_name)), _column_names(std::move(column_names))
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the name of the table to load.
   */
  [[nodiscard]] std::string table_name() const noexcept { return _table_name; }

  /**
   * @brief Return the names of the columns to load.
   */
  [[nodiscard]] std::vector<std::string> column_names() const noexcept { return _column_names; }

 private:
  std::string _table_name;
  std::vector<std::string> _column_names;
};

}  // namespace physical
}  // namespace gqe
