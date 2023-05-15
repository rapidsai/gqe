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

#include <gqe/physical/relation.hpp>

#include <memory>
#include <vector>

namespace gqe {

namespace physical {

/**
 * @brief Physical relation for storing data to tables.
 */
class write_relation : public relation {
 public:
  /**
   * @brief Construct a new physical write relation.
   *
   * @param[in] input Input table that is written out.
   * @param[in] column_names Names of the columns to be loaded.
   * @param[in] table_name Name of the table to be written to.
   */
  write_relation(std::shared_ptr<relation> input,
                 std::vector<std::string> column_names,
                 std::string table_name)
    : relation({std::move(input)}, {}),
      _column_names(std::move(column_names)),
      _table_name(std::move(table_name))
  {
  }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

  /**
   * @brief Return the name of the table to store.
   */
  [[nodiscard]] std::string table_name() const noexcept { return _table_name; }

  /**
   * @brief Return the names of the columns to store.
   */
  [[nodiscard]] std::vector<std::string> column_names() const noexcept { return _column_names; }

 private:
  std::vector<std::string> _column_names;
  std::string _table_name;
};

}  // namespace physical
}  // namespace gqe
