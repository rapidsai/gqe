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

#include <gqe/logical/relation.hpp>

#include <cudf/types.hpp>

#include <memory>

namespace gqe {
namespace logical {

class write_relation : public relation {
 public:
  /**
   * @brief Construct a new write relation object
   *
   * @param input_relation Input table to use as data source.
   * @param column_names Names of the columns to be written to.
   * @param column_types Data types of the columns to be written to.
   * @param table_name Name of the table to be written to.
   */
  write_relation(std::shared_ptr<relation> input_relation,
                 std::vector<std::string> column_names,
                 std::vector<cudf::data_type> column_types,
                 std::string table_name);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override;

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the name of the table to write to.
   *
   * @return Name of table to write to.
   */
  [[nodiscard]] std::string table_name() const;

  /**
   * @brief Return the names of the columns to write to.
   *
   * @return List of columns to write to.
   */
  [[nodiscard]] std::vector<std::string> column_names() const;

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _column_types;
  std::string _table_name;
};

}  // namespace logical
}  // namespace gqe
