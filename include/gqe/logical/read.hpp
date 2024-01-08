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

#include <gqe/logical/relation.hpp>

namespace gqe {
namespace logical {

class read_relation : public relation {
 public:
  /**
   * @brief Construct a new read relation object
   *
   * @param subquery_relations Subquery relations that are referenced within the
   * `partial_filter` expression
   * @param column_names Names of the columns to be loaded
   * @param column_types Types of the columns to be loaded
   * @param table_name Name of the table to be loaded
   * @param partial_filter Expression determining which rows of the input data are to be loaded.
   * Note that this is an optimization hint and the read relation makes no guarantee that this
   * filter will be applied.
   */
  read_relation(std::vector<std::shared_ptr<relation>> subquery_relations,
                std::vector<std::string> column_names,
                std::vector<cudf::data_type> column_types,
                std::string table_name,
                std::unique_ptr<expression> partial_filter);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::read; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override
  {
    return this->_data_types;  // initialized in constructor
  }
  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the name of the table to read from
   *
   * @return Name of table to read from
   */
  [[nodiscard]] std::string table_name() const { return _table_name; }

  /**
   * @brief Return the names of the columns to read
   *
   * @return List of columns to read
   */
  [[nodiscard]] std::vector<std::string> column_names() const { return _column_names; }

  /**
   * @brief Return a raw pointer to partial filter
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   *
   * @return Filter hint for read relation
   */
  [[nodiscard]] expression* partial_filter_unsafe() const { return _partial_filter.get(); }

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::vector<std::string> _column_names;
  std::vector<cudf::data_type> _data_types;
  std::string _table_name;
  std::unique_ptr<expression> _partial_filter;
};

}  // namespace logical
}  // namespace gqe