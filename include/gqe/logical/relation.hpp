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

#include <gqe/expression/expression.hpp>

#include <cudf/types.hpp>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace gqe {
namespace logical {
class relation {
 public:
  enum class relation_type { fetch, sort, project, aggregation, join, read, filter };

  /**
   * @brief Construct a relation node.
   *
   * @param[in] children Child nodes of the new relation node.
   */
  relation(std::vector<std::shared_ptr<relation>> children) : _children(std::move(children)) {}

  virtual ~relation();
  relation(const relation&) = delete;
  relation& operator=(const relation&) = delete;

  /**
   * @brief Return the operator type of the relation.
   */
  [[nodiscard]] virtual relation_type type() const noexcept = 0;

  /**
   * @brief Return the output data types of this relation.
   *
   * @return A vector whose size is equal to the number of columns in the
   * output relation. Element `i` of the vector records the type of column
   * `i`.
   */
  [[nodiscard]] virtual std::vector<cudf::data_type> data_types() const = 0;

  /**
   * @brief Return the children nodes.
   *
   * @note The returned relations do not share ownership. This object must be kept alive for the
   * returned relations to be valid.
   */
  [[nodiscard]] std::vector<relation*> children() const noexcept
  {
    std::vector<relation*> children_to_return(_children.size());

    for (auto const& child : _children)
      children_to_return.push_back(child.get());

    return children_to_return;
  }

  /**
   * @brief Return the number of children
   */
  [[nodiscard]] std::size_t children_size() const noexcept { return _children.size(); }

 private:
  // Child nodes of the current relation
  std::vector<std::shared_ptr<relation>> _children;
};

class read_relation : public relation {
 public:
  /**
   * @brief A struct containing information about the local file(s) to read.
   *
   * @note This is only used if the Substrait's read_type is `LocalFiles`
   */
  struct file_or_files {
    // Available extensions
    // TODO: Add support for more formats
    enum struct file_extension { parquet };
    // Local path to the file or directory of files to read
    std::string file_path;
    // Extension
    file_extension extension;
  };

  /**
   * @brief Construct a read relation.
   */
  read_relation(std::vector<std::shared_ptr<relation>> children,
                std::vector<std::string> column_names,
                std::vector<cudf::data_type> column_types,
                std::variant<std::string, std::vector<file_or_files>> read_location);

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

 private:
  // List of columns to read
  std::vector<std::string> _column_names;
  // Where to read data from
  // If string is initialized, then reading from a named table
  // If the vector if file_or_files is initialized, then read from the list of file paths
  // TODO: Add accessors
  std::variant<std::string, std::vector<file_or_files>> _read_location;
  // Data types of columns in the output relation
  mutable std::vector<cudf::data_type> _data_types;
};

class project_relation : public relation {
 public:
  /**
   * @brief Constructs a projection relation.
   */
  project_relation(std::vector<std::shared_ptr<relation>> children,
                   std::vector<std::shared_ptr<expression>> output_expressions);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::project; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override
  {
    assert(this->children_size() ==
           1);  // There should only be one input relation to a projection relation
    if (!this->_data_types) {
      for (auto const& output_expression : output_expressions)
        this->_data_types.value().push_back(
          output_expression->data_type(this->children()[0]->data_types()));
    }
    return this->_data_types.value();
  }

  // List of one or more expressions to add to the input
  // This is usually used in SELECT and its order of selection
  std::vector<std::shared_ptr<expression>> output_expressions;
  // Data types of columns in the output relation
  mutable std::optional<std::vector<cudf::data_type>> _data_types;
};
}  // namespace logical
}  // namespace gqe