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

class filter_relation : public relation {
 public:
  /**
   * @brief Construct a new filter relation object
   *
   * @param input_relation Input relation to apply filter on
   * @param condition Filter expression to apply to the input
   */
  filter_relation(std::shared_ptr<relation> input_relation,
                  std::vector<std::shared_ptr<relation>> subquery_relations,
                  std::unique_ptr<expression> condition);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::filter; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override { return _data_types; };

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the filter condition for this relation
   *
   * The condition defines which rows to return
   *
   * @return Filter condition
   *
   * @note This function does not share ownership. The caller is responsible for keeping
   * the returned pointer alive.
   */
  [[nodiscard]] expression* condition() const noexcept { return _condition.get(); }

 private:
  std::vector<cudf::data_type> _data_types;
  std::unique_ptr<expression> _condition;
};

}  // namespace logical
}  // namespace gqe