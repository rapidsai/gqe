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
#include <gqe/physical/relation.hpp>

#include <memory>

namespace gqe {
namespace physical {

/**
 * @brief Physical relation for getting a subset of rows based on conditions.
 */
class filter_relation : public relation {
 public:
  /**
   * @brief Construct a physical filter relation.
   *
   * @param[in] input Input table to be filtered.
   * @param[in] subquery_relations Subquery relations that are referenced within the `condition`
   * expression.
   * @param[in] condition A boolean expression evaluated on `input` to represent the filter
   * condition.
   */
  filter_relation(std::shared_ptr<relation> input,
                  std::vector<std::shared_ptr<relation>> subquery_relations,
                  std::unique_ptr<expression> condition)
    : relation({std::move(input)}, std::move(subquery_relations)), _condition(std::move(condition))
  {
  }

  /**
   * @brief Return the filter condition.
   */
  expression* condition_unsafe() { return _condition.get(); }

  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }

 private:
  std::unique_ptr<expression> _condition;
};

}  // namespace physical
}  // namespace gqe
