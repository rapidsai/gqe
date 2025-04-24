/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
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
#include <gqe/utility/helpers.hpp>

#include <cudf/aggregation.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace gqe {
namespace physical {

/**
 * @brief Abstract base class for all physical aggregate relations.
 */
class aggregate_relation_base : public relation {
 public:
  /**
   * @brief Construct a physical aggregate relation.
   *
   * @param[in] input Input table to aggregate on.
   * @param[in] subquery_relations Subquery relations that are referenced within the `keys`
   * and/or `values` expressions.
   * @param[in] keys Expressions evaluated on `input` to represent groups. Rows with the same keys
   * will be grouped together. Note that this argument can be an empty vector. In that case, all
   * rows belong to the same group (reductions).
   * @param[in] values List of `(op, expr)` pairs such that each `expr` will be evaluated on `input`
   * and then rows of the evaluated results in the same group will be combined together using `op`.
   * @param[in] condition An optional boolean expression evaluated on `input` to represent the
   * filter condition. Note: That this is currently not supported for pure reductions
   */
  aggregate_relation_base(
    std::shared_ptr<relation> input,
    std::vector<std::shared_ptr<relation>> subquery_relations,
    std::vector<std::unique_ptr<expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> values,
    std::unique_ptr<expression> condition = nullptr)
    : relation({std::move(input)}, std::move(subquery_relations)),
      _keys(std::move(keys)),
      _values(std::move(values)),
      _condition(std::move(condition))
  {
  }

  /**
   * @brief Return the grouping keys.
   */
  std::vector<expression*> keys_unsafe() { return utility::to_raw_ptrs(_keys); }

  /**
   * @brief Returns the filter condition.
   */
  expression* condition_unsafe() { return _condition.get(); }

  /**
   * @brief Return the values to be aggregated on.
   */
  std::vector<std::pair<cudf::aggregation::Kind, expression*>> values_unsafe()
  {
    std::vector<std::pair<cudf::aggregation::Kind, expression*>> values;
    for (auto const& [kind, agg] : _values)
      values.emplace_back(kind, agg.get());
    return values;
  }

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> _values;
  std::unique_ptr<expression> _condition;
};

/**
 * @brief Apply-concat-apply aggregation.
 *
 * Apply an aggregation on each partition, concatenate the aggregation results into a single
 * partition, and finally apply another aggregation. The result would have one partition only.
 */
class concatenate_aggregate_relation : public aggregate_relation_base {
  using aggregate_relation_base::aggregate_relation_base;

 public:
  /**
   * @copydoc gqe::physical::relation::accept(relation_visitor&)
   */
  void accept(relation_visitor& visitor) override { visitor.visit(this); }
};

}  // namespace physical
}  // namespace gqe
