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
namespace optimizer {
class optimization_rule;
}  // namespace optimizer
namespace logical {

class aggregate_relation : public relation {
  friend class gqe::optimizer::optimization_rule;

 public:
  using measure_type = std::pair<cudf::aggregation::Kind, expression*>;
  /**
   * @brief Construct a new aggregate relation object
   *
   * @param input_relation Input table to aggregate on.
   * @param subquery_relations Subquery relations that are referenced within the `keys`
   * and/or `measures` expressions.
   * @param keys Expressions evaluated on `input` to represent groups. Rows with the same keys
   * will be grouped together. Note that this argument can be an empty vector. In that case, all
   * rows belong to the same group (reductions).
   * @param measures List of `(op, expr)` pairs such that each `expr` will be evaluated on `input`
   * and then rows of the evaluated results in the same group will be aggregated together using
   * `op`. For example, the query `select sum(c3), avg(c4) from t1 group by c1, c2;` will have [c1,
   * c2] as `keys` and [(sum, c3), (avg, c4)] as `measures`
   */
  aggregate_relation(
    std::shared_ptr<relation> input_relation,
    std::vector<std::shared_ptr<relation>> subquery_relations,
    std::vector<std::unique_ptr<expression>> keys,
    std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> measures);

  /**
   * @copydoc relation::type()
   */
  [[nodiscard]] relation_type type() const noexcept override { return relation_type::aggregate; }

  /**
   * @copydoc relation::data_types()
   */
  [[nodiscard]] std::vector<cudf::data_type> data_types() const override;

  /**
   * @copydoc relation::to_string()
   */
  [[nodiscard]] std::string to_string() const override;

  /**
   * @brief Return the list of keys to group by
   *
   * @note The returned keys do not share ownership. This object must be kept alive for the
   * returned keys to be valid.
   *
   * @return List of group by keys
   */
  [[nodiscard]] std::vector<expression*> keys_unsafe() const noexcept;

  /**
   * @brief Return the list of measures
   *
   * Each measure is a pair of cudf aggregation operation kind and expression.
   * This indicates the type of aggregate operation to perform on each value.
   * For example, the query:
   *
   * `select c0, sum(c1) from table_name grouby c0;`
   *
   * will result in a plan with
   * `keys = {col_reference(0)}`
   * `measures = {cudf::aggregation::SUM : col_reference(1)}`
   *
   * @return List of aggregate measures
   */
  [[nodiscard]] std::vector<measure_type> measures_unsafe() const noexcept;

  /**
   * @copydoc relation::operator==(const relation& other)
   */
  bool operator==(const relation& other) const override;

 private:
  std::vector<std::unique_ptr<expression>> _keys;
  std::vector<std::pair<cudf::aggregation::Kind, std::unique_ptr<expression>>> _measures;
};

}  // namespace logical
}  // namespace gqe