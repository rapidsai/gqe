/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gqe/optimizer/estimator.hpp>

#include <gqe/logical/aggregate.hpp>
#include <gqe/logical/fetch.hpp>
#include <gqe/logical/filter.hpp>
#include <gqe/logical/join.hpp>
#include <gqe/logical/project.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/logical/set.hpp>
#include <gqe/logical/sort.hpp>
#include <gqe/logical/user_defined.hpp>
#include <gqe/logical/window.hpp>
#include <gqe/logical/write.hpp>
#include <gqe/optimizer/statistics.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace gqe {

class estimator::statistics_visitor : public logical::relation_visitor {
 public:
  statistics_visitor(estimator const& owner) : _owner{owner} {}

  [[nodiscard]] table_statistics const& result() const noexcept { return _result; }

  void visit(logical::fetch_relation* fetch) override { _result.num_rows = fetch->count(); }

  // FIXME: Currently assume keys are distinct
  void visit(logical::aggregate_relation* aggregate) override
  {
    _propagate_single_child(aggregate);
  }

  // FIXME: Currently assume filtering in all rows
  void visit(logical::filter_relation* filter) override { _propagate_single_child(filter); }

  void visit(logical::project_relation* project) override { _propagate_single_child(project); }
  void visit(logical::sort_relation* sort) override { _propagate_single_child(sort); }
  void visit(logical::window_relation* window) override { _propagate_single_child(window); }

  void visit(logical::read_relation* read) override
  {
    _result = _owner._catalog->statistics(read->table_name())->statistics();
  }

  void visit(logical::join_relation* join) override
  {
    // FIXME: Estimate join selectivity based on input statistics
    auto children = join->children_unsafe();
    assert(children.size() == 2);
    auto const left_num_rows  = _owner(children[0]).num_rows;
    auto const right_num_rows = _owner(children[1]).num_rows;
    _result.num_rows          = std::max(left_num_rows, right_num_rows);
  }

  void visit(logical::write_relation*) override
  {
    _result.num_rows = 0;  // write relation does not produce an output
  }

  void visit(logical::set_relation* set_op) override
  {
    auto children = set_op->children_unsafe();
    assert(children.size() == 2);
    auto const left_num_rows  = _owner(children[0]).num_rows;
    auto const right_num_rows = _owner(children[1]).num_rows;

    // Heuristic: since we don't know the input distribution, we cannot estimate the matching
    // rate, so the estimates below use the maximum sizes.
    switch (set_op->set_operator()) {
      case logical::set_relation::set_union:
      case logical::set_relation::set_union_all:
        // The output size of the union can be at most the two input sizes combined, when there
        // are no duplicates.
        _result.num_rows = left_num_rows + right_num_rows;
        break;
      case logical::set_relation::set_intersect:
        // An output of the intersection must appear in both input tables. Therefore, the
        // maximum output size is the smaller size of the inputs.
        _result.num_rows = std::min(left_num_rows, right_num_rows);
        break;
      case logical::set_relation::set_minus:
        // An output of the minus must appear in the left table. Therefore, the output size must
        // be at most the size of the left table.
        _result.num_rows = left_num_rows;
        break;
      default: throw std::logic_error("Unknown set relation encountered in the estimator");
    }
  }

  void visit(logical::user_defined_relation* user_defined) override
  {
    // Heuristic: we use the maximum input size as the output size
    auto children              = user_defined->children_unsafe();
    int64_t maximum_input_size = 0;
    for (auto const& child : children) {
      auto const child_size = _owner(child).num_rows;
      if (child_size > maximum_input_size) maximum_input_size = child_size;
    }
    _result.num_rows = maximum_input_size;
  }

 private:
  void _propagate_single_child(logical::relation* rel)
  {
    auto children = rel->children_unsafe();
    assert(children.size() == 1);
    _result = _owner(children[0]);
  }

  estimator const& _owner;
  table_statistics _result;
};

table_statistics estimator::operator()(logical::relation const* input_relation) const
{
  auto cache_iter = _cache.find(input_relation);
  if (cache_iter != _cache.end()) return cache_iter->second;

  statistics_visitor visitor{*this};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): visitor does not mutate input
  // Remove const_cast once we have a non-const accept method
  const_cast<logical::relation*>(input_relation)->accept(visitor);

  return _cache[input_relation] = visitor.result();
}

}  // namespace gqe
