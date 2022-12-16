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

#include <gqe/logical/fetch.hpp>
#include <gqe/logical/read.hpp>
#include <gqe/optimizer/estimator.hpp>

#include <algorithm>
#include <stdexcept>

namespace gqe {

table_statistics estimator::operator()(logical::relation const* input_relation) const
{
  auto cache_iter = _cache.find(input_relation);
  if (cache_iter != _cache.end()) return cache_iter->second;

  table_statistics stats;

  // FIXME: Implement this switch statement using a visitor pattern.
  switch (input_relation->type()) {
    case logical::relation::relation_type::fetch: {
      auto fetch_input_relation = dynamic_cast<logical::fetch_relation const*>(input_relation);
      stats.num_rows            = fetch_input_relation->count();
      break;
    }
    case logical::relation::relation_type::sort:
    case logical::relation::relation_type::project:
    case logical::relation::relation_type::aggregate:  // FIXME: Currently assume keys are distinct
    case logical::relation::relation_type::filter:  // FIXME: Currently assume filtering in all rows
    {
      auto children = input_relation->children_unsafe();
      assert(children.size() == 1);
      stats = operator()(children[0]);
      break;
    }
    case logical::relation::relation_type::read: {
      auto read_input_relation = dynamic_cast<logical::read_relation const*>(input_relation);
      stats                    = _catalog->statistics(read_input_relation->table_name());
      break;
    }
    case logical::relation::relation_type::join: {
      // FIXME: Estimate join selectivity based on input statistics
      auto children = input_relation->children_unsafe();
      assert(children.size() == 2);
      auto const left_num_rows  = operator()(children[0]).num_rows;
      auto const right_num_rows = operator()(children[1]).num_rows;
      stats.num_rows            = std::max(left_num_rows, right_num_rows);
      break;
    }
    default: throw std::logic_error("Unknown relation encountered in the estimator");
  }

  _cache[input_relation] = stats;
  return stats;
}

}  // namespace gqe
