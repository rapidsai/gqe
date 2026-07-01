/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <gqe/physical/aggregate.hpp>
#include <gqe/physical/fetch.hpp>
#include <gqe/physical/filter.hpp>
#include <gqe/physical/gen_ident_col.hpp>
#include <gqe/physical/join.hpp>
#include <gqe/physical/project.hpp>
#include <gqe/physical/read.hpp>
#include <gqe/physical/set.hpp>
#include <gqe/physical/shuffle.hpp>
#include <gqe/physical/sort.hpp>
#include <gqe/physical/user_defined.hpp>
#include <gqe/physical/window.hpp>
#include <gqe/physical/write.hpp>

namespace gqe::physical {

void relation_visitor::visit_children(relation* rel)
{
  for (auto* child : rel->children_unsafe()) {
    child->accept(*this);
  }
  for (auto* sub : rel->subqueries_unsafe()) {
    sub->accept(*this);
  }
}

void relation_visitor::visit(read_relation* rel) { visit_children(rel); }
void relation_visitor::visit(write_relation* rel) { visit_children(rel); }
void relation_visitor::visit(broadcast_join_relation* rel) { visit_children(rel); }
void relation_visitor::visit(shuffle_join_relation* rel) { visit_children(rel); }
void relation_visitor::visit(project_relation* rel) { visit_children(rel); }
void relation_visitor::visit(concatenate_sort_relation* rel) { visit_children(rel); }
void relation_visitor::visit(filter_relation* rel) { visit_children(rel); }
void relation_visitor::visit(concatenate_aggregate_relation* rel) { visit_children(rel); }
void relation_visitor::visit(fetch_relation* rel) { visit_children(rel); }
void relation_visitor::visit(union_all_relation* rel) { visit_children(rel); }
void relation_visitor::visit(user_defined_relation* rel) { visit_children(rel); }
void relation_visitor::visit(window_relation* rel) { visit_children(rel); }
void relation_visitor::visit(gen_ident_col_relation* rel) { visit_children(rel); }
void relation_visitor::visit(shuffle_relation* rel) { visit_children(rel); }

}  // namespace gqe::physical
