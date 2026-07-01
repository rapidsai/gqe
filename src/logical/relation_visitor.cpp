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

namespace gqe::logical {

void relation_visitor::visit_children(relation* rel)
{
  for (auto* child : rel->children_unsafe()) {
    child->accept(*this);
  }
  for (auto* sub : rel->subqueries_unsafe()) {
    sub->accept(*this);
  }
}

void relation_visitor::visit_relation(relation* rel) { visit_children(rel); }

void relation_visitor::visit(aggregate_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(fetch_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(filter_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(join_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(project_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(read_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(set_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(sort_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(user_defined_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(window_relation* rel) { visit_relation(rel); }
void relation_visitor::visit(write_relation* rel) { visit_relation(rel); }

}  // namespace gqe::logical
