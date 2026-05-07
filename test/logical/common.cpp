/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "common.hpp"

#include <string>

void topological_sort(gqe::logical::relation* root,
                      std::unordered_set<gqe::logical::relation*>& visited,
                      std::stack<gqe::logical::relation*>& stack)
{
  // Mark current relation as visited
  visited.insert(root);

  // Traverse child relations
  for (auto child_rel : root->children_unsafe()) {
    auto it = visited.find(child_rel);
    if (it == visited.end()) {
      // Has not been visited
      topological_sort(child_rel, visited, stack);
    }
  }
  stack.push(root);
}

std::vector<gqe::logical::relation*> ordered_relation_list(gqe::logical::relation* root)
{
  std::vector<gqe::logical::relation*> ordered_relations;
  std::stack<gqe::logical::relation*> stack;
  std::unordered_set<gqe::logical::relation*> visited;

  // Start with the root
  topological_sort(root, visited, stack);

  // Reverse the order from the stack
  while (!stack.empty()) {
    ordered_relations.push_back(stack.top());
    stack.pop();
  }

  return ordered_relations;
}

std::string build_plan_string(gqe::logical::relation* root,
                              std::unordered_set<gqe::logical::relation*>& visited)
{
  // Mark current relation as visited
  visited.insert(root);

  // String to return
  std::string plan_str = root->to_string();

  // Traverse child relations
  for (auto child_rel : root->children_unsafe()) {
    auto it = visited.find(child_rel);
    if (it == visited.end()) {
      // Has not been visited
      plan_str += build_plan_string(child_rel, visited) + ", ";
    }
  }

  return "{" + plan_str + "}";
}

void print_plan(gqe::logical::relation* root)
{
  std::unordered_set<gqe::logical::relation*> visited;
  std::string plan_str = build_plan_string(root, visited);
  std::cout << plan_str << std::endl;
}
