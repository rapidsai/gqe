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
