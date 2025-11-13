/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <gqe/logical/from_substrait.hpp>

#include <stack>

void topological_sort(gqe::logical::relation* root,
                      std::unordered_set<gqe::logical::relation*>& visited,
                      std::stack<gqe::logical::relation*>& stack);

std::vector<gqe::logical::relation*> ordered_relation_list(gqe::logical::relation* root);

/**
 * @brief Given a logical plan rooted at relation `root`, print in json format
 *
 * @param root The root relation of the plan to print
 */
void print_plan(gqe::logical::relation* root);
