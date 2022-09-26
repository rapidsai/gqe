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

#include <gqe/logical/from_substrait.hpp>

void topological_sort(gqe::logical::relation* root,
                      std::unordered_set<gqe::logical::relation*>& visited,
                      std::stack<gqe::logical::relation*>& stack);

std::vector<gqe::logical::relation*> ordered_relation_list(gqe::logical::relation* root);
