/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <concepts>
#include <cstddef>
#include <functional>

namespace gqe::utility {

/**
 * @brief A type hashable via `std::hash`.
 *
 * The standard library has no hashability concept, therefore it's taken from the C++ Reference
 * example.
 *
 * Source: https://en.cppreference.com/cpp/language/constraints
 */
template <typename T>
concept hashable = requires(T const a) {
  { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

}  // namespace gqe::utility
