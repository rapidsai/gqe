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

#include <boost/interprocess/managed_shared_memory.hpp>

namespace gqe::utility {

/**
 * @brief Find an object in a shared memory segment.
 *
 * Throws an error if the object is not found.
 *
 * Returns a raw pointer to the object.
 */
template <typename T>
T* find_object(boost::interprocess::managed_shared_memory* segment, std::string name)
{
  auto found = segment->find<T>(name.c_str());
  if (found.first == nullptr) {
    throw std::runtime_error("Object with name " + name + " not found in shared memory segment!");
  }
  return found.first;
}

}  // namespace gqe::utility
