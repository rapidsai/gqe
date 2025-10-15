/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
