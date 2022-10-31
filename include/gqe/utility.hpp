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

#include <memory>
#include <vector>

namespace gqe {
namespace utility {

/**
 * @brief Helper function for converting a vector of smart pointers to raw pointers.
 */
template <typename smart_ptr_type>
inline std::vector<typename smart_ptr_type::element_type*> to_raw_ptrs(
  std::vector<smart_ptr_type> const& ptrs)
{
  std::vector<typename smart_ptr_type::element_type*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

/**
 * @brief Helper function for converting a vector of smart pointers to const raw pointers.
 */
template <typename smart_ptr_type>
inline std::vector<typename smart_ptr_type::element_type const*> to_const_raw_ptrs(
  std::vector<smart_ptr_type> const& ptrs)
{
  std::vector<typename smart_ptr_type::element_type const*> raw_ptrs;
  raw_ptrs.reserve(ptrs.size());
  for (auto const& ptr : ptrs)
    raw_ptrs.push_back(ptr.get());
  return raw_ptrs;
}

}  // namespace utility
}  // namespace gqe
