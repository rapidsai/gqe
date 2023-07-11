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

#include <gqe/utility/linux.hpp>

#include <cstddef>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace gqe {

namespace utility {

template <typename CharT, typename Traits>
void check_stream(const std::basic_istream<CharT, Traits>& istream)
{
  if (istream.fail()) {
    throw std::logic_error("Looks like there's a bug in the /proc/meminfo parser.");
  }
}

std::unordered_map<std::string, std::size_t> get_meminfo(const std::string& meminfo_path)
{
  const std::unordered_map<std::string, std::size_t> unit_size = {{"kB", 1024UL}};

  std::unordered_map<std::string, std::size_t> info_map;
  std::ifstream meminfo(meminfo_path);

  std::string name, value_str;
  std::size_t value;
  std::string unit;
  bool has_unit;

  while (std::getline(meminfo, name, ':') && std::getline(meminfo, value_str)) {
    std::stringstream ss(value_str);

    check_stream(ss >> value);

    if (ss >> unit) {
      has_unit = true;
    } else {
      has_unit = false;
    }

    info_map[name] = has_unit ? value * unit_size.at(unit) : value;
  }

  return info_map;
}

}  // namespace utility

}  // namespace gqe
