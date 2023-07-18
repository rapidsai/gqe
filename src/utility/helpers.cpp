/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/utility/helpers.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>

namespace gqe::utility {

namespace {

// Return true if two strings compare equal case insensitive
bool strcmp_insensitive(std::string const& first, std::string const& second)
{
  return std::equal(first.begin(),
                    first.end(),
                    second.begin(),
                    second.end(),
                    [](char char_first, char char_second) {
                      return std::tolower(char_first) == std::tolower(char_second);
                    });
}

}  // namespace

bool parse_boolean_env_variable(std::string const& env_variable, bool default_val)
{
  bool rtv            = default_val;
  auto const env_char = std::getenv(env_variable.c_str());
  if (env_char) {
    std::string env_string(env_char);
    if (strcmp_insensitive(env_string, "true") || strcmp_insensitive(env_string, "on") ||
        strcmp_insensitive(env_string, "yes")) {
      rtv = true;
    } else if (strcmp_insensitive(env_string, "false") || strcmp_insensitive(env_string, "off") ||
               strcmp_insensitive(env_string, "no")) {
      rtv = false;
    } else {
      throw std::runtime_error("Invalid value for environment variable: " + env_variable);
    }
  }
  return rtv;
}

}  // namespace gqe::utility
