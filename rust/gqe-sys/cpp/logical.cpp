/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cxx_gqe/logical.hpp>

// Include wrappers of Rust std library types.
#include "rust/cxx.h"

#include <memory>
#include <string>
#include <vector>

namespace cxx_gqe {

std::unique_ptr<std::vector<shared_logical_relation>> substrait_parser::from_file(
  const rust::Str substrait_file)
{
  std::string cxx_file(substrait_file);
  return std::make_unique<std::vector<std::shared_ptr<logical_relation>>>(
    _parser.from_file(std::move(cxx_file)));
}

std::unique_ptr<substrait_parser> new_substrait_parser(catalog& catalog)
{
  return std::make_unique<substrait_parser>(std::move(gqe::substrait_parser(&catalog.get())));
}

}  // namespace cxx_gqe
