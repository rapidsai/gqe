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

#include <cudf/types.hpp>

#include <map>
#include <vector>

struct ddl_t {
  std::vector<std::string> column_names;
  std::vector<cudf::data_type> column_types;
  std::vector<std::string> file_paths;
};

// TODO: Add more tables from TPC-DS
std::map<std::string, ddl_t> ddls = {
  {"SUPPLIER",
   {{"S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_NATIONKEY", "S_PHONE", "S_ACCTBAL", "S_COMMENT"},
    {cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::STRING),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::INT64),
     cudf::data_type(cudf::type_id::STRING)},
    // TODO: Build catalog with file paths for executor
    {"dummy_path"}}}};
