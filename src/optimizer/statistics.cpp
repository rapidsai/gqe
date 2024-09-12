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

#include <gqe/optimizer/statistics.hpp>

#include <mutex>

namespace gqe {

void table_statistics_manager::add_rows(int rows)
{
  std::unique_lock latch_guard(_statistics_latch);
  _statistics.num_rows += rows;
}

table_statistics table_statistics_manager::statistics()
{
  std::shared_lock latch_guard(_statistics_latch);
  return _statistics;
}

}  // namespace gqe
