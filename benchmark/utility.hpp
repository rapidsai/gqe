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

#pragma once

#include <gqe/utility/error.hpp>

namespace gqe::benchmark {

inline std::size_t get_memory_pool_size()
{
  std::size_t free_memory, total_memory;
  GQE_CUDA_TRY(cudaMemGetInfo(&free_memory, &total_memory));
  auto pool_size =
    total_memory / 284 * 256;  // ~90% of the total memory and have 256-byte alignment
  return pool_size;
}

}  // namespace gqe::benchmark
