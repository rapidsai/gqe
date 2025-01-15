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

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/error.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime_api.h>

#include <memory>

namespace gqe {

namespace utility {

rmm::cuda_device_id current_cuda_device_id()
{
  int id{};
  GQE_CUDA_TRY(cudaGetDevice(&id));
  return rmm::cuda_device_id{id};
}

}  // namespace utility

}  // namespace gqe
