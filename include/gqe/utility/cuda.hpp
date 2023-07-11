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

#include <rmm/cuda_device.hpp>

#include <memory>

namespace gqe {

namespace utility {

/**
 * @brief Return the current CUDA device ID.
 */
rmm::cuda_device_id current_cuda_device_id();

/**
 * @brief Return the CUDA device properties.
 */
std::unique_ptr<cudaDeviceProp> get_cuda_device_property(
  rmm::cuda_device_id id = current_cuda_device_id());

}  // namespace utility

}  // namespace gqe
