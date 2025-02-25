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

#include <cuda.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/LogicalResult.h>

/**
 * @brief Helper to initialize CUDA and create a context
 *
 * @param[in] device_id A numeric device ID.
 *
 * @return The new CUDA context.
 */
CUcontext cuda_init_and_context(int device_id);

/**
 * @brief Helper to get CUDA compute capability of device by ID
 *
 * See:
 * https://github.com/rapidsai/rapids-cmake/blob/branch-24.12/rapids-cmake/cuda/detail/detect_architectures.cmake
 *
 * @param[in] device_id A numeric device ID.
 *
 * @return The compute capability, encoded as `10 * major + minor`. E.g., `85`.
 */
int detect_architecture_by_id(int device_id);

/**
 * @brief Helper to get CUDA compute capability of device by handle
 *
 * See:
 * https://github.com/rapidsai/rapids-cmake/blob/branch-24.12/rapids-cmake/cuda/detail/detect_architectures.cmake
 *
 * @param[in] device A CUDA device handle.
 *
 * @return The compute capability, encoded as `10 * major + minor`. E.g., `85`.
 */
int detect_architecture_by_device(CUdevice device);

/**
 * @brief Launch a kernel with CUDA Module API
 *
 * The Module API is the "old" way of launching kernels. It's assiciated with a
 * given CUDA Context.
 *
 * @param[in] cubin The device binary. Passing PTX is also possible, in which
 * case it will be compiled while loading the module.
 * @param[in] kernel_name The name of the kernel to call.
 */
void launch_kernel_ctx_dependent(llvm::ArrayRef<char> cubin, const char* kernel_name);

/**
 * Launch a kernel with CUDA Library API
 *
 * This function uses CUDA's context-independent module loading feature.
 *
 * See: https://developer.nvidia.com/blog/cuda-context-independent-module-loading/
 *
 * @param[in] cubin The device binary. Passing PTX is also possible, in which
 * case it will be compiled while loading the library.
 * @param[in] kernel_name The name of the kernel to call.
 */
void launch_kernel_ctx_independent(llvm::ArrayRef<char> cubin, const char* kernel_name);
