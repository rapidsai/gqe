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

#include <kernel_launch.hpp>

#include <gqe/utility/error.hpp>

#include <cuda.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>

#include <array>
#include <cstdint>

/**
 * @brief Check CUDA function error code
 *
 * Cannot use GQE_CUDA_TRY, because it's for the runtime API. This macro
 * supports the driver API.
 */
#define CHECK_CUDA(statement, message)                     \
  {                                                        \
    assert_cuda((statement), message, __FILE__, __LINE__); \
  }

/**
 * @brief Helper for `CHECK_CUDA` macro
 */
void assert_cuda(CUresult code, const llvm::StringRef message, const llvm::StringRef file, int line)
{
  if (code != CUDA_SUCCESS) {
    const char* error;
    cuGetErrorString(code, &error);

    auto error_message = llvm::Twine{"CUDA error at "} + file + ":" + llvm::Twine(line) + " with " +
                         error + ": " + message;

    throw gqe::cuda_error{error_message.str()};
  }
}

CUcontext cuda_init_and_context(int device_id)
{
  CHECK_CUDA(cuInit(0), "Failed to initialize CUDA");

  // Get handle for device
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, device_id), "Failed to get CUDA device");

  CUcontext context;
  CHECK_CUDA(cuCtxCreate(&context, 0, device), "Failed to create CUDA context");

  return context;
}

int detect_architecture_by_id(int device_id)
{
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, device_id), "Failed to get CUDA device");

  return detect_architecture_by_device(device);
}

int detect_architecture_by_device(CUdevice device)
{
  int arch_major = 0;
  CHECK_CUDA(
    cuDeviceGetAttribute(&arch_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
    "Failed to detect device architecture");

  int arch_minor = 0;
  CHECK_CUDA(
    cuDeviceGetAttribute(&arch_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
    "Failed to detect device architecture");

  return arch_major * 10 + arch_minor;
}

void launch_kernel_ctx_dependent(llvm::ArrayRef<char> cubin, const char* kernel_name)
{
  CUmodule module;
  CHECK_CUDA(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr),
             "Failed to load module");

  CUfunction function;
  CHECK_CUDA(cuModuleGetFunction(&function, module, kernel_name), "Failed to get function");

  std::array<void*, 0> args{};

  // 3 dimensions of grid size and block size are all set to `1`.
  //
  // The static shared memory reservation is `0`.
  //
  // Setting the stream to `nullptr` launches on default stream.
  //
  // Kernel parameters are provided.
  //
  // No extra options are provided.
  CHECK_CUDA(cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args.data(), nullptr),
             "Failed to launch kernel");

  CHECK_CUDA(cuCtxSynchronize(), "Failed to synchronize");

  CHECK_CUDA(cuModuleUnload(module), "Failed to unload module");
}

void launch_kernel_ctx_independent(llvm::ArrayRef<char> cubin, const char* kernel_name)
{
  CUlibrary library;
  CHECK_CUDA(cuLibraryLoadData(&library, cubin.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
             "Failed to load library");

  uint32_t num_kernels;
  CHECK_CUDA(cuLibraryGetKernelCount(&num_kernels, library), "Failed to get num kernels");
  llvm::errs() << "num kernels: " << num_kernels << "\n";

  CUkernel kernel;
  CHECK_CUDA(cuLibraryGetKernel(&kernel, library, kernel_name), "Failed to load kernel");

  std::array<void*, 0> args{};

  // Context-independent kernels can be launched directly, instead of first
  // retrieving a CUfunction. See the CUDA docs for `cuLaunchKernel`.
  //
  // 3 dimensions of grid size and block size are all set to `1`.
  //
  // The static shared memory reservation is `0`.
  //
  // Setting the stream to `nullptr` launches on default stream.
  //
  // Kernel parameters are provided.
  //
  // No extra options are provided.
  CHECK_CUDA(
    cuLaunchKernel(
      reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1, 1, 1, 0, nullptr, args.data(), nullptr),
    "Failed to launch kernel");

  CHECK_CUDA(cuCtxSynchronize(), "Failed to synchronize");

  CHECK_CUDA(cuLibraryUnload(library), "Failed to unload library");
}
