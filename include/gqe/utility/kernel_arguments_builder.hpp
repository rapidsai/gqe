/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier:
 * LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <cstdint>
#include <gqe/compiler/Types.hpp>

#include <cudf/column/column_view.hpp>

#include <llvm/ADT/SmallVector.h>

#include <rmm/device_scalar.hpp>

#include <cassert>
#include <memory>

namespace gqe {
namespace utility {

/**
 * @brief A owning kernel argument wrapper.
 *
 * Wraps a kernel argument, for which it implements the conversion to `void*`.
 *
 * Owns the underlying argument to keep it alive until the kernel is launched.
 */
class KernelArg {
 public:
  /**
   * @brief The output container for kernel arguments.
   *
   * Size chosen as 11, because usually a kernel is passed:
   *
   *  - A table view length.
   *  - An input MemRef (converts to 3-5 arguments).
   *  - A result MemRef (another 3-5 arguments).
   */
  using Container = llvm::SmallVector<void*, 11>;

  /**
   * @brief Converts to the raw `void*` pointers required by the CUDA kernel
   * launch function.
   *
   * The conversion takes into account the MLIR calling convention for the
   * concrete type.
   *
   * @param[out] args The generated kernel arguments.
   */
  virtual void toPointers(Container& args) = 0;
};

/**
 * @brief A kernel arguments builder.
 *
 * Dynamically builds a `void*` vector of kernel arguments to be passed to
 * `KernelLauncher::launch()` at runtime. Building arguments at runtime enables
 * the builder to be used for JIT-compiled kernels.
 */
class KernelArgsBuilder {
 public:
  using Container = llvm::SmallVector<std::unique_ptr<KernelArg>, 0>;

  /**
   * @brief Builds the kernel arguments.
   */
  [[nodiscard]] KernelArg::Container build();

  /**
   * @brief Appends an MLIR IndexType.
   *
   * @param value The IndexType.
   */
  KernelArgsBuilder& append(gqe::compiler::mlirIndexType value);

  /**
   * @brief Appends an RMM device scalar of type int32_t.
   *
   * @param value The device scalar.
   */
  KernelArgsBuilder& append(rmm::device_scalar<int32_t>& value);

  /**
   * @brief Appends an RMM device scalar of type int64_t.
   *
   * @param value The device scalar.
   */
  KernelArgsBuilder& append(rmm::device_scalar<int64_t>& value);

  /**
   * @brief Appends an RMM device scalar of type float.
   *
   * @param value The device scalar.
   */
  KernelArgsBuilder& append(rmm::device_scalar<float>& value);

  /**
   * @brief Appends an RMM device scalar of type double.
   *
   * @param value The device scalar.
   */
  KernelArgsBuilder& append(rmm::device_scalar<double>& value);

  /**
   * @brief Appends a cuDF mutable column view.
   *
   * @param column The mutable column view.
   */
  KernelArgsBuilder& append(cudf::mutable_column_view column);

 private:
  Container args;
};

}  // namespace utility
}  // namespace gqe
