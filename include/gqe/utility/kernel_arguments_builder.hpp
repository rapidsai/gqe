/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <gqe/compiler/Types.hpp>

#include <cudf/column/column_view.hpp>
#include <llvm/ADT/SmallVector.h>
#include <rmm/device_scalar.hpp>

#include <cassert>
#include <cstdint>
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
