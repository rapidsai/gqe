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

#include <gqe/utility/kernel_arguments_builder.hpp>

#include <gqe/compiler/Types.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <llvm/ADT/SmallVector.h>
#include <rmm/device_scalar.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace gqe {
namespace utility {

namespace {

/**
 * @brief Scalar literal calling convention helper.
 *
 * References:
 * https://mlir.llvm.org/docs/TargetLLVMIR/#default-calling-convention-for-ranked-memref
 */
template <typename T>
class ScalarLiteralArg : public KernelArg {
 public:
  explicit ScalarLiteralArg(T value) : value(value) {}

  void toPointers(Container& args) override { args.push_back(reinterpret_cast<void*>(&value)); }

 protected:
  T value;
};

/**
 * @brief Scalar MemRef calling convention helper.
 *
 * References:
 * https://mlir.llvm.org/docs/TargetLLVMIR/#default-calling-convention-for-ranked-memref
 */
template <typename T>
class ScalarMemrefArg : public KernelArg {
 public:
  explicit ScalarMemrefArg(T* allocated, T* aligned, gqe::compiler::mlirIndexType offset)
    : allocated(allocated), aligned(aligned), offset(offset)
  {
  }

  void toPointers(Container& args) override
  {
    args.push_back(reinterpret_cast<void*>(&allocated));
    args.push_back(reinterpret_cast<void*>(&aligned));
    args.push_back(reinterpret_cast<void*>(&offset));
  }

 protected:
  T* allocated;
  T* aligned;
  gqe::compiler::mlirIndexType offset;
};

template <typename T>
class rmmScalarMemrefArg : public ScalarMemrefArg<T> {
 public:
  explicit rmmScalarMemrefArg(rmm::device_scalar<T>& scalar)
    : ScalarMemrefArg<T>(scalar.data(), scalar.data(), 0)
  {
  }
};

/**
 * @brief Dynamically-sized MemRef calling convention helper
 *
 * The associated MemRef kernel argument must reference a 1-dimensional array.
 *
 * References:
 * https://mlir.llvm.org/docs/TargetLLVMIR/#default-calling-convention-for-ranked-memref
 */
template <typename T>
class DynamicMemrefArg : public KernelArg {
 public:
  explicit DynamicMemrefArg(T* allocated,
                            T* aligned,
                            gqe::compiler::mlirIndexType offset,
                            gqe::compiler::mlirIndexType size,
                            gqe::compiler::mlirIndexType stride)
    : allocated(allocated), aligned(aligned), offset(offset), size(size), stride(stride)
  {
  }

  void toPointers(Container& args) override
  {
    args.push_back(reinterpret_cast<void*>(&allocated));
    args.push_back(reinterpret_cast<void*>(&aligned));
    args.push_back(reinterpret_cast<void*>(&offset));
    args.push_back(reinterpret_cast<void*>(&size));
    args.push_back(reinterpret_cast<void*>(&stride));
  }

 protected:
  T* allocated;
  T* aligned;
  gqe::compiler::mlirIndexType offset;
  gqe::compiler::mlirIndexType size;
  gqe::compiler::mlirIndexType stride;
};

/**
 * @brief cuDF column calling convention helper.
 *
 * The cuDF column is converted to a dynamically-sized MemRef.
 */
template <typename T>
class CudfColumnViewMemrefArg : public DynamicMemrefArg<T> {
 public:
  explicit CudfColumnViewMemrefArg(cudf::mutable_column_view column)
    : DynamicMemrefArg<T>(column.head<T>(), column.data<T>(), column.offset(), column.size(), 1)
  {
  }
};

/**
 * @brief Dispatch functor that converts a cuDF mutable column view into raw
 * kernel arguments.
 *
 * Note that not all cuDF types are implemented. The dispatcher validates that
 * the column view contains a supported type.
 */
struct CudfArgFunctor {
  using Container = KernelArgsBuilder::Container;

  /**
   * @brief The base-level dispatcher functor for a cuDF mutable column view.
   *
   * @param column The mutable column view to dispatch.
   * @param[out] args The generated kernel arguments.
   *
   * @throws std::logic_error Throws if called on an unsupported cuDF type.
   */
  template <typename T>
  void operator()(cudf::mutable_column_view column, Container& args) const
  {
    throw std::logic_error("expected a supported cuDF kernel argument type");
  }
};

template <>
void CudfArgFunctor::operator()<int32_t>(cudf::mutable_column_view column, Container& args) const
{
  auto arg = std::make_unique<CudfColumnViewMemrefArg<int32_t>>(column);
  args.push_back(std::move(arg));
}

template <>
void CudfArgFunctor::operator()<float>(cudf::mutable_column_view column, Container& args) const
{
  auto arg = std::make_unique<CudfColumnViewMemrefArg<float>>(column);
  args.push_back(std::move(arg));
}

template <>
void CudfArgFunctor::operator()<double>(cudf::mutable_column_view column, Container& args) const
{
  auto arg = std::make_unique<CudfColumnViewMemrefArg<double>>(column);
  args.push_back(std::move(arg));
}

template <>
void CudfArgFunctor::operator()<cudf::timestamp_D>(cudf::mutable_column_view column,
                                                   Container& args) const
{
  auto arg = std::make_unique<CudfColumnViewMemrefArg<cudf::timestamp_D>>(column);
  args.push_back(std::move(arg));
}

}  // namespace

KernelArg::Container KernelArgsBuilder::build()
{
  KernelArg::Container rawArgs;

  for (auto& arg : args) {
    arg->toPointers(rawArgs);
  }

  return rawArgs;
}

KernelArgsBuilder& KernelArgsBuilder::append(gqe::compiler::mlirIndexType value)
{
  auto arg = std::make_unique<ScalarLiteralArg<gqe::compiler::mlirIndexType>>(value);
  args.push_back(std::move(arg));
  return *this;
}

KernelArgsBuilder& KernelArgsBuilder::append(rmm::device_scalar<int32_t>& value)
{
  auto arg = std::make_unique<rmmScalarMemrefArg<int32_t>>(value);
  args.push_back(std::move(arg));
  return *this;
}

KernelArgsBuilder& KernelArgsBuilder::append(rmm::device_scalar<int64_t>& value)
{
  auto arg = std::make_unique<rmmScalarMemrefArg<int64_t>>(value);
  args.push_back(std::move(arg));
  return *this;
}

KernelArgsBuilder& KernelArgsBuilder::append(rmm::device_scalar<float>& value)
{
  auto arg = std::make_unique<rmmScalarMemrefArg<float>>(value);
  args.push_back(std::move(arg));
  return *this;
}

KernelArgsBuilder& KernelArgsBuilder::append(rmm::device_scalar<double>& value)
{
  auto arg = std::make_unique<rmmScalarMemrefArg<double>>(value);
  args.push_back(std::move(arg));
  return *this;
}

KernelArgsBuilder& KernelArgsBuilder::append(cudf::mutable_column_view column)
{
  cudf::type_dispatcher(column.type(), CudfArgFunctor{}, column, args);
  return *this;
}

}  // namespace utility
}  // namespace gqe
