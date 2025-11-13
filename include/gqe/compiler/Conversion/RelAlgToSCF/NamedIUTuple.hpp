/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>
#include <optional>

namespace gqe::compiler::relalg {

/**
 * @brief An IU with an optional name.
 */
struct NamedIU {
  mlir::Value iu;
  std::optional<mlir::StringAttr> name;
};

/**
 * @brief A named IU tuple for RelAlgToSCF conversion.
 *
 * The named IU tuple maps the index of a RelAlg op's operand to an IU. Example
 * IUs are a value at a column offset, a value in a hash map entry, or a scalar
 * constant value.
 *
 * The named IU tuple implements the schema of a tuple stream. Thus, each tuple
 * is associated with a tuple stream.
 *
 * The IU can optionally have a name, which is useful for printing the query
 * plan and for debugging.
 */
class NamedIUTuple {
 public:
  /**
   * @brief Create a new named IU tuple.
   *
   * @invariant iuTupleStream.getType() == relalg::TupleStreamType
   *
   * @param iuTupleStream The tuple stream that the IU belongs to.
   */
  NamedIUTuple(mlir::Value iuTupleStream);

  /**
   * @brief Insert a new IU without name at the back of the tuple.
   *
   * @param iu The IU to push back.
   */
  void pushBack(mlir::Value iu);

  /**
   * @brief Insert a new named IU at the back of the tuple.
   *
   * @param iu The IU to push back.
   * @param name The name of the IU.
   */
  void pushBack(mlir::Value iu, mlir::StringAttr name);

  /**
   * @brief Append the named IUs from another named IU tuple.
   *
   * @param other The named IU tuple to append.
   */
  void append(NamedIUTuple const& other);

  /**
   * @brief Lookup an IU by an index attribute.
   *
   * @param index The index to lookup.
   *
   * @return The named IU.
   */
  [[nodiscard]] NamedIU lookup(mlir::IntegerAttr index) const;

  /**
   * @brief Lookup an IU by its index.
   *
   * @param index The index to lookup.
   *
   * @return The named IU.
   */
  [[nodiscard]] NamedIU lookup(int32_t index) const;

  /**
   *  @brief Return the associated tuple stream.
   */
  [[nodiscard]] mlir::Value getTupleStream() const;

 private:
  mlir::Value iuTupleStream;
  mlir::SmallVector<NamedIU> tuple;
};

/**
 * A collection of named IU tuples.
 *
 * The collection facilitates finding the named IU tuple of an op, as well as
 * functionality that touches multiple IU tuples.
 */
class NamedIUTupleCollection {
 public:
  /**
   * @brief Lookup, or create, a named IU tuple for a tuple stream.
   *
   * A new named IU tuple is created if the tuple stream does not yet have one.
   */
  [[nodiscard]] NamedIUTuple& getNamedIUTuple(mlir::Value tupleStream);

  /**
   * @brief Append the named IU tuple of a tuple stream to another tuple
   * stream.
   *
   * @param srcTupleStream The source tuple stream.
   * @param dstTupleStream The destination tuple stream.
   */
  void append(mlir::Value srcTupleStream, mlir::Value dstTupleStream);

 private:
  mlir::DenseMap<mlir::Value,
                 NamedIUTuple>
    namedIUTuples;  ///< Map a tuple stream to its
                    ///< named IU tuple.
};

}  // namespace gqe::compiler::relalg
