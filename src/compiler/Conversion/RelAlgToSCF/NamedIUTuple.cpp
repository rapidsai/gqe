/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/compiler/Conversion/RelAlgToSCF/NamedIUTuple.hpp>

#include <gqe/compiler/Dialect/RelAlg/IR/RelAlgTypes.hpp>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstdint>
#include <optional>
#include <utility>

namespace gqe {
namespace compiler {
namespace relalg {

NamedIUTuple::NamedIUTuple(mlir::Value iuTupleStream) : iuTupleStream(iuTupleStream)
{
  assert(mlir::isa<TupleStreamType>(iuTupleStream.getType()) && "expected a tuple stream value");
}

void NamedIUTuple::pushBack(mlir::Value iu) { tuple.push_back(NamedIU{iu, std::nullopt}); }

void NamedIUTuple::pushBack(mlir::Value iu, mlir::StringAttr name)
{
  tuple.push_back(NamedIU{iu, name});
}

NamedIU NamedIUTuple::lookup(mlir::IntegerAttr index) const
{
  auto numericIndex = index.getInt();
  assert(numericIndex <= static_cast<int32_t>(tuple.size()) && "lookup index is out-of-bounds");
  assert(index.getType().isSignedInteger(32) && "expected the index attribute to be an int32_t");

  auto const& ref = tuple[numericIndex];
  return {ref.iu, ref.name};
}

NamedIU NamedIUTuple::lookup(int32_t index) const
{
  assert(index <= static_cast<int32_t>(tuple.size()) && "lookup index is out-of-bounds");

  auto const& ref = tuple[index];
  return {ref.iu, ref.name};
}

void NamedIUTuple::append(NamedIUTuple const& other) { tuple.append(other.tuple); }

mlir::Value NamedIUTuple::getTupleStream() const { return iuTupleStream; }

NamedIUTuple& NamedIUTupleCollection::getNamedIUTuple(mlir::Value tupleStream)
{
  auto it = namedIUTuples.find(tupleStream);

  if (it == namedIUTuples.end()) {
    return namedIUTuples.insert(std::make_pair(tupleStream, relalg::NamedIUTuple(tupleStream)))
      .first->getSecond();
  } else {
    return it->getSecond();
  }
}

void NamedIUTupleCollection::append(mlir::Value srcTupleStream, mlir::Value dstTupleStream)
{
  assert(mlir::isa<TupleStreamType>(srcTupleStream.getType()));
  assert(mlir::isa<TupleStreamType>(dstTupleStream.getType()));

  auto srcIt = namedIUTuples.find(srcTupleStream);
  auto dstIt = namedIUTuples.find(dstTupleStream);

  assert(srcIt != namedIUTuples.end() && "expected that the source tuple stream exists");
  assert(dstIt != namedIUTuples.end() && "expected that the destination tuple stream exists");

  dstIt->second.append(srcIt->second);
}

}  // namespace relalg
}  // namespace compiler
}  // namespace gqe
