//===- RelAlgOps.td - Relational Algebra dialect ops -*- tablegen -*-------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
//===----------------------------------------------------------------------===//

#ifndef RELALG_OPS
#define RELALG_OPS

include "RelAlgAttrs.td"
include "RelAlgDialect.td"
include "RelAlgTraits.td"
include "RelAlgTypes.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def RelAlg_AggregateOp : RelAlg_Op<"aggregate", [Pure, TupleStreamConsumer]> {
  let summary = "Aggregates values by key.";
  let description = [{
    The `relalg.aggregate` operation computes aggregates of its inputs grouped
    by one or more keys.

    The AggregateOp is a pipeline breaker.

    An AggregateOp takes an input tuple stream, an array of group keys, and an
    array of aggregate function arguments. It returns a tuple stream.
  }];

  let arguments = (ins RelAlg_TupleStream:$input, RelAlg_IURefArray:$group_keys,
      RelAlg_AggregateFnArgArray:$fn_args);
  let results = (outs RelAlg_TupleStream:$result);
}

// TODO: Add an MLIR transform that specializes AggregateOp.
def RelAlg_ScalarAggregateOp
    : RelAlg_Op<"scalar_aggregate", [Pure, TupleStreamConsumer]> {
  let summary = "Aggregates each column to a scalar value.";
  let description = [{
    The `relalg.scalar_aggregate` operation is a specialization of
    `relalg.aggregate`. It aggregates each of its input columns to a scalar
    result value. It does not group by key.

    The ScalarAggregateOp is a pipeline breaker.

    A ScalarAggregateOp takes an input tuple stream, and array of aggregate
    function arguments.

    Precondition: The aggregation must be guaranteed to be scalar.
  }];

  let arguments = (ins RelAlg_TupleStream:$input,
      RelAlg_AggregateFnArgArray:$fn_args);
  let results = (outs RelAlg_TupleStream:$result);
}

def RelAlg_CudfTableScanOp : RelAlg_Op<"cudf_table_scan", [Pure]> {
  let summary = "Scans a cuDF table.";
  let description = [{
        The `relalg.cudf_table_scan` scans a cuDF table. The table can be in GPU
        memory or CPU memory, as long as its accessible by the GPU.

        A CudfTableScanOp takes a table name and an array of column definitions.
        It returns a tuple stream.
    }];

  let arguments = (ins StrAttr:$table_name,
      RelAlg_ColumnDefArray:$load_columns);
  let results = (outs RelAlg_TupleStream:$result);
}

def RelAlg_FilterOp
    : RelAlg_Op<"filter", [Pure, Idempotent, SameOperandsAndResultType,
                           SameOperandsAndResultShape,
                           SingleBlockImplicitTerminator<"YieldOp">,
                           TupleStreamConsumer]> {
  let summary = "Filters rows according to a Boolean predicate.";
  let description = [{
    The `relalg.filter` operation takes a Boolean predicate and outputs only the
    rows that satisfy this predicate.
  }];

  let arguments = (ins RelAlg_TupleStream:$input);
  let results = (outs RelAlg_TupleStream:$result);

  let regions = (region SizedRegion<1>:$predicate);
}

def RelAlg_MapOp
    : RelAlg_Op<"map", [Pure, SingleBlockImplicitTerminator<"YieldOp">,
                        TupleStreamConsumer]> {
  let summary = "Computes a scalar expression on a row.";
  let description = [{
    The `relalg.map` operation takes an expression and applies it to each input
    row. It outputs the expression result for each row.

    Note that the map expression must yield at least one MLIR value.
  }];

  let arguments = (ins RelAlg_TupleStream:$input);
  let results = (outs RelAlg_TupleStream:$result);

  let regions = (region SizedRegion<1>:$expression);

  let hasRegionVerifier = 1;
}

def RelAlg_YieldOp : RelAlg_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "Expression yield and termination operation";
  let description = [{
    The `relalg.yield` operation yields a value from an expression and
  

    The `yield` operation is modeled after the `scf.yield` operation. The
    keyword is chosen over `return` because an expression is, unlike a function,
    conceptually not a control flow construct.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{  attr-dict ($results^ `:` type($results))? }];
}

def RelAlg_GetIUOp : RelAlg_Op<"getIU", []> {
  let summary = "Dereferences the IU reference and returns its value.";
  let description = [{
    The `relalg.getIU` operation takes a tuple stream and an IU reference, and
    returns an MLIR attribute of that IU.

    An IU abstractly represents a concrete value.
  }];

  let arguments = (ins RelAlg_IURef:$iu_ref);
  let results = (outs AnyType:$result);
}

def RelAlg_ForwardTupleStreamOp
    : RelAlg_Op<"forwardts", [/* TODO: add side effect traits */]> {
  let summary = "Create a new tuple stream";

  let description = [{
    The `relalg.forwardts` operation creates a new tuple stream. The op marks
    the insert location for the parent operator.

    Restriction: The tuple stream can only be forwarded once per operator.
  }];

  let arguments = (ins I1:$is_thread_active);
}

#endif // RELALG_OPS