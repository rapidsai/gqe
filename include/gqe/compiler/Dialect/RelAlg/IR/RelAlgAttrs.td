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
//===- RelAlgAttrs.td - Relational Algebra attributes -*- tablegen -*------===//

#ifndef RELALG_ATTRIBUTES
#define RELALG_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/EnumAttr.td"
include "RelAlgDialect.td"

//===----------------------------------------------------------------------===//
// RelAlg attribute definitions
//===----------------------------------------------------------------------===//

class RelAlg_Attr<string name, string attrMnemonic>
    : AttrDef<RelAlg_Dialect, name, []> {
  let mnemonic = attrMnemonic;
}

def RelAlg_AggregateFunction
    : I32EnumAttr<"AggregateFunction",
                  "Relational algebra aggregate functions.",
                  [I32EnumAttrCase<"Count", 0, "count">,
                   I32EnumAttrCase<"Sum", 1, "sum">,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::gqe::compiler::relalg";
}

def RelAlg_AggregateFunctionAttr
    : EnumAttr<RelAlg_Dialect, RelAlg_AggregateFunction, "aggregate_function"> {

  let summary = "A relational algebra aggregate function.";

  let description = [{
    The attribute defines the supported aggregate functions, e.g., `count` and
    `sum`. These can be passed to `relalg.aggregate` as part of a
    `aggregate_fn_arg`.
  }];

  let assemblyFormat = "`<` $value `>`";
}

def RelAlg_ColumnDef : RelAlg_Attr<"ColumnDef", "column_def"> {
  let summary = "A table column definition.";

  let description = [{
    The ColumnDef references the column of a table, as per the database catalog.
    The column definition declares an IU that can be referenced from other ops.
  }];

  let parameters = (ins StringRefParameter<>:$column_name,
      "mlir::Type":$column_type);

  let assemblyFormat = "`(` custom<ColumnDef>($column_name, $column_type) `)`";

  let extraClassDefinition = [{
      mlir::ParseResult parseColumnDef(mlir::AsmParser &parser, std::string& columnName, mlir::Type& columnType) {
        mlir::StringAttr columnNameAttr;
        if (parser.parseSymbolName(columnNameAttr).failed()) {
          return mlir::LogicalResult::failure();
        }

        columnName = columnNameAttr.str();

        if (parser.parseColonType(columnType).failed()) {
          return mlir::LogicalResult::failure();
        }

        return mlir::LogicalResult::success();
      }

      void printColumnDef(mlir::AsmPrinter &printer, mlir::StringRef columnName, mlir::Type columnType) {
        printer.printSymbolName(columnName);
        printer << " : ";
        printer.printType(columnType);
      }
    }];
}

def RelAlg_IURef : RelAlg_Attr<"IURef", "iu_ref"> {
  let summary = "An information unit reference.";

  let description = [{
        This attribute is a reference to an information unit. An information
        unit (IU) is defined as "a single IU corresponds to a variable that can
        be bound to a value" by Guido Moerkotte in his book "Building Query
        Compilers".

        Moerkotte further states examples of IUs "are attributes of a relation
        or, as we will see, intermediate results".

        Therefore, an IU can originate from a table column, but not necessarily.
        An IU could also originate from, e.g., a constant value declared via a
        `SELECT 0;` SQL statement.

        The IURef takes an operand index and an optional IU name. The index
        refers to the operand's position in the (table or intermediate result)
        schema.
    }];

  let parameters = (ins "int32_t":$index,
      OptionalParameter<"mlir::StringAttr">:$iu_name);

  let assemblyFormat = "`(` $index (`,` $iu_name^)? `)`";
}

def RelAlg_AggregateFnArg : RelAlg_Attr<"AggregateFnArg", "aggregate_fn_arg"> {
  let summary = "An aggregate function argument.";

  let description = [{
    The AggregateFnArg combines an AggregateFunction with the IURef it operates
    on.

    The AggregateFnArg takes an IURef and an AggregateFunction.
  }];

  let parameters = (ins RelAlg_IURef:$iu_ref,
      "::gqe::compiler::relalg::AggregateFunction":$agg_fn);

  let assemblyFormat = "`<` $agg_fn `(` $iu_ref `)` `>`";
}

//===----------------------------------------------------------------------===//
// RelAlg attribute array aliases
//===----------------------------------------------------------------------===//

def RelAlg_AggregateFnArgArray
    : TypedArrayAttrBase<RelAlg_AggregateFnArg,
                         "An array of aggregate function argument attributes.">;
def RelAlg_ColumnDefArray
    : TypedArrayAttrBase<RelAlg_ColumnDef,
                         "An array of column definition attributes.">;
def RelAlg_IURefArray
    : TypedArrayAttrBase<
          RelAlg_IURef,
          "An array attribute of information unit reference attributes.">;

#endif // RELALG_ATTRIBUTES
