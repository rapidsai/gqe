// Copyright 2026 NVIDIA Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Conversion between Arrow schemas/types and Substrait `NamedStruct`/`Type`.

use arrow::datatypes::{DataType, Field, Schema};
use substrait::proto::r#type::{self, Kind, Nullability};
use substrait::proto::{self, NamedStruct};

/// Convert an Arrow `Schema` to a Substrait `NamedStruct`.
pub fn schema_to_named_struct(schema: &Schema) -> NamedStruct {
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let types: Vec<proto::Type> = schema
        .fields()
        .iter()
        .map(|f| arrow_to_substrait_type(f.data_type(), f.is_nullable()))
        .collect();

    NamedStruct {
        names,
        r#struct: Some(r#type::Struct {
            types,
            type_variation_reference: 0,
            nullability: Nullability::Required as i32,
        }),
    }
}

/// Convert a Substrait `NamedStruct` to an Arrow `Schema`.
pub fn named_struct_to_schema(ns: &NamedStruct) -> Result<Schema, String> {
    let st = ns.r#struct.as_ref().ok_or("NamedStruct has no struct")?;

    if ns.names.len() != st.types.len() {
        return Err(format!(
            "NamedStruct name count ({}) != type count ({})",
            ns.names.len(),
            st.types.len()
        ));
    }

    let mut fields = vec![];
    for (name, ty) in ns.names.iter().zip(st.types.iter()) {
        let (dt, nullable) = substrait_type_to_arrow(ty)?;
        fields.push(Field::new(name, dt, nullable));
    }
    Ok(Schema::new(fields))
}

/// Convert an Arrow `DataType` + nullable flag to a Substrait `Type`.
fn arrow_to_substrait_type(dt: &DataType, nullable: bool) -> proto::Type {
    let nullability = if nullable {
        Nullability::Nullable as i32
    } else {
        Nullability::Required as i32
    };

    let kind = match dt {
        DataType::Boolean => Kind::Bool(r#type::Boolean {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Int8 => Kind::I8(r#type::I8 {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Int16 => Kind::I16(r#type::I16 {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Int32 => Kind::I32(r#type::I32 {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Int64 => Kind::I64(r#type::I64 {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Float32 => Kind::Fp32(r#type::Fp32 {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Float64 => Kind::Fp64(r#type::Fp64 {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Kind::String(r#type::String {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Date32 => Kind::Date(r#type::Date {
            type_variation_reference: 0,
            nullability,
        }),
        DataType::Decimal128(precision, scale) => Kind::Decimal(r#type::Decimal {
            precision: *precision as i32,
            scale: *scale as i32,
            type_variation_reference: 0,
            nullability,
        }),
        other => panic!("Unsupported Arrow type for Substrait encoding: {other:?}"),
    };

    proto::Type { kind: Some(kind) }
}

/// Convert a Substrait `Type` to an Arrow `DataType` and nullable flag.
fn substrait_type_to_arrow(ty: &proto::Type) -> Result<(DataType, bool), String> {
    let kind = ty.kind.as_ref().ok_or("Substrait type has no kind")?;

    let (dt, nullability_i32) = match kind {
        Kind::Bool(t) => (DataType::Boolean, t.nullability),
        Kind::I8(t) => (DataType::Int8, t.nullability),
        Kind::I16(t) => (DataType::Int16, t.nullability),
        Kind::I32(t) => (DataType::Int32, t.nullability),
        Kind::I64(t) => (DataType::Int64, t.nullability),
        Kind::Fp32(t) => (DataType::Float32, t.nullability),
        Kind::Fp64(t) => (DataType::Float64, t.nullability),
        Kind::String(t) => (DataType::Utf8, t.nullability),
        Kind::Varchar(t) => (DataType::Utf8, t.nullability),
        Kind::Date(t) => (DataType::Date32, t.nullability),
        Kind::Decimal(t) => (
            DataType::Decimal128(t.precision as u8, t.scale as i8),
            t.nullability,
        ),
        Kind::Struct(t) => {
            let mut fields = vec![];
            for (i, sub_ty) in t.types.iter().enumerate() {
                let (sub_dt, sub_nullable) = substrait_type_to_arrow(sub_ty)?;
                fields.push(Field::new(format!("field_{i}"), sub_dt, sub_nullable));
            }
            (DataType::Struct(fields.into()), t.nullability)
        }
        other => return Err(format!("Unsupported Substrait type: {other:?}")),
    };

    let nullable = nullability_i32 != Nullability::Required as i32;
    Ok((dt, nullable))
}
