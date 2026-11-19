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

//! Substrait encoding and decoding for DDL statements (CREATE TABLE, DROP TABLE).

use std::collections::HashMap;

use arrow::datatypes::Schema;
use pbjson_types::Any;
use prost::Message;
use substrait::proto::ddl_rel::{DdlObject, DdlOp, WriteType};
use substrait::proto::extensions::AdvancedExtension;
use substrait::proto::{self, NamedObjectWrite, Plan, PlanRel, Rel, plan_rel, rel};

use super::types::schema_to_named_struct;

/// Type URL for `CreateTableExtension` packed as `google.protobuf.Any`.
/// Must match the proto package + message name in `proto/ddl_extension.proto`.
const CREATE_TABLE_EXTENSION_TYPE_URL: &str =
    "type.googleapis.com/gqe.proto.CreateTableExtension";

/// Mirror of `proto/ddl_extension.proto :: gqe.proto.StorageOptionValue`.
/// Field numbers must stay in sync with the .proto file.
#[derive(Clone, PartialEq, prost::Message)]
pub struct IntList {
    #[prost(int64, repeated, tag = "1")]
    pub values: Vec<i64>,
}

pub mod storage_option_value {
    #[derive(Clone, PartialEq, prost::Oneof)]
    pub enum Value {
        /// Scalar string (e.g. storage_kind, page_kind).
        #[prost(string, tag = "1")]
        StringVal(String),
        /// Scalar integer (e.g. device_id, numa_node_id).
        #[prost(int64, tag = "2")]
        IntVal(i64),
        /// Integer list (e.g. numa_node_set).
        #[prost(message, tag = "3")]
        IntListVal(super::IntList),
    }
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct StorageOptionValue {
    #[prost(oneof = "storage_option_value::Value", tags = "1, 2, 3")]
    pub value: Option<storage_option_value::Value>,
}

/// Mirror of `proto/ddl_extension.proto :: gqe.proto.CreateTableExtension`.
/// Defined inline to avoid a build.rs / prost-build pipeline just for this message.
/// Field numbers must stay in sync with the .proto file.
#[derive(Clone, PartialEq, prost::Message)]
struct UniqueKey {
    #[prost(uint32, repeated, tag = "1")]
    column_indices: Vec<u32>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct CreateTableExtension {
    /// All UNIQUE / PRIMARY KEY constraints. Size-1 inner vec = single-column; size ≥ 2 = composite.
    #[prost(message, repeated, tag = "1")]
    unique_keys: Vec<UniqueKey>,
    /// Typed WITH (...) key-value pairs from the CREATE TABLE statement.
    /// Parsed and validated server-side into a concrete storage_kind variant.
    #[prost(map = "string, message", tag = "2")]
    storage_options: HashMap<String, StorageOptionValue>,
}

/// Encode a CREATE TABLE statement as a serialized Substrait `Plan`.
///
/// `unique_keys` lists all UNIQUE / PRIMARY KEY constraints (0-based column positions).
/// Each inner `Vec` is one key: size 1 = single-column, size ≥ 2 = composite.
/// Pass an empty slice when there are no constraints.
///
/// `storage_options` carries the typed WITH (...) key-value pairs; pass an
/// empty map when no WITH clause was given.
pub fn encode_create_table(
    table_name: &str,
    schema: &Schema,
    unique_keys: &[Vec<u32>],
    storage_options: &HashMap<String, StorageOptionValue>,
) -> Vec<u8> {
    encode_ddl(
        table_name,
        DdlOp::Create,
        Some(schema),
        Some(unique_keys),
        Some(storage_options),
    )
}

/// Encode a CREATE OR REPLACE TABLE statement as a serialized Substrait `Plan`.
pub fn encode_create_or_replace_table(
    table_name: &str,
    schema: &Schema,
    unique_keys: &[Vec<u32>],
    storage_options: &HashMap<String, StorageOptionValue>,
) -> Vec<u8> {
    encode_ddl(
        table_name,
        DdlOp::CreateOrReplace,
        Some(schema),
        Some(unique_keys),
        Some(storage_options),
    )
}

/// Encode a DROP TABLE statement as a serialized Substrait `Plan`.
pub fn encode_drop_table(table_name: &str) -> Vec<u8> {
    encode_ddl(table_name, DdlOp::Drop, None, None, None)
}

/// Encode a DROP TABLE IF EXISTS statement as a serialized Substrait `Plan`.
pub fn encode_drop_table_if_exists(table_name: &str) -> Vec<u8> {
    encode_ddl(table_name, DdlOp::DropIfExist, None, None, None)
}

fn encode_ddl(
    table_name: &str,
    op: DdlOp,
    schema: Option<&Schema>,
    unique_keys: Option<&[Vec<u32>]>,
    storage_options: Option<&HashMap<String, StorageOptionValue>>,
) -> Vec<u8> {
    let mut ext = CreateTableExtension::default();
    if let Some(keys) = unique_keys {
        ext.unique_keys = keys
            .iter()
            .map(|cols| UniqueKey { column_indices: cols.clone() })
            .collect();
    }
    if let Some(opts) = storage_options {
        ext.storage_options = opts.clone();
    }

    let advanced_extension = if !ext.unique_keys.is_empty() || !ext.storage_options.is_empty() {
        let mut encoded = Vec::new();
        ext.encode(&mut encoded)
            .expect("CreateTableExtension encoding failed");
        Some(AdvancedExtension {
            enhancement: Some(Any {
                type_url: CREATE_TABLE_EXTENSION_TYPE_URL.to_string(),
                value: encoded.into(),
            }),
            optimization: vec![],
        })
    } else {
        None
    };

    let ddl = proto::DdlRel {
        write_type: Some(WriteType::NamedObject(NamedObjectWrite {
            names: vec![table_name.to_string()],
            advanced_extension: None,
        })),
        table_schema: schema.map(schema_to_named_struct),
        table_defaults: None,
        object: DdlObject::Table as i32,
        op: op as i32,
        view_definition: None,
        common: None,
        advanced_extension,
    };

    let plan = Plan {
        relations: vec![PlanRel {
            rel_type: Some(plan_rel::RelType::Rel(Rel {
                rel_type: Some(rel::RelType::Ddl(Box::new(ddl))),
            })),
        }],
        ..Default::default()
    };

    let mut buf = Vec::new();
    plan.encode(&mut buf)
        .expect("Substrait plan encoding failed");
    buf
}

/// Decoded contents of a Substrait `DdlRel`.
pub struct DecodedDdl {
    pub table_name: String,
    pub op: DdlOp,
    pub schema: Option<Schema>,
}

/// Decode a Substrait `DdlRel` into its components.
pub fn decode_ddl(ddl: &proto::DdlRel) -> Result<DecodedDdl, String> {
    use super::types::named_struct_to_schema;

    let object = DdlObject::try_from(ddl.object)
        .map_err(|_| format!("Unknown DDL object: {}", ddl.object))?;
    let op = DdlOp::try_from(ddl.op).map_err(|_| format!("Unknown DDL op: {}", ddl.op))?;

    if object != DdlObject::Table {
        return Err(format!("DDL object type not supported: {object:?}"));
    }

    let table_name = match &ddl.write_type {
        Some(WriteType::NamedObject(named)) => named
            .names
            .last()
            .ok_or("DDL NamedObject has no names")?
            .clone(),
        _ => return Err("DDL has no named object write type".into()),
    };

    let schema = ddl
        .table_schema
        .as_ref()
        .map(named_struct_to_schema)
        .transpose()?;

    Ok(DecodedDdl {
        table_name,
        op,
        schema,
    })
}
