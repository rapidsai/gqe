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

//! Substrait encoding and decoding for Write relations (COPY FROM).

use arrow::datatypes::Schema;
use prost::Message;
use substrait::proto::read_rel::local_files::FileOrFiles;
use substrait::proto::read_rel::local_files::file_or_files::{
    FileFormat, ParquetReadOptions, PathType,
};
use substrait::proto::read_rel::{self, LocalFiles};
use substrait::proto::write_rel::{OutputMode, WriteOp, WriteType};
use substrait::proto::{
    self, NamedObjectWrite, Plan, PlanRel, ReadRel, Rel, WriteRel, plan_rel, rel,
};

use super::types::schema_to_named_struct;

/// Encode a COPY FROM statement as a serialized Substrait `Plan`.
pub fn encode_copy_from(table_name: &str, file_path: &str, schema: &Schema) -> Vec<u8> {
    let named_struct = schema_to_named_struct(schema);

    let read = ReadRel {
        common: None,
        base_schema: Some(named_struct.clone()),
        filter: None,
        best_effort_filter: None,
        projection: None,
        advanced_extension: None,
        read_type: Some(read_rel::ReadType::LocalFiles(LocalFiles {
            items: vec![FileOrFiles {
                partition_index: 0,
                start: 0,
                length: 0,
                path_type: Some(PathType::UriPath(file_path.to_string())),
                file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
            }],
            advanced_extension: None,
        })),
    };

    let write = WriteRel {
        write_type: Some(WriteType::NamedTable(NamedObjectWrite {
            names: vec![table_name.to_string()],
            advanced_extension: None,
        })),
        table_schema: Some(named_struct),
        op: WriteOp::Insert as i32,
        input: Some(Box::new(Rel {
            rel_type: Some(rel::RelType::Read(Box::new(read))),
        })),
        output: OutputMode::NoOutput as i32,
        common: None,
        advanced_extension: None,
        create_mode: 0,
    };

    let plan = Plan {
        relations: vec![PlanRel {
            rel_type: Some(plan_rel::RelType::Rel(Rel {
                rel_type: Some(rel::RelType::Write(Box::new(write))),
            })),
        }],
        ..Default::default()
    };

    let mut buf = Vec::new();
    plan.encode(&mut buf)
        .expect("Substrait plan encoding failed");
    buf
}

/// Decoded contents of a Substrait `WriteRel` (COPY FROM).
pub struct DecodedCopyFrom {
    pub table_name: String,
    pub file_path: String,
}

/// Decode a Substrait `WriteRel` into its components.
///
/// Currently only supports INSERT with a `ReadRel(LocalFiles)` input
/// (i.e. COPY FROM a local parquet file/directory).
pub fn decode_copy_from(write: &proto::WriteRel) -> Result<DecodedCopyFrom, String> {
    let op = WriteOp::try_from(write.op).map_err(|_| format!("Unknown write op: {}", write.op))?;
    if op != WriteOp::Insert {
        return Err(format!("Write operation not supported: {op:?}"));
    }

    let table_name = match &write.write_type {
        Some(WriteType::NamedTable(named)) => named
            .names
            .last()
            .ok_or("WriteRel NamedTable has no names")?
            .clone(),
        _ => return Err("WriteRel has no named table write type".into()),
    };

    let input = write.input.as_ref().ok_or("WriteRel has no input")?;

    let file_path = match &input.rel_type {
        Some(rel::RelType::Read(read_rel)) => match &read_rel.read_type {
            Some(read_rel::ReadType::LocalFiles(local_files)) => {
                let item = local_files.items.first().ok_or("LocalFiles has no items")?;
                match &item.path_type {
                    Some(PathType::UriPath(p) | PathType::UriFile(p) | PathType::UriFolder(p)) => {
                        p.clone()
                    }
                    other => return Err(format!("Unsupported path type: {other:?}")),
                }
            }
            other => return Err(format!("Expected LocalFiles read type, got: {other:?}")),
        },
        other => {
            return Err(format!(
                "Expected ReadRel input for WriteRel, got: {other:?}"
            ));
        }
    };

    Ok(DecodedCopyFrom {
        table_name,
        file_path,
    })
}
