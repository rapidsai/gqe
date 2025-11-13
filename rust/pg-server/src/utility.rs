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

use crate::error::{pg_wire_usererror, PgErrorCode, PgErrorSeverity};
use crate::Result;
use async_walkdir::{Filtering, WalkDir};
use gqe_rs::api::{CompressionFormat, DataType as GqeDataType, IOEngineType};
use gqe_rs::executor::OptimizationParameters;
use pgwire::error::{ErrorInfo, PgWireError};
use sqlparser::ast::{self, DataType as DfDataType};
use std::ffi::OsString;
use tokio_stream::StreamExt;

pub(crate) fn convert_data_type_df_to_gqe(df_type: &DfDataType) -> Result<GqeDataType> {
    match df_type {
        DfDataType::Int(ref _display_width) => Ok(GqeDataType::Int32),
        DfDataType::Integer(ref _display_width) => Ok(GqeDataType::Int32),
        DfDataType::BigInt(ref _display_width) => Ok(GqeDataType::Int64),
        DfDataType::Real => Ok(GqeDataType::Float32),
        DfDataType::Double => Ok(GqeDataType::Float64),
        DfDataType::DoublePrecision => Ok(GqeDataType::Float64),
        DfDataType::Decimal(ref _number_info) => Ok(GqeDataType::Float64),
        DfDataType::Numeric(ref _number_info) => Ok(GqeDataType::Float64),
        DfDataType::Date => Ok(GqeDataType::TimestampDays),
        DfDataType::String(_) => Ok(GqeDataType::String),
        _ => Err(PgWireError::UserError(Box::new(ErrorInfo::new(
            "ERROR".to_owned(),
            PgErrorCode::InvalidColumnDefinition.as_code().to_owned(),
            format!("Data type {} is not implemented.", df_type),
        )))),
    }
}

pub(crate) fn expr_value_to_str(expr: &ast::Expr) -> Result<&str> {
    match expr {
        ast::Expr::Value(ast::Value::SingleQuotedString(ref s))
        | ast::Expr::Value(ast::Value::DoubleQuotedString(ref s)) => Ok(s.as_str()),
        ast::Expr::Identifier(_) => Err(pg_wire_usererror(
            PgErrorSeverity::Error,
            PgErrorCode::InvalidParameterValue,
            format!(
                "Parameter {} is an identifier, not a string. Hint: SQL strings are single-quoted.",
                expr
            )
            .as_str(),
        )),
        _ => Err(pg_wire_usererror(
            PgErrorSeverity::Error,
            PgErrorCode::InvalidParameterValue,
            format!("Parameter \"{}\" is not a string.", expr).as_str(),
        )),
    }
}

pub(crate) async fn get_parquet_files(path: &str) -> Result<Vec<String>> {
    let mut entries = WalkDir::new(path).filter(|entry| async move {
        if entry.path().is_file()
            && entry.path().extension().map(|s| s.to_ascii_lowercase())
                == Some(OsString::from("parquet"))
        {
            Filtering::Continue
        } else {
            Filtering::Ignore
        }
    });

    let mut files = Vec::new();
    loop {
        match entries.next().await {
            Some(Ok(entry)) => {
                let path = entry.path();
                files.push(path.to_string_lossy().to_string());
            }
            Some(Err(e)) => Err(pg_wire_usererror(
                crate::error::PgErrorSeverity::Error,
                PgErrorCode::IoError,
                format!("Failed to access Parquet path \"{}\": {}", path, e).as_str(),
            ))?,
            None => break,
        }
    }

    Ok(files)
}

/// Tries to set a parameter from a value string.
///
/// Returns an error if the parameter does not exist, or cannot be set from
/// the given value.
pub fn try_set_optimization_parameter(
    opms: &mut OptimizationParameters,
    name: &str,
    value: &str,
) -> Result<()> {
    let error = || {
        PgWireError::UserError(Box::new(ErrorInfo::new(
            "ERROR".to_owned(),
            PgErrorCode::UndefinedParameter.as_code().to_owned(),
            format!("Parameter {} has wrong value type.", name),
        )))
    };

    match name {
        "MAX_NUM_WORKERS" => {
            opms.max_num_workers = value.parse().map_err(move |_| error())?;
        }
        "MAX_NUM_PARTITIONS" => {
            opms.max_num_partitions = value.parse().map_err(move |_| error())?;
        }
        "GQE_LOG_LEVEL" => {
            opms.log_level = value.to_string();
        }
        "GQE_JOIN_USE_HASH_MAP_CACHE" => {
            opms.join_use_hash_map_cache = value.parse().map_err(move |_| error())?;
        }
        "GQE_READ_USE_ZERO_COPY" => {
            opms.read_zero_copy_enable = value.parse().map_err(move |_| error())?;
        }
        "GQE_USE_CUSTOMIZED_IO" => {
            opms.use_customized_io = value.parse().map_err(move |_| error())?
        }
        "GQE_IO_BOUNCE_BUFFER_SIZE" => {
            opms.io_bounce_buffer_size = value.parse().map_err(move |_| error())?
        }
        "GQE_IO_AUXILIARY_THREADS" => {
            opms.io_auxiliary_threads = value.parse().map_err(move |_| error())?
        }
        "GQE_IN_MEMORY_TABLE_COMP_FORMAT" => {
            opms.in_memory_table_compression_format = match value.to_lowercase().as_str() {
                "none" => CompressionFormat::None,
                "ans" => CompressionFormat::ANS,
                _ => Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                    "ERROR".to_owned(),
                    PgErrorCode::UndefinedParameter.as_code().to_owned(),
                    format!("Compression format \"{}\" is invalid.", value),
                ))))?,
            }
        }
        "GQE_IO_BLOCK_SIZE" => opms.io_block_size = value.parse().map_err(move |_| error())?,
        "GQE_IO_ENGINE" => {
            opms.io_engine = match value.to_lowercase().as_str() {
                "auto" | "automatic" => IOEngineType::Automatic,
                "io_uring" => IOEngineType::IOUring,
                "psync" => IOEngineType::PSync,
                _ => Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                    "ERROR".to_owned(),
                    PgErrorCode::UndefinedParameter.as_code().to_owned(),
                    format!("I/O engine type \"{}\" is invalid.", value),
                ))))?,
            }
        }
        "GQE_IO_PIPELINING_ENABLE" => {
            opms.io_pipelining = value.parse().map_err(move |_| error())?
        }
        "GQE_IO_ALIGNMENT" => opms.io_alignment = value.parse().map_err(move |_| error())?,
        _ => Err(PgWireError::UserError(Box::new(ErrorInfo::new(
            "ERROR".to_owned(),
            PgErrorCode::UndefinedParameter.as_code().to_owned(),
            format!("Parameter {} has wrong value type.", name),
        ))))?,
    }

    Ok(())
}
