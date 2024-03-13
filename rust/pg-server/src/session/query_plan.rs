/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

use crate::error::{pg_wire_usererror, PgErrorCode, PgErrorSeverity, Result};
use gqe_rs::api::Catalog;
use gqe_rs::logical::LogicalPlan;
use gqe_rs::storage as gqe_storage;
use pgwire::error::{ErrorInfo, PgWireError};

pub(crate) fn new_table_copy_plan(
    catalog: &Catalog,
    src_table_name: &str,
    dst_table_name: &str,
) -> Result<LogicalPlan> {
    let src_column_names = catalog.column_names(src_table_name).map_err(|e| {
        pg_wire_usererror(
            PgErrorSeverity::Error,
            PgErrorCode::UndefinedTable,
            e.to_string().as_str(),
        )
    })?;
    let dst_column_names = catalog.column_names(dst_table_name).map_err(|e| {
        pg_wire_usererror(
            PgErrorSeverity::Error,
            PgErrorCode::UndefinedTable,
            e.to_string().as_str(),
        )
    })?;

    if src_column_names.len() != dst_column_names.len() {
        return Err(pg_wire_usererror(
            PgErrorSeverity::Error,
            PgErrorCode::InvalidParameterValue,
            "Failed to copy data to in-memory table because column names do not match.",
        ));
    }

    let src_column_types = src_column_names
        .iter()
        .map(|c| {
            catalog.column_type(src_table_name, c).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::UndefinedTable,
                    e.to_string().as_str(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let dst_column_types = dst_column_names
        .iter()
        .map(|c| {
            catalog.column_type(dst_table_name, c).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::UndefinedTable,
                    e.to_string().as_str(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    src_column_types
        .iter()
        .zip(dst_column_types.iter())
        .try_for_each(|(&src, &dst)| {
            if src == dst {
                Ok(())
            } else {
                Err(pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::DataTypeMismatch,
                    format!("Data types mismatch: expected {:?}, got {:?}.", src, dst).as_str(),
                ))
            }
        })?;

    let read_table = gqe_storage::new_read_relation(
        &[],
        src_column_names.as_slice(),
        src_column_types.as_slice(),
        src_table_name,
    )
    .map_err(|e| {
        PgWireError::UserError(Box::new(ErrorInfo::new(
            "ERROR".to_owned(),
            PgErrorCode::InternalError.as_code().to_owned(),
            format!("Failed to read relation: {}", e),
        )))
    })?;
    let copy_plan = gqe_storage::new_write_relation(
        read_table,
        dst_column_names.as_slice(),
        dst_column_types.as_slice(),
        dst_table_name,
    )
    .map_err(|e| {
        PgWireError::UserError(Box::new(ErrorInfo::new(
            "ERROR".to_owned(),
            PgErrorCode::InternalError.as_code().to_owned(),
            format!("Failed to write relation: {}", e),
        )))
    })?;

    Ok(copy_plan)
}
