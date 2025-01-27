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
use crate::session::{query_plan, GqeSession};
use crate::utility;
use gqe_rs::api::{Catalog, ColumnSchema as GqeColumnSchema, PartitioningSchemaKind, StorageKind};
use gqe_rs::executor::{self as gqe_executor, TaskGraphBuilder};
use gqe_rs::logical::SubstraitParser;
use gqe_rs::physical::PhysicalPlanBuilder;
use gqe_rs::task_manager_context::TaskManagerContext;
use gqe_rs::query_context::QueryContext;
use log::trace;
use std::ops::DerefMut;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::task;

pub(super) async fn create_table_from_parquet(
    catalog: &mut Catalog,
    table_name: &str,
    columns: &[GqeColumnSchema],
    dataset_location: &str,
) -> Result<()> {
    let file_paths = utility::get_parquet_files(dataset_location).await?;

    catalog
        .register_table(
            table_name,
            columns,
            &StorageKind::ParquetFile {
                file_paths: file_paths.clone(),
            },
            &PartitioningSchemaKind::Automatic,
        )
        .map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to register table \"{}\": {}", table_name, e).as_str(),
            )
        })?;

    trace!(
        "name: {:?}, columns: {:?}, path: {:?}",
        table_name,
        columns,
        file_paths
    );

    Ok(())
}

pub(super) async fn create_temporary_duplicate_table(
    catalog: &mut Catalog,
    dst_table_name: &str,
    src_table_name: &str,
    storage_kind: &StorageKind,
    partitioning_schema_kind: &PartitioningSchemaKind,
) -> Result<()> {
    let columns = catalog.column_names(src_table_name).map_err(|e| {
        pg_wire_usererror(
            PgErrorSeverity::Error,
            PgErrorCode::UndefinedTable,
            e.to_string().as_str(),
        )
    })?;
    let column_schemas = columns
        .iter()
        .map(|column| {
            Ok(GqeColumnSchema {
                column_name: column.to_owned(),
                data_type: catalog.column_type(src_table_name, column).map_err(|e| {
                    pg_wire_usererror(
                        PgErrorSeverity::Error,
                        PgErrorCode::UndefinedTable,
                        e.to_string().as_str(),
                    )
                })?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    catalog
        .register_table(
            dst_table_name,
            column_schemas.as_slice(),
            storage_kind,
            partitioning_schema_kind,
        )
        .map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to register table \"{}\": {}", dst_table_name, e).as_str(),
            )
        })?;

    trace!(
        "name: {:?}, columns: {:?}, storage_kind: {:?}",
        dst_table_name,
        column_schemas,
        storage_kind
    );

    Ok(())
}

pub(super) async fn execute_prepared_statement(
    session_arc: Arc<Mutex<GqeSession>>,
    substrait_query_file: &str,
) -> Result<f32> {
    let substrait_query_file = substrait_query_file.to_owned();

    task::spawn_blocking(move || -> Result<f32> {
        // Get the MutexGuard. We need to keep this instance alive until
        // the session is no longer needed.
        let mut session = session_arc.blocking_lock();

        // We need to split-borrow the session struct's fields. However,
        // we cannot take two (mutable) references of the MutexGuard.
        // Thus, borrow the actual session, and then perform the field
        // accesses. We can hide the MutexGuard variable to clean things
        // up a bit.
        let session = session.deref_mut();

        let mut task_manager_ctx = TaskManagerContext::new().map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to construct a db context: {}", e).as_str(),
            )
        })?;

        let mut query_ctx = QueryContext::new(&session.parameters).map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to construct a query context: {}", e).as_str(),
            )
        })?;

        let logical_plan = {
            let mut parser = SubstraitParser::new(&mut session.catalog).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::InternalError,
                    format!("Failed to construct a Substrait parser: {}", e).as_str(),
                )
            })?;

            parser
                .from_file(substrait_query_file.as_str())
                .map_err(|e| {
                    pg_wire_usererror(
                        PgErrorSeverity::Error,
                        PgErrorCode::InternalError,
                        format!("Failed to parse the Substrait file: {}", e).as_str(),
                    )
                })?
        };

        let logical_plan = logical_plan.first().ok_or_else(|| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InvalidPreparedStatementDefinition,
                "Substrait file must contain exactly one SQL statement.",
            )
        })?;

        let mut physical_plan = {
            let mut physical_plan_builder = PhysicalPlanBuilder::new(&mut session.catalog)
                .map_err(|e| {
                    pg_wire_usererror(
                        PgErrorSeverity::Error,
                        PgErrorCode::InternalError,
                        format!("Failed to construct a physical plan builder: {}", e).as_str(),
                    )
                })?;

            physical_plan_builder.build(logical_plan).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::InternalError,
                    format!("Failed to build the physical plan: {}", e).as_str(),
                )
            })?
        };

        let task_graph = {
            let mut task_graph_builder = TaskGraphBuilder::new(&mut task_manager_ctx, &mut query_ctx, &mut session.catalog)
                .map_err(|e| {
                    pg_wire_usererror(
                        PgErrorSeverity::Error,
                        PgErrorCode::InternalError,
                        format!("Failed to construct a task graph builder: {}", e).as_str(),
                    )
                })?;

            task_graph_builder.build(&mut physical_plan).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::InternalError,
                    format!("Failed to build the task graph: {}", e).as_str(),
                )
            })?
        };

        let timer = Instant::now();
        gqe_executor::execute_task_graph_single_gpu(&mut task_manager_ctx, &mut query_ctx, &task_graph).map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to execute the task graph: {}", e).as_str(),
            )
        })?;
        let duration = timer.elapsed().as_secs_f32();

        Ok(duration)
    })
    .await
    .map_err(|e| {
        pg_wire_usererror(
            PgErrorSeverity::Fatal,
            PgErrorCode::InternalError,
            format!("Tokio task failed with: {}", e).as_str(),
        )
    })?
}

pub(super) async fn insert_from_table(
    session_arc: Arc<Mutex<GqeSession>>,
    dst_table_name: &str,
    src_table_name: &str,
) -> Result<f32> {
    let src_table_name = src_table_name.to_owned();
    let dst_table_name = dst_table_name.to_owned();

    task::spawn_blocking(move || -> Result<f32> {
        // Get the MutexGuard. We need to keep this instance alive until
        // the session is no longer needed.
        let mut session = session_arc.blocking_lock();

        // We need to split-borrow the session struct's fields. However,
        // we cannot take two (mutable) references of the MutexGuard.
        // Thus, borrow the actual session, and then perform the field
        // accesses. We can hide the MutexGuard variable to clean things
        // up a bit.
        let session = session.deref_mut();

        let logical_plan = query_plan::new_table_copy_plan(
            &session.catalog,
            src_table_name.as_str(),
            dst_table_name.as_str(),
        )?;
        
        let mut task_manager_ctx = TaskManagerContext::new().map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to construct a db context: {}", e).as_str(),
            )
        })?;

        let mut query_ctx = QueryContext::new(&session.parameters).map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to construct a query context: {}", e).as_str(),
            )
        })?;

        let mut physical_plan = {
            let mut physical_plan_builder = PhysicalPlanBuilder::new(&mut session.catalog)
                .map_err(|e| {
                    pg_wire_usererror(
                        PgErrorSeverity::Error,
                        PgErrorCode::InternalError,
                        format!("Failed to construct a physical plan builder: {}", e).as_str(),
                    )
                })?;

            physical_plan_builder.build(&logical_plan).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::InternalError,
                    format!("Failed to build the physical plan: {}", e).as_str(),
                )
            })?
        };

        let task_graph = {
            let mut task_graph_builder = TaskGraphBuilder::new(&mut task_manager_ctx, &mut query_ctx, &mut session.catalog)
                .map_err(|e| {
                    pg_wire_usererror(
                        PgErrorSeverity::Error,
                        PgErrorCode::InternalError,
                        format!("Failed to construct a task graph builder: {}", e).as_str(),
                    )
                })?;

            task_graph_builder.build(&mut physical_plan).map_err(|e| {
                pg_wire_usererror(
                    PgErrorSeverity::Error,
                    PgErrorCode::InternalError,
                    format!("Failed to build the task graph: {}", e).as_str(),
                )
            })?
        };

        let timer = Instant::now();
        gqe_executor::execute_task_graph_single_gpu(&mut task_manager_ctx, &mut query_ctx, &task_graph).map_err(|e| {
            pg_wire_usererror(
                PgErrorSeverity::Error,
                PgErrorCode::InternalError,
                format!("Failed to execute the task graph: {}", e).as_str(),
            )
        })?;
        let duration = timer.elapsed().as_secs_f32();

        Ok(duration)
    })
    .await
    .map_err(|e| {
        pg_wire_usererror(
            PgErrorSeverity::Fatal,
            PgErrorCode::InternalError,
            format!("Tokio task failed with: {}", e).as_str(),
        )
    })?
}
