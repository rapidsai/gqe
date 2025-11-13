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

mod query_plan;
mod statement;

use crate::error::{pg_wire_usererror, PgErrorCode, PgErrorSeverity, Result};
use crate::utility;
use async_trait::async_trait;
use gqe_rs::api::{
    Catalog, ColumnSchema as GqeColumnSchema, PageKind, PartitioningSchemaKind, StorageKind,
};
use gqe_rs::executor::OptimizationParameters;
use gqe_rs::utility::CpuSet;
use log::trace;
use pgwire::api::query::SimpleQueryHandler;
use pgwire::api::results::{Response, Tag};
use pgwire::api::ClientInfo;
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use sqlparser::ast::{self, ColumnDef, FileFormat, Ident, ObjectName, SqlOption, Statement};
use sqlparser::parser::Parser;
use std::collections::HashMap;
use std::ops::DerefMut;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;

pub(super) struct GqeSession {
    catalog: Catalog,
    parameters: OptimizationParameters,
}

pub(super) struct GqeSessionService {
    session: Arc<Mutex<GqeSession>>,
}

impl GqeSessionService {
    pub(super) fn new() -> GqeSessionService {
        let catalog = Catalog::default();
        let parameters =
            OptimizationParameters::new().expect("Failed to instantiate optimization parameters.");

        let session = GqeSession {
            catalog,
            parameters,
        };

        GqeSessionService {
            session: Arc::new(Mutex::new(session)),
        }
    }
}

#[async_trait]
impl SimpleQueryHandler for GqeSessionService {
    async fn do_query<'a, C>(
        &self,
        _client: &mut C,
        query: &'a str,
    ) -> PgWireResult<Vec<Response<'a>>>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        let ast = Parser::parse_sql(&super::SQL_DIALECT, query)
            .map_err(|e| PgWireError::ApiError(Box::new(e)))?;

        trace!("Parsed: {:?}", ast);

        let statement: &Statement = ast.first().ok_or_else(|| {
            PgWireError::UserError(Box::new(ErrorInfo::new(
                "ERROR".to_owned(),
                PgErrorCode::SyntaxError.as_code().to_owned(),
                "No SQL statement found.".to_owned(),
            )))
        })?;
        match statement {
            Statement::CreateTable {
                external: true,
                name: ObjectName(idents),
                columns: sql_columns,
                file_format: Some(FileFormat::PARQUET),
                location: Some(ref dataset_location),
                ..
            } => {
                let [Ident {
                    value: ref table_name,
                    quote_style: None,
                }] = idents.as_slice()
                else {
                    return Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::InvalidName.as_code().to_owned(),
                        "Invalid table name.".to_owned(),
                    ))));
                };

                let columns = sql_columns
                    .iter()
                    .map(|c| {
                        if let ColumnDef {
                            name:
                                Ident {
                                    value: ref col_name,
                                    quote_style: None,
                                },
                            ref data_type,
                            ..
                        } = c
                        {
                            Ok(GqeColumnSchema {
                                column_name: col_name.to_owned(),
                                data_type: utility::convert_data_type_df_to_gqe(data_type)?,
                            })
                        } else {
                            Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                                "ERROR".to_owned(),
                                PgErrorCode::InvalidColumnDefinition.as_code().to_owned(),
                                "Invalid column definition.".to_owned(),
                            ))))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut session = self.session.lock().await;
                statement::create_table_from_parquet(
                    &mut session.catalog,
                    table_name,
                    columns.as_slice(),
                    dataset_location.as_str(),
                )
                .await?;

                Ok(vec![Response::Execution(Tag::new("OK"))])
            }
            Statement::CreateTable {
                name: ObjectName(dst_idents),
                like: Some(ObjectName(src_idents)),
                temporary: true,
                with_options,
                ..
            } => {
                let [Ident {
                    value: ref dst_table_name,
                    quote_style: None,
                }] = dst_idents.as_slice()
                else {
                    return Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::InvalidName.as_code().to_owned(),
                        "Invalid table name.".to_owned(),
                    ))));
                };

                let [Ident {
                    value: ref src_table_name,
                    quote_style: None,
                }] = src_idents.as_slice()
                else {
                    return Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::InvalidName.as_code().to_owned(),
                        "Invalid LIKE table name.".to_owned(),
                    ))));
                };

                let table_options = with_options
                    .iter()
                    .map(
                        |SqlOption {
                             ref name,
                             ref value,
                         }| (name.value.to_lowercase(), value),
                    )
                    .collect::<HashMap<_, _>>();

                let parameter_error = |var_name| {
                    PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::UndefinedParameter.as_code().to_owned(),
                        format!("{} received a wrong or unknown parameter.", var_name),
                    )))
                };

                let missing_error = |context_name, var_name| {
                    PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::UndefinedParameter.as_code().to_owned(),
                        format!("{} requires a \"{}\" parameter.", context_name, var_name,),
                    )))
                };

                let storage_kind = match table_options
                    .get("storage_kind")
                    .map(|v| utility::expr_value_to_str(v))
                    .transpose()?
                    .map(|s| s.to_lowercase())
                    .as_deref()
                {
                    None => StorageKind::PinnedMemory,
                    Some("system_memory") => StorageKind::SystemMemory,
                    Some("numa_memory") => {
                        let numa_node =
                            if let Some(ast::Expr::Value(ast::Value::Number(val, false))) =
                                table_options.get("numa_node")
                            {
                                u16::from_str(val).map_err(move |_| parameter_error("numa_node"))?
                            } else {
                                return Err(missing_error("numa_memory", "numa_node"));
                            };
                        let mut numa_node_set = CpuSet::new();
                        numa_node_set.add(numa_node);
                        let numa_node_set = numa_node_set;

                        let page_kind = match table_options
                            .get("page_kind")
                            .map(|v| utility::expr_value_to_str(v))
                            .transpose()?
                            .map(|s| s.to_lowercase())
                            .as_deref()
                        {
                            Some("default") | Some("system_default") => PageKind::SystemDefault,
                            Some("small") => PageKind::Small,
                            Some("transparent_huge") => PageKind::TransparentHuge,
                            Some("huge2mb") => PageKind::Huge2Mb,
                            Some("huge1gb") => PageKind::Huge1Gb,
                            Some(_) => return Err(parameter_error("page_kind")),
                            None => return Err(missing_error("numa_memory", "page_kind")),
                        };

                        StorageKind::NumaMemory {
                            numa_node_set,
                            page_kind,
                        }
                    }
                    Some("pinned_memory") => StorageKind::PinnedMemory,
                    Some("device_memory") => {
                        let device_id =
                            if let Some(ast::Expr::Value(ast::Value::Number(val, false))) =
                                table_options.get("device_id")
                            {
                                i32::from_str(val).map_err(move |_| parameter_error("device_id"))?
                            } else {
                                return Err(missing_error("device_memory", "device_id"));
                            };

                        StorageKind::DeviceMemory { device_id }
                    }
                    Some("managed_memory") => StorageKind::ManagedMemory,
                    Some("parquet_file") => {
                        let file_path = match table_options
                            .get("file_path")
                            .map(|v| utility::expr_value_to_str(v))
                            .transpose()?
                        {
                            Some(s) => s.to_owned(),
                            None => return Err(missing_error("parquet_file", "file_path")),
                        };

                        StorageKind::ParquetFile {
                            file_paths: vec![file_path],
                        }
                    }
                    _ => {
                        return Err(pg_wire_usererror(
                            PgErrorSeverity::Error,
                            PgErrorCode::UndefinedParameter,
                            "Invalid storage kind.",
                        ))
                    }
                };

                let mut session = self.session.lock().await;

                // FIXME: parse partitioning schema kind from table parameters
                let partitioning_schema_kind = PartitioningSchemaKind::Automatic;

                statement::create_temporary_duplicate_table(
                    &mut session.catalog,
                    dst_table_name.as_str(),
                    src_table_name.as_str(),
                    &storage_kind,
                    &partitioning_schema_kind,
                )
                .await?;

                Ok(vec![Response::Execution(Tag::new("OK"))])
            }
            Statement::Execute {
                name:
                    Ident {
                        value: substrait_query_file,
                        ..
                    },
                parameters,
            } if parameters.is_empty() => {
                let duration = statement::execute_prepared_statement(
                    self.session.clone(),
                    substrait_query_file,
                )
                .await?;

                Ok(vec![Response::Execution(Tag::new(&format!(
                    "Elapsed: {:.*} s",
                    crate::TIMER_PRECISION,
                    duration
                )))])
            }
            Statement::SetVariable {
                variable: ObjectName(idents),
                value,
                hivevar: false,
                local: false,
            } => {
                let [Ident {
                    value: ref var_name,
                    ..
                }] = idents.as_slice()
                else {
                    return Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::SyntaxError.as_code().to_owned(),
                        "Syntax error in variable assignment.".to_owned(),
                    ))));
                };

                let error = || {
                    PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::UndefinedParameter.as_code().to_owned(),
                        format!("Parameter {} is not defined.", var_name),
                    )))
                };

                let mut session = self.session.lock().await;
                let session = session.deref_mut();

                let normalized_var_name = var_name.to_uppercase();
                match value.as_slice() {
                    [ast::Expr::Value(ast::Value::SingleQuotedString(ref val))]
                    | [ast::Expr::Value(ast::Value::Number(ref val, false))] => {
                        utility::try_set_optimization_parameter(
                            &mut session.parameters,
                            normalized_var_name.as_str(),
                            val,
                        )?;
                    }
                    [ast::Expr::Value(ast::Value::Boolean(ref val))] => {
                        utility::try_set_optimization_parameter(
                            &mut session.parameters,
                            normalized_var_name.as_str(),
                            val.to_string().as_ref(),
                        )?;
                    }
                    [ast::Expr::Identifier(ref val)]  => {
                        return Err(pg_wire_usererror(
                            PgErrorSeverity::Error,
                            PgErrorCode::InvalidParameterValue,
                            format!("Parameter {} is an identifier, not a string. Hint: SQL strings are single-quoted.", val).as_str()))}
                    _ => {
                        return Err(error());
                    }
                }

                Ok(vec![Response::Execution(Tag::new("OK"))])
            }
            Statement::Insert {
                or: None,
                ignore: false,
                into: true,
                table_name: ObjectName(idents),
                table_alias: None,
                columns,
                overwrite: false,
                source: Some(a_source),
                partitioned: None,
                after_columns,
                table: false,
                on: None,
                returning: None,
                replace_into: false,
                priority: None,
            } if matches!(*a_source.body, ast::SetExpr::Table(_))
                && columns.is_empty()
                && after_columns.is_empty() =>
            {
                let invalid_table_name_error = || {
                    PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::InvalidName.as_code().to_owned(),
                        "Invalid table name.".to_owned(),
                    )))
                };

                let [Ident {
                    value: dst_table_name,
                    quote_style: None,
                }] = idents.as_slice()
                else {
                    return Err(invalid_table_name_error());
                };

                let ast::SetExpr::Table(src_table) = a_source.body.as_ref() else {
                    return Err(PgWireError::UserError(Box::new(ErrorInfo::new(
                        "ERROR".to_owned(),
                        PgErrorCode::FeatureNotSupported.as_code().to_owned(),
                        "GQE currently only supports \"INSERT INTO dst TABLE src\".".to_owned(),
                    ))));
                };

                let ast::Table {
                    table_name: Some(src_table_name),
                    schema_name: None,
                } = src_table.as_ref()
                else {
                    return Err(invalid_table_name_error());
                };

                let duration = statement::insert_from_table(
                    self.session.clone(),
                    dst_table_name,
                    src_table_name,
                )
                .await?;

                Ok(vec![Response::Execution(Tag::new(&format!(
                    "Elapsed: {:.*} s",
                    crate::TIMER_PRECISION,
                    duration
                )))])
            }
            Statement::StartTransaction { .. } => {
                // Handle clients, e.g. Python Psycopg, that automatically execute BEGIN TRANSATION
                Ok(vec![Response::Execution(Tag::new(""))])
            }
            _ => Ok(vec![Response::Error(Box::new(ErrorInfo::new(
                "ERROR".to_owned(),
                PgErrorCode::FeatureNotSupported.as_code().to_owned(),
                "GQE is a prototype engine. This feature is not yet supported.".to_owned(),
            )))]),
        }
    }
}
