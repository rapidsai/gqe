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

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::substrait_ext::ddl::{IntList, StorageOptionValue, storage_option_value};

use arrow::array::{Array, BinaryArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::flight_descriptor::DescriptorType;
use arrow_flight::flight_service_client::FlightServiceClient;
use arrow_flight::sql::{
    CommandGetTables, CommandStatementSubstraitPlan, DoPutUpdateResult, ProstMessageExt,
    SubstraitPlan,
};
use arrow_flight::{Action, FlightData, FlightDescriptor};
use arrow_ipc::convert::try_schema_from_ipc_buffer;
use datafusion::catalog::TableProvider;
use datafusion::common::Statistics;
use datafusion::common::stats::Precision;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::datasource::MemTable;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::empty::EmptyExec;
use datafusion::prelude::{Expr, SessionContext};
use datafusion::sql::sqlparser::ast::{
    ColumnOption, CopyOption, CopySource, CopyTarget, CreateTable, CreateTableOptions,
    Expr as SqlExpr, FileFormat, HiveIOFormat, ObjectType, Set, SqlOption,
    Statement as SqlStatement, TableConstraint, Value, ValueWithSpan,
};
use datafusion::sql::sqlparser::dialect::PostgreSqlDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion_substrait::logical_plan::producer::to_substrait_plan;
use futures::TryStreamExt;
use log::info;
use prost::Message;

/// Maximum decoding message size for incoming Flight RPC responses.
const MAX_DECODING_MESSAGE_SIZE: usize = 512 * 1024 * 1024;


/// Connect to a Flight SQL server and return a configured client.
pub(crate) async fn connect(
    server_url: &str,
) -> Result<FlightServiceClient<tonic::transport::Channel>, Box<dyn std::error::Error>> {
    let channel = tonic::transport::Endpoint::new(server_url.to_string())?
        .connect()
        .await?;
    Ok(FlightServiceClient::new(channel).max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE))
}

pub async fn run_against(
    server_url: &str,
    sql: &str,
    parquet_output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let result = execute(server_url, sql).await?;
    present_result(&result, parquet_output).await
}

/// Execute a serialized `gqe.proto.PhysicalRelation` against a Flight SQL server
/// and print or persist the results.
pub async fn run_physical_plan_against(
    server_url: &str,
    plan_bytes: Vec<u8>,
    parquet_output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut client = connect(server_url).await?;
    let batches = execute_physical_plan(&mut client, plan_bytes).await?;
    present_result(&ExecuteResult::Query(batches), parquet_output).await
}

async fn present_result(
    result: &ExecuteResult,
    parquet_output: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    match result {
        ExecuteResult::Query(batches) => {
            if let Some(path) = parquet_output {
                write_batches_to_parquet(batches, path).await?;
                let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                println!("Wrote {total_rows} row(s) to {path}");
            } else {
                let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
                println!("Results ({total_rows} rows):");
                arrow::util::pretty::print_batches(batches)?;
            }
        }
        ExecuteResult::Statement(rows_affected) => {
            println!("Statement executed ({rows_affected} rows affected)");
        }
    }
    Ok(())
}

/// Result of executing SQL against a Flight SQL server.
pub enum ExecuteResult {
    /// Query result (SELECT, etc.) — zero or more record batches.
    Query(Vec<RecordBatch>),
    /// Statement result (CREATE, DROP, COPY, etc.) — number of rows affected,
    /// or -1 if the server did not report a count.
    Statement(i64),
}

/// Execute SQL via Flight SQL + Substrait and return the result.
pub async fn execute(
    server_url: &str,
    sql: &str,
) -> Result<ExecuteResult, Box<dyn std::error::Error>> {
    // 1. Connect to the server
    let mut client = connect(server_url).await?;

    // 2. Discover tables via GetTables (with schemas and row counts)
    let discovered_tables = discover_tables(&mut client).await?;

    info!(
        "Discovered {} table(s): {:?}",
        discovered_tables.len(),
        discovered_tables
            .iter()
            .map(|t| format!(
                "{}({})",
                t.name,
                t.row_count.map_or_else(|| "?".into(), |n| n.to_string())
            ))
            .collect::<Vec<_>>()
    );

    // 3. Build local SessionContext with discovered schemas (for SQL planning only)
    let ctx = SessionContext::new();
    for table_info in &discovered_tables {
        let provider = SchemaOnlyTable {
            schema: table_info.schema.clone(),
            row_count: table_info.row_count,
        };
        ctx.register_table(&table_info.name, Arc::new(provider))?;
    }

    // 4. Parse SQL into individual statements and execute
    // Some TPC-H queries (e.g. Q15) contain multiple statements (CREATE VIEW, SELECT, DROP VIEW)
    let statements = Parser::parse_sql(&PostgreSqlDialect {}, sql)?;
    let mut last_result = ExecuteResult::Query(vec![]);

    for stmt in statements {
        match &stmt {
            // CREATE TABLE (not CTAS) — register locally, encode as Substrait DDL
            SqlStatement::CreateTable(create) if create.query.is_none() => {
                let sql_text = stmt.to_string();
                info!(
                    "Sending CREATE TABLE to server: {}",
                    &sql_text[..sql_text.len().min(80)]
                );

                let table_name = create.name.to_string();
                // ctx.sql is used only to derive the Arrow schema from the declared
                // columns and register the table locally for planning. Strip the WITH
                // clause and any EXTERNAL/STORED AS/LOCATION clauses so DataFusion sees a
                // plain CREATE TABLE. (These clauses may not even be supported by DataFusion's planner, which rejects CREATE TABLE WITH (...)
                // outright with NotImplemented, for example.)
                let sql_for_datafusion = {
                    let mut c = create.clone();
                    c.table_options = CreateTableOptions::None;
                    c.external = false;
                    c.file_format = None;
                    c.location = None;
                    c.hive_formats = None;
                    SqlStatement::CreateTable(c).to_string()
                };
                ctx.sql(&sql_for_datafusion).await?;
                let schema: SchemaRef =
                    Arc::new(ctx.table(&table_name).await?.schema().as_arrow().clone());

                // Collect all unique key constraints (single-column and composite) as key-sets.
                let column_names: Vec<String> = create
                    .columns
                    .iter()
                    .map(|c| c.name.value.to_lowercase())
                    .collect();
                let mut unique_keys: Vec<Vec<u32>> = Vec::new();

                // Inline: col TYPE ... UNIQUE / PRIMARY KEY (always single-column)
                for (idx, col_def) in create.columns.iter().enumerate() {
                    for opt in &col_def.options {
                        if matches!(opt.option, ColumnOption::Unique { .. }) {
                            unique_keys.push(vec![idx as u32]);
                        }
                    }
                }

                // Table-level: PRIMARY KEY (cols...) / UNIQUE (cols...)
                for constraint in &create.constraints {
                    let cols = match constraint {
                        TableConstraint::PrimaryKey { columns, .. } => Some(columns.as_slice()),
                        TableConstraint::Unique { columns, .. } => Some(columns.as_slice()),
                        _ => None,
                    };
                    let Some(cols) = cols else { continue };

                    // Resolve every column to its index; skip the whole key if any can't be
                    // resolved.
                    let mut indices: Vec<u32> = Vec::with_capacity(cols.len());
                    let mut all_resolved = true;
                    for col in cols {
                        if let SqlExpr::Identifier(ident) = &col.column.expr {
                            let name = ident.value.to_lowercase();
                            if let Some(idx) = column_names.iter().position(|n| n == &name) {
                                indices.push(idx as u32);
                                continue;
                            }
                        }
                        all_resolved = false;
                        break;
                    }
                    if all_resolved && !indices.is_empty() {
                        unique_keys.push(indices);
                    }
                }

                let storage_options = storage_options_for_create_table(create)?;

                let plan_bytes = if create.or_replace {
                    crate::substrait_ext::ddl::encode_create_or_replace_table(
                        &table_name,
                        schema.as_ref(),
                        &unique_keys,
                        &storage_options,
                    )
                } else {
                    crate::substrait_ext::ddl::encode_create_table(
                        &table_name,
                        schema.as_ref(),
                        &unique_keys,
                        &storage_options,
                    )
                };
                let rows = execute_substrait_update(&mut client, plan_bytes).await?;
                last_result = ExecuteResult::Statement(rows);
            }

            // DROP TABLE — encode as Substrait DDL
            SqlStatement::Drop {
                object_type: ObjectType::Table,
                if_exists,
                names,
                ..
            } => {
                let table_name = names[0].to_string();
                info!("Sending DROP TABLE to server: {table_name}");

                let plan_bytes = if *if_exists {
                    crate::substrait_ext::ddl::encode_drop_table_if_exists(&table_name)
                } else {
                    crate::substrait_ext::ddl::encode_drop_table(&table_name)
                };
                let rows = execute_substrait_update(&mut client, plan_bytes).await?;
                last_result = ExecuteResult::Statement(rows);
                let _ = ctx.deregister_table(&table_name);
            }

            // COPY FROM — encode as Substrait Write
            SqlStatement::Copy {
                to: false,
                source:
                    CopySource::Table {
                        table_name: copy_table,
                        ..
                    },
                target: CopyTarget::File { filename },
                options,
                ..
            } => {
                let table_name = copy_table.to_string();
                let file_path = filename.clone();
                let format = options
                    .iter()
                    .find_map(|o| match o {
                        CopyOption::Format(ident) => Some(ident.value.to_lowercase()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "parquet".to_string());
                info!("Sending COPY to server: {table_name} FROM '{file_path}' (FORMAT {format})");

                let schema: SchemaRef =
                    Arc::new(ctx.table(&table_name).await?.schema().as_arrow().clone());

                let plan_bytes = crate::substrait_ext::write::encode_copy_from(
                    &table_name,
                    &file_path,
                    schema.as_ref(),
                );
                let rows_affected = execute_substrait_update(&mut client, plan_bytes).await?;
                last_result = ExecuteResult::Statement(rows_affected);
                info!("COPY loaded {rows_affected} row(s) into {table_name}");

                ctx.deregister_table(&table_name)?;
                let provider = SchemaOnlyTable {
                    schema,
                    row_count: None,
                };
                ctx.register_table(&table_name, Arc::new(provider))?;
            }

            // SHOW <name> | SHOW ALL
            SqlStatement::ShowVariable { variable } => {
                let name = variable.iter().map(|i| i.value.as_str()).collect::<Vec<_>>().join(".");
                let batches = execute_get_session_options(&mut client, &name).await?;
                last_result = ExecuteResult::Query(batches);
            }

            // SET optimization_parameter TO value
            SqlStatement::Set(Set::SingleAssignment { variable, values, .. }) => {
                let name = variable.to_string();
                let val = values.first().ok_or("SET statement has no value")?;
                let option_value = infer_session_option_value(val);
                info!("Sending SET {name} to server");
                execute_set_session_options(&mut client, &name, option_value).await?;
                last_result = ExecuteResult::Statement(0);
            }

            // All other SQL — plan via DataFusion and send as Substrait
            _ => {
                let sql_text = stmt.to_string();
                info!("Planning SQL: {}", &sql_text[..sql_text.len().min(80)]);
                let df = ctx.sql(&sql_text).await?;
                let plan = df.into_optimized_plan()?;

                let substrait_plan = to_substrait_plan(&plan, &ctx.state())?;

                let mut plan_bytes = Vec::new();
                substrait_plan.encode(&mut plan_bytes)?;
                info!("Substrait plan size: {} bytes", plan_bytes.len());

                let batches = execute_substrait_plan(&mut client, plan_bytes).await?;
                last_result = ExecuteResult::Query(batches);
            }
        }
    }

    Ok(last_result)
}

/// Metadata about a discovered table.
pub struct DiscoveredTable {
    pub name: String,
    pub schema: SchemaRef,
    pub row_count: Option<usize>,
}

/// Connect to a server, perform handshake, and discover tables with schemas and row counts.
pub async fn discover(
    server_url: &str,
) -> Result<Vec<DiscoveredTable>, Box<dyn std::error::Error>> {
    let mut client = connect(server_url).await?;
    discover_tables(&mut client).await
}

/// A schema-only TableProvider that holds no data but reports statistics.
/// Used on the client side so the optimizer has row counts for cost-based decisions.
#[derive(Debug)]
struct SchemaOnlyTable {
    schema: SchemaRef,
    row_count: Option<usize>,
}

#[async_trait::async_trait]
impl TableProvider for SchemaOnlyTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> datafusion::logical_expr::TableType {
        datafusion::logical_expr::TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let projected_schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => self.schema.clone(),
        };
        Ok(Arc::new(EmptyExec::new(projected_schema)))
    }

    fn statistics(&self) -> Option<Statistics> {
        self.row_count.map(|n| Statistics {
            num_rows: Precision::Exact(n),
            total_byte_size: Precision::Absent,
            column_statistics: vec![],
        })
    }
}

impl fmt::Display for SchemaOnlyTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SchemaOnlyTable")
    }
}

async fn discover_tables(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
) -> Result<Vec<DiscoveredTable>, Box<dyn std::error::Error>> {
    let get_tables = CommandGetTables {
        catalog: None,
        db_schema_filter_pattern: None,
        table_name_filter_pattern: None,
        table_types: vec![],
        include_schema: true,
    };

    let descriptor = FlightDescriptor {
        r#type: DescriptorType::Cmd as i32,
        cmd: get_tables.as_any().encode_to_vec().into(),
        path: vec![],
    };

    let request = tonic::Request::new(descriptor);

    let flight_info = client.get_flight_info(request).await?.into_inner();
    let ticket = flight_info.endpoint[0].ticket.clone().unwrap();

    let do_get_request = tonic::Request::new(ticket);

    let stream = client.do_get(do_get_request).await?.into_inner();
    let batches = decode_flight_data_stream(stream).await?;

    let mut discovered_tables = vec![];
    for batch in &batches {
        let table_names = batch
            .column_by_name("table_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let table_schemas = batch
            .column_by_name("table_schema")
            .unwrap()
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let row_counts = batch
            .column_by_name("row_count")
            .and_then(|c| c.as_any().downcast_ref::<Int64Array>());

        for i in 0..batch.num_rows() {
            let name = table_names.value(i).to_string();
            let schema_bytes = table_schemas.value(i);
            let schema = try_schema_from_ipc_buffer(schema_bytes)?;
            let row_count = row_counts
                .filter(|arr| !arr.is_null(i))
                .map(|arr| arr.value(i) as usize);
            discovered_tables.push(DiscoveredTable {
                name,
                schema: Arc::new(schema),
                row_count,
            });
        }
    }
    Ok(discovered_tables)
}

async fn execute_physical_plan(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
    plan_bytes: Vec<u8>,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    // Bytes are already serialized; use prost::Name only for the type URL.
    let any = prost_types::Any {
        type_url: <crate::gqe::proto::PhysicalRelation as prost::Name>::type_url(),
        value: plan_bytes.into(),
    };

    let descriptor = FlightDescriptor {
        r#type: DescriptorType::Cmd as i32,
        cmd: any.encode_to_vec().into(),
        path: vec![],
    };

    let request = tonic::Request::new(descriptor);

    let flight_info = client.get_flight_info(request).await?.into_inner();

    let ticket = flight_info.endpoint[0].ticket.clone().unwrap();
    let do_get_request = tonic::Request::new(ticket);

    let stream = client.do_get(do_get_request).await?.into_inner();
    decode_flight_data_stream(stream).await
}

async fn execute_substrait_plan(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
    plan_bytes: Vec<u8>,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    let substrait_msg = SubstraitPlan {
        plan: plan_bytes.into(),
        version: "0.63.0".to_string(),
    };
    let cmd = CommandStatementSubstraitPlan {
        plan: Some(substrait_msg),
        transaction_id: None,
    };

    let descriptor = FlightDescriptor {
        r#type: DescriptorType::Cmd as i32,
        cmd: cmd.as_any().encode_to_vec().into(),
        path: vec![],
    };

    let request = tonic::Request::new(descriptor);

    let flight_info = client.get_flight_info(request).await?.into_inner();

    let ticket = flight_info.endpoint[0].ticket.clone().unwrap();
    let do_get_request = tonic::Request::new(ticket);

    let stream = client.do_get(do_get_request).await?.into_inner();
    decode_flight_data_stream(stream).await
}

/// Execute a Substrait plan via DoPut (for statements that don't return rows).
/// Returns the number of rows affected.
async fn execute_substrait_update(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
    plan_bytes: Vec<u8>,
) -> Result<i64, Box<dyn std::error::Error>> {
    let substrait_msg = SubstraitPlan {
        plan: plan_bytes.into(),
        version: "0.63.0".to_string(),
    };
    let cmd = CommandStatementSubstraitPlan {
        plan: Some(substrait_msg),
        transaction_id: None,
    };

    let descriptor = FlightDescriptor {
        r#type: DescriptorType::Cmd as i32,
        cmd: cmd.as_any().encode_to_vec().into(),
        path: vec![],
    };

    // DoPut expects a stream of FlightData.  The first message carries the
    // descriptor (the command); no record batches follow for a pure statement.
    let flight_data = FlightData {
        flight_descriptor: Some(descriptor),
        ..Default::default()
    };

    let request = tonic::Request::new(futures::stream::iter(vec![flight_data]));

    let mut response_stream = client.do_put(request).await?.into_inner();

    // The server returns a single PutResult containing a serialized
    // DoPutUpdateResult with the record_count field.
    if let Some(put_result) = response_stream.message().await? {
        let update_result = DoPutUpdateResult::decode(put_result.app_metadata)?;
        Ok(update_result.record_count)
    } else {
        Ok(0)
    }
}

/// Format a `SessionOptionValue` as a human-readable string for display.
fn format_session_option_value(v: &crate::flight_ext::session::SessionOptionValue) -> String {
    use crate::flight_ext::session::session_option_value::ValueType;
    match &v.value_type {
        Some(ValueType::BoolValue(b)) => b.to_string(),
        Some(ValueType::Int64Value(i)) => i.to_string(),
        Some(ValueType::DoubleValue(f)) => f.to_string(),
        Some(ValueType::StringValue(s)) => s.clone(),
        Some(ValueType::StringListValue(l)) => l.values.join(", "),
        None => "<unset>".to_string(),
    }
}

/// Send `GetSessionOptions` and return as a RecordBatch.
///
/// `name` is either `"ALL"` (case-insensitive) to return all options, or an exact key name.
async fn execute_get_session_options(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
    name: &str,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    use crate::flight_ext::session::{GetSessionOptionsRequest, GetSessionOptionsResult};
    use prost::Message;

    let body = GetSessionOptionsRequest {}.encode_to_vec();
    let action = Action { r#type: "GetSessionOptions".to_string(), body: body.into() };

    let mut stream = client.do_action(tonic::Request::new(action)).await?.into_inner();

    let mut all_options = std::collections::HashMap::new();
    while let Some(result) = stream.message().await? {
        let response = GetSessionOptionsResult::decode(result.body.as_ref())?;
        all_options.extend(response.session_options);
    }

    let mut names: Vec<String> = if name.eq_ignore_ascii_case("ALL") {
        all_options.keys().cloned().collect()
    } else {
        all_options.keys().filter(|k| k.as_str() == name).cloned().collect()
    };
    names.sort();

    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, false),
    ]));

    if names.is_empty() {
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }

    let name_col: arrow::array::StringArray =
        names.iter().map(|n| Some(n.as_str())).collect();
    let value_col: arrow::array::StringArray = names
        .iter()
        .map(|n| Some(format_session_option_value(all_options.get(n).unwrap())))
        .collect();

    Ok(vec![RecordBatch::try_new(schema, vec![Arc::new(name_col), Arc::new(value_col)])?])
}

/// Extract the WITH (...) clause of a CREATE TABLE statement as a typed option map.
///
/// Keys are lowercased. Values are encoded as `StorageOptionValue`:
/// single-quoted strings and identifiers become `StringVal`; integer literals
/// become `IntVal`; ARRAY[...] and tuple expressions of integers become
/// `IntListVal`; anything else falls back to `StringVal`.
fn extract_with_options(
    options: &CreateTableOptions,
) -> Result<HashMap<String, StorageOptionValue>, Box<dyn std::error::Error>> {
    let opts = match options {
        CreateTableOptions::With(opts) => opts.as_slice(),
        CreateTableOptions::None => return Ok(HashMap::new()),
        other => return Err(format!("Unsupported CREATE TABLE options syntax: {other}").into()),
    };
    let mut map = HashMap::with_capacity(opts.len());
    for opt in opts {
        match opt {
            SqlOption::KeyValue { key, value } => {
                let k = key.value.to_lowercase();
                let v = sql_expr_to_storage_option_value(value)?;
                if map.insert(k.clone(), v).is_some() {
                    return Err(format!("Duplicate WITH option: {k}").into());
                }
            }
            other => return Err(format!("Unsupported WITH option syntax: {other}").into()),
        }
    }
    // Parquet-backed tables are created via CREATE EXTERNAL TABLE, not the WITH clause.
    // The server still accepts a `parquet_file` storage option (that's how the external
    // table path encodes it), so it must be rejected here rather than left to fail
    // server-side, where it would silently succeed as a second way to make the same table.
    if let Some(StorageOptionValue {
        value: Some(storage_option_value::Value::StringVal(s)),
    }) = map.get("storage_kind")
    {
        if s.eq_ignore_ascii_case("parquet_file") {
            return Err("WITH (storage_kind = 'parquet_file') is not supported; \
                use CREATE EXTERNAL TABLE ... STORED AS PARQUET LOCATION '...'"
                .into());
        }
    }
    Ok(map)
}

/// Derive the storage option map for a CREATE TABLE statement.
///
/// A statement is treated as parquet-backed when it carries any file-backed clause
/// (STORED AS / LOCATION / ROW FORMAT), with or without the EXTERNAL keyword — EXTERNAL
/// is optional, matching the leniency of Hive/Spark. Routing every file-backed clause to
/// the parquet builder (rather than letting it fall through) is what prevents a stray
/// STORED AS / LOCATION from silently producing an empty in-memory table.
///
/// - `[EXTERNAL] TABLE ... STORED AS PARQUET LOCATION '...'` -> parquet_file option.
/// - `CREATE TABLE ... WITH (...)`                           -> in-memory storage options.
fn storage_options_for_create_table(
    create: &CreateTable,
) -> Result<HashMap<String, StorageOptionValue>, Box<dyn std::error::Error>> {
    // NB: sqlparser sets hive_formats to Some(empty) even for a plain CREATE TABLE, so test
    // its *content* (an actual STORED AS / LOCATION) rather than is_some() -- otherwise every
    // table would route to the parquet builder.
    let hive_file_backed = create
        .hive_formats
        .as_ref()
        .is_some_and(|hive| hive.storage.is_some() || hive.location.is_some());
    let is_file_backed = create.external
        || create.file_format.is_some()
        || create.location.is_some()
        || hive_file_backed;
    if is_file_backed {
        build_file_backed_storage_options(create)
    } else {
        extract_with_options(&create.table_options)
    }
}

/// Build the `parquet_file` storage option map for a `[EXTERNAL] TABLE ... STORED AS PARQUET
/// LOCATION '...'` statement. The result is the same option the server consumes, so no
/// server-side change is needed.
fn build_file_backed_storage_options(
    create: &CreateTable,
) -> Result<HashMap<String, StorageOptionValue>, Box<dyn std::error::Error>> {
    // STORED AS / LOCATION always populate hive_formats (the flat file_format/location
    // fields are only *additionally* set for CREATE EXTERNAL TABLE), so read both from
    // hive_formats uniformly, independent of the EXTERNAL keyword.
    let (format, location) = match &create.hive_formats {
        Some(hive) => {
            let format = match &hive.storage {
                Some(HiveIOFormat::FileFormat { format }) => Some(*format),
                _ => None,
            };
            (format, hive.location.clone())
        }
        None => (None, None),
    };

    match format {
        Some(FileFormat::PARQUET) => {}
        Some(other) => {
            return Err(format!("only STORED AS PARQUET is supported, got {other:?}").into());
        }
        None => return Err("a file-backed table requires STORED AS PARQUET".into()),
    }
    let location = location.ok_or("a file-backed table requires a LOCATION")?;
    // STORED AS / LOCATION and WITH (...) are two ways to specify storage; reject the
    // contradiction.
    if !matches!(create.table_options, CreateTableOptions::None) {
        return Err("a file-backed table does not accept a WITH clause".into());
    }
    let string_val = |v: &str| StorageOptionValue {
        value: Some(storage_option_value::Value::StringVal(v.to_string())),
    };
    Ok(HashMap::from([
        ("storage_kind".to_string(), string_val("parquet_file")),
        ("location".to_string(), string_val(&location)),
    ]))
}

/// If every element of `iter` is an integer literal, return an `IntListVal`
/// carrying those integers; otherwise return a `StringVal` of `expr`'s SQL text.
fn int_list_or_string<'a, I>(iter: I, expr: &SqlExpr) -> storage_option_value::Value
where
    I: IntoIterator<Item = &'a SqlExpr>,
{
    let maybe_ints: Option<Vec<i64>> = iter
        .into_iter()
        .map(|e| match e {
            SqlExpr::Value(ValueWithSpan { value: Value::Number(n, _), .. }) => {
                n.parse::<i64>().ok()
            }
            _ => None,
        })
        .collect();
    match maybe_ints {
        Some(values) => storage_option_value::Value::IntListVal(IntList { values }),
        None => storage_option_value::Value::StringVal(expr.to_string()),
    }
}

/// Convert a SQL expression from a WITH clause into a typed `StorageOptionValue`.
///
/// Single-quoted strings become `StringVal` (quotes stripped).
/// Integer literals become `IntVal`. Non-integer numbers fall back to `StringVal`.
/// ARRAY[n, ...] expressions of integers become `IntListVal`;
/// mixed-type arrays fall back to `StringVal` of the SQL text.
/// Unquoted identifiers are rejected: WITH option values must be explicit
/// quoted strings, numbers, or arrays.
fn sql_expr_to_storage_option_value(
    expr: &SqlExpr,
) -> Result<StorageOptionValue, Box<dyn std::error::Error>> {
    use storage_option_value::Value as Opt;
    let v = match expr {
        SqlExpr::Value(ValueWithSpan { value: Value::SingleQuotedString(s), .. }) => {
            Opt::StringVal(s.clone())
        }
        SqlExpr::Value(ValueWithSpan { value: Value::Number(n, _), .. }) => {
            if let Ok(i) = n.parse::<i64>() { Opt::IntVal(i) } else { Opt::StringVal(n.clone()) }
        }
        SqlExpr::Identifier(ident) => {
            return Err(format!(
                "WITH option value must be a quoted string, integer, or array; \
                 got bare identifier '{}'. Did you mean '{}'?",
                ident.value, ident.value
            )
            .into());
        }
        SqlExpr::Array(arr) => int_list_or_string(&arr.elem, expr),
        other => Opt::StringVal(other.to_string()),
    };
    Ok(StorageOptionValue { value: Some(v) })
}

/// Map a sqlparser expression to a typed `SessionOptionValue` variant.
///
/// Numeric literals are interpreted as `i64` when integral, `f64` otherwise.
/// Unrecognised expressions fall back to their string representation.
fn infer_session_option_value(
    expr: &SqlExpr,
) -> crate::flight_ext::session::session_option_value::ValueType {
    use crate::flight_ext::session::session_option_value::ValueType;
    match expr {
        SqlExpr::Value(ValueWithSpan { value: Value::Boolean(b), .. }) => ValueType::BoolValue(*b),
        SqlExpr::Value(ValueWithSpan { value: Value::Number(s, _), .. }) => {
            if let Ok(i) = s.parse::<i64>() {
                ValueType::Int64Value(i)
            } else if let Ok(f) = s.parse::<f64>() {
                ValueType::DoubleValue(f)
            } else {
                ValueType::StringValue(s.clone())
            }
        }
        SqlExpr::Value(ValueWithSpan { value: Value::SingleQuotedString(s), .. }) => {
            ValueType::StringValue(s.clone())
        }
        other => ValueType::StringValue(other.to_string()),
    }
}

/// Send a `SetSessionOptions` DoAction to the server for a single (name, value) pair.
async fn execute_set_session_options(
    client: &mut FlightServiceClient<tonic::transport::Channel>,
    name: &str,
    value: crate::flight_ext::session::session_option_value::ValueType,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::flight_ext::session::{SessionOptionValue, SetSessionOptionsRequest};
    use prost::Message;

    let mut session_options = std::collections::HashMap::new();
    session_options.insert(
        name.to_string(),
        SessionOptionValue { value_type: Some(value) },
    );
    let body = SetSessionOptionsRequest { session_options }.encode_to_vec();

    let action = Action {
        r#type: "SetSessionOptions".to_string(),
        body: body.into(),
    };

    let mut stream = client.do_action(tonic::Request::new(action)).await?.into_inner();
    while stream.message().await?.is_some() {}
    Ok(())
}

async fn write_batches_to_parquet(
    batches: &[RecordBatch],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => return Ok(()),
    };
    let ctx = SessionContext::new();
    let table = MemTable::try_new(schema, vec![batches.to_vec()])?;
    ctx.register_table("__output", Arc::new(table))?;
    ctx.table("__output")
        .await?
        .write_parquet(path, DataFrameWriteOptions::new(), None)
        .await?;
    Ok(())
}

async fn decode_flight_data_stream(
    stream: tonic::Streaming<FlightData>,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    let mut stream = FlightRecordBatchStream::new_from_flight_data(
        stream.map_err(|e| FlightError::Tonic(Box::new(e))),
    );

    let mut batches = vec![];
    while let Some(batch) = stream.try_next().await? {
        batches.push(batch);
    }
    Ok(batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::sql::sqlparser::dialect::PostgreSqlDialect;
    use datafusion::sql::sqlparser::parser::Parser;

    /// Parse a CREATE TABLE statement and return its table_options.
    fn parse_with_options(sql: &str) -> CreateTableOptions {
        parse_create_table(sql).table_options
    }

    /// Parse a CREATE [EXTERNAL] TABLE statement and return the CreateTable AST.
    fn parse_create_table(sql: &str) -> CreateTable {
        let mut stmts = Parser::parse_sql(&PostgreSqlDialect {}, sql).unwrap();
        match stmts.remove(0) {
            SqlStatement::CreateTable(ct) => ct,
            _ => panic!("expected CREATE TABLE"),
        }
    }

    fn str_val(s: &str) -> StorageOptionValue {
        StorageOptionValue { value: Some(storage_option_value::Value::StringVal(s.to_string())) }
    }
    fn int_val(n: i64) -> StorageOptionValue {
        StorageOptionValue { value: Some(storage_option_value::Value::IntVal(n)) }
    }
    fn int_list_val(ns: &[i64]) -> StorageOptionValue {
        StorageOptionValue {
            value: Some(storage_option_value::Value::IntListVal(IntList {
                values: ns.to_vec(),
            })),
        }
    }

    #[test]
    fn extract_with_options_no_with_clause() {
        let opts = parse_with_options("CREATE TABLE t (x INT)");
        let map = extract_with_options(&opts).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn extract_with_options_quoted_string_value() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = 'device_memory')",
        );
        let map = extract_with_options(&opts).unwrap();
        assert_eq!(map["storage_kind"], str_val("device_memory"));
    }

    #[test]
    fn extract_with_options_integer_value() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = 'device_memory', device_id = 2)",
        );
        let map = extract_with_options(&opts).unwrap();
        assert_eq!(map["device_id"], int_val(2));
    }

    #[test]
    fn extract_with_options_unquoted_identifier_value_rejected() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = device_memory)",
        );
        let err = extract_with_options(&opts).unwrap_err();
        assert!(err.to_string().contains("device_memory"));
    }

    #[test]
    fn extract_with_options_key_is_lowercased() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (Storage_Kind = 'device_memory')",
        );
        let map = extract_with_options(&opts).unwrap();
        assert!(map.contains_key("storage_kind"), "key should be lowercased");
        assert!(!map.contains_key("Storage_Kind"));
    }

    #[test]
    fn extract_with_options_array_integer_list() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = 'numa_pinned_memory', numa_node_set = ARRAY[0,1,2])",
        );
        let map = extract_with_options(&opts).unwrap();
        assert_eq!(map["numa_node_set"], int_list_val(&[0, 1, 2]));
    }

    #[test]
    fn extract_with_options_duplicate_key_case_insensitive_fails() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = 'device_memory', Storage_Kind = 'system_memory')",
        );
        assert!(extract_with_options(&opts).is_err());
    }

    #[test]
    fn extract_with_options_multiple_keys() {
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = 'numa_pinned_memory', numa_node_set = ARRAY[0, 1], page_kind = 'huge2mb')",
        );
        let map = extract_with_options(&opts).unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map["storage_kind"], str_val("numa_pinned_memory"));
        assert_eq!(map["numa_node_set"], int_list_val(&[0, 1]));
        assert_eq!(map["page_kind"], str_val("huge2mb"));
    }

    #[test]
    fn extract_with_options_rejects_parquet_file_storage_kind() {
        // Parquet-backed tables must be created via CREATE EXTERNAL TABLE, not WITH.
        let opts = parse_with_options(
            "CREATE TABLE t (x INT) WITH (storage_kind = 'parquet_file', location = '/data/t')",
        );
        let err = extract_with_options(&opts).unwrap_err();
        assert!(err.to_string().contains("CREATE EXTERNAL TABLE"));
    }

    #[test]
    fn parquet_storage_options_built_from_stored_as_and_location() {
        let create = parse_create_table(
            "CREATE EXTERNAL TABLE t (x INT) STORED AS PARQUET LOCATION '/data/t'",
        );
        let map = build_file_backed_storage_options(&create).unwrap();
        assert_eq!(map["storage_kind"], str_val("parquet_file"));
        assert_eq!(map["location"], str_val("/data/t"));
    }

    #[test]
    fn parquet_missing_location_fails() {
        let create = parse_create_table("CREATE EXTERNAL TABLE t (x INT) STORED AS PARQUET");
        assert!(build_file_backed_storage_options(&create).is_err());
    }

    #[test]
    fn parquet_non_parquet_format_fails() {
        let create =
            parse_create_table("CREATE EXTERNAL TABLE t (x INT) STORED AS ORC LOCATION '/data/t'");
        assert!(build_file_backed_storage_options(&create).is_err());
    }

    #[test]
    fn stored_as_parquet_without_external_is_accepted() {
        // EXTERNAL is optional: STORED AS PARQUET + LOCATION is parquet-backed either way.
        let create =
            parse_create_table("CREATE TABLE t (x INT) STORED AS PARQUET LOCATION '/data/t'");
        let map = storage_options_for_create_table(&create).unwrap();
        assert_eq!(map["storage_kind"], str_val("parquet_file"));
        assert_eq!(map["location"], str_val("/data/t"));
    }

    #[test]
    fn external_stored_as_parquet_dispatches_to_parquet_file() {
        let create = parse_create_table(
            "CREATE EXTERNAL TABLE t (x INT) STORED AS PARQUET LOCATION '/data/t'",
        );
        let map = storage_options_for_create_table(&create).unwrap();
        assert_eq!(map["storage_kind"], str_val("parquet_file"));
        assert_eq!(map["location"], str_val("/data/t"));
    }

    #[test]
    fn location_without_stored_as_fails() {
        // A file-backed clause (LOCATION) still routes to the parquet builder, which
        // rejects it for lacking STORED AS PARQUET rather than silently ignoring it.
        let create = parse_create_table("CREATE TABLE t (x INT) LOCATION '/data/t'");
        assert!(storage_options_for_create_table(&create).is_err());
    }

    #[test]
    fn plain_create_table_uses_with_options() {
        let create = parse_create_table("CREATE TABLE t (x INT) WITH (storage_kind = 'numa_memory')");
        let map = storage_options_for_create_table(&create).unwrap();
        assert_eq!(map["storage_kind"], str_val("numa_memory"));
    }

    // The following exercise the non-EXTERNAL spellings, where format/location live only in
    // hive_formats (the flat fields are empty) -- they guard against reintroducing a flat-field
    // dependency in the parquet builder.

    #[test]
    fn stored_as_parquet_clause_order_independent() {
        // LOCATION before STORED AS, no EXTERNAL.
        let create =
            parse_create_table("CREATE TABLE t (x INT) LOCATION '/data/t' STORED AS PARQUET");
        let map = storage_options_for_create_table(&create).unwrap();
        assert_eq!(map["storage_kind"], str_val("parquet_file"));
        assert_eq!(map["location"], str_val("/data/t"));
    }

    #[test]
    fn create_or_replace_stored_as_parquet_without_external() {
        let create = parse_create_table(
            "CREATE OR REPLACE TABLE t (x INT) STORED AS PARQUET LOCATION '/data/t'",
        );
        let map = storage_options_for_create_table(&create).unwrap();
        assert_eq!(map["storage_kind"], str_val("parquet_file"));
        assert_eq!(map["location"], str_val("/data/t"));
    }

    #[test]
    fn non_parquet_format_without_external_fails() {
        // Format is read from hive_formats on the reject path too.
        let create = parse_create_table("CREATE TABLE t (x INT) STORED AS ORC LOCATION '/data/t'");
        assert!(storage_options_for_create_table(&create).is_err());
    }
}
