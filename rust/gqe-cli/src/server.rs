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

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::IpcWriteOptions;
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_descriptor::DescriptorType;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::Result as FlightResult;
use arrow_flight::sql::server::{FlightSqlService, PeekableFlightDataStream};
use arrow_flight::sql::{
    Any, CommandGetTables, CommandStatementSubstraitPlan, ProstMessageExt, SqlInfo,
};
use arrow_flight::{Action, FlightDescriptor, FlightEndpoint, FlightInfo, IpcMessage, SchemaAsIpc, Ticket};
use crate::flight_ext::session::{GetSessionOptionsResult, SessionOptionValue, SetSessionOptionsRequest};
use dashmap::DashMap;
use datafusion::datasource::MemTable;
use datafusion::prelude::{DataFrame, ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
use futures::{StreamExt, TryStreamExt};
use log::info;
use prost::Message;
use substrait::proto::{plan_rel, rel};
use tonic::transport::Server;
use tonic::{Request, Response, Status};
use uuid::Uuid;

macro_rules! status {
    ($desc:expr, $err:expr) => {
        Status::internal(format!("{}: {} at {}:{}", $desc, $err, file!(), line!()))
    };
}

pub struct SubstraitFlightSqlService {
    /// Shared session context for all clients. Tables created by any client are
    /// visible to all subsequent sessions, which is the expected behavior for a
    /// data warehouse server.
    ctx: Arc<SessionContext>,
    results: Arc<DashMap<String, Vec<RecordBatch>>>,
    session_options: Arc<DashMap<String, SessionOptionValue>>,
}

impl SubstraitFlightSqlService {
    fn get_ctx<T>(&self, _req: &Request<T>) -> Result<Arc<SessionContext>, Status> {
        Ok(self.ctx.clone())
    }

    fn get_result(&self, handle: &str) -> Result<Vec<RecordBatch>, Status> {
        if let Some(result) = self.results.get(handle) {
            Ok(result.clone())
        } else {
            Err(Status::internal(format!(
                "Request handle not found: {handle}"
            )))?
        }
    }

    async fn tables_with_schema(&self, ctx: Arc<SessionContext>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("catalog_name", DataType::Utf8, true),
            Field::new("db_schema_name", DataType::Utf8, true),
            Field::new("table_name", DataType::Utf8, false),
            Field::new("table_type", DataType::Utf8, false),
            Field::new("table_schema", DataType::Binary, false),
            Field::new("row_count", DataType::Int64, true),
        ]));

        let mut catalogs = vec![];
        let mut schemas = vec![];
        let mut names = vec![];
        let mut types = vec![];
        let mut table_schemas: Vec<Vec<u8>> = vec![];
        let mut row_counts: Vec<Option<i64>> = vec![];

        for catalog in ctx.catalog_names() {
            let catalog_provider = ctx.catalog(&catalog).unwrap();
            for schema_name in catalog_provider.schema_names() {
                let schema_provider = catalog_provider.schema(&schema_name).unwrap();
                for table in schema_provider.table_names() {
                    let table_provider = schema_provider.table(&table).await.unwrap().unwrap();
                    catalogs.push(catalog.clone());
                    schemas.push(schema_name.clone());
                    names.push(table.clone());
                    types.push(table_provider.table_type().to_string());

                    let arrow_schema = table_provider.schema();
                    let ipc_bytes = serialize_schema_to_ipc(&arrow_schema);
                    table_schemas.push(ipc_bytes);

                    // Get row count: try table statistics first, fall back to COUNT(*)
                    let row_count = match table_provider
                        .statistics()
                        .and_then(|s| s.num_rows.get_value().copied())
                    {
                        Some(n) => Some(n as i64),
                        None => query_row_count(&ctx, &table).await,
                    };
                    row_counts.push(row_count);
                }
            }
        }

        let table_schema_refs: Vec<&[u8]> = table_schemas.iter().map(|b| b.as_slice()).collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(catalogs)) as ArrayRef,
                Arc::new(StringArray::from(schemas)) as ArrayRef,
                Arc::new(StringArray::from(names)) as ArrayRef,
                Arc::new(StringArray::from(types)) as ArrayRef,
                Arc::new(BinaryArray::from(table_schema_refs)) as ArrayRef,
                Arc::new(Int64Array::from(row_counts)) as ArrayRef,
            ],
        )
        .unwrap()
    }
}

/// Handle a Substrait DdlRel (CREATE TABLE / DROP TABLE).
async fn handle_ddl_rel(
    ctx: &SessionContext,
    ddl: &substrait::proto::DdlRel,
) -> Result<Vec<RecordBatch>, Status> {
    use substrait::proto::ddl_rel::DdlOp;

    let decoded = crate::substrait_ext::ddl::decode_ddl(ddl)
        .map_err(|e| status!("Failed to decode DdlRel", e))?;

    match decoded.op {
        DdlOp::Create => {
            let schema = Arc::new(
                decoded
                    .schema
                    .ok_or_else(|| Status::internal("CREATE TABLE has no schema"))?,
            );

            info!(
                "CREATE TABLE {} with {} columns",
                decoded.table_name,
                schema.fields().len()
            );

            let mem_table = MemTable::try_new(schema, vec![vec![]])
                .map_err(|e| status!("Failed to create MemTable", e))?;
            ctx.register_table(decoded.table_name.as_str(), Arc::new(mem_table))
                .map_err(|e| status!("Failed to register table", e))?;

            Ok(vec![])
        }
        DdlOp::Drop | DdlOp::DropIfExist => {
            info!("DROP TABLE {}", decoded.table_name);
            ctx.deregister_table(decoded.table_name.as_str())
                .map_err(|e| status!("Failed to deregister table", e))?;
            Ok(vec![])
        }
        other => Err(Status::unimplemented(format!(
            "DDL operation not supported: {other:?}"
        ))),
    }
}

/// Handle a Substrait WriteRel (COPY FROM).
async fn handle_write_rel(
    ctx: &SessionContext,
    write: &substrait::proto::WriteRel,
) -> Result<Vec<RecordBatch>, Status> {
    let decoded = crate::substrait_ext::write::decode_copy_from(write)
        .map_err(|e| status!("Failed to decode WriteRel", e))?;

    info!("COPY {} FROM '{}'", decoded.table_name, decoded.file_path);

    // Deregister existing (empty) table, then register parquet source
    let _ = ctx.deregister_table(decoded.table_name.as_str());
    let opts = ParquetReadOptions::default();
    ctx.register_parquet(decoded.table_name.as_str(), &decoded.file_path, opts)
        .await
        .map_err(|e| {
            status!(
                format!("Failed to register parquet for '{}'", decoded.table_name),
                e
            )
        })?;

    Ok(vec![])
}

/// Extract the top-level Rel from a Substrait plan, looking through both
/// `PlanRel::Rel` and `PlanRel::Root` wrappers.
fn extract_top_rel(plan: &substrait::proto::Plan) -> Option<&substrait::proto::Rel> {
    let plan_rel = plan.relations.first()?;
    match &plan_rel.rel_type {
        Some(plan_rel::RelType::Rel(rel)) => Some(rel),
        Some(plan_rel::RelType::Root(root)) => root.input.as_ref(),
        None => None,
    }
}

async fn query_row_count(ctx: &SessionContext, table_name: &str) -> Option<i64> {
    let table = datafusion::common::TableReference::bare(table_name);
    let count = ctx.table(table).await.ok()?.count().await.ok()?;
    Some(count as i64)
}

fn serialize_schema_to_ipc(schema: &Arc<Schema>) -> Vec<u8> {
    let options = IpcWriteOptions::default();
    let IpcMessage(bytes) = SchemaAsIpc::new(schema, &options)
        .try_into()
        .expect("IPC schema serialization failed");
    bytes.to_vec()
}

#[tonic::async_trait]
impl FlightSqlService for SubstraitFlightSqlService {
    type FlightService = SubstraitFlightSqlService;

    async fn get_flight_info_substrait_plan(
        &self,
        query: CommandStatementSubstraitPlan,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        info!("get_flight_info_substrait_plan");
        let ctx = self.get_ctx(&request)?;

        let substrait_plan_msg = query
            .plan
            .ok_or_else(|| Status::invalid_argument("Missing SubstraitPlan"))?;

        let plan = substrait::proto::Plan::decode(substrait_plan_msg.plan)
            .map_err(|e| Status::internal(format!("Failed to decode Substrait plan: {e}")))?;

        let results = execute_query_plan(&ctx, &plan).await?;

        let handle = Uuid::new_v4().hyphenated().to_string();
        let schema = match results.first() {
            None => Schema::empty(),
            Some(batch) => (*batch.schema()).clone(),
        };
        self.results.insert(handle.clone(), results);

        let fetch = FetchResults {
            handle: handle.clone(),
        };
        let buf = fetch.as_any().encode_to_vec().into();
        let ticket = Ticket { ticket: buf };

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .expect("encoding failed")
            .with_endpoint(FlightEndpoint::new().with_ticket(ticket))
            .with_descriptor(FlightDescriptor {
                r#type: DescriptorType::Cmd.into(),
                cmd: Default::default(),
                path: vec![],
            });
        Ok(Response::new(info))
    }

    async fn do_put_substrait_plan(
        &self,
        query: CommandStatementSubstraitPlan,
        request: Request<PeekableFlightDataStream>,
    ) -> Result<i64, Status> {
        info!("do_put_substrait_plan");
        let ctx = self.get_ctx(&request)?;

        let substrait_plan_msg = query
            .plan
            .ok_or_else(|| Status::invalid_argument("Missing SubstraitPlan"))?;

        let plan = substrait::proto::Plan::decode(substrait_plan_msg.plan)
            .map_err(|e| Status::internal(format!("Failed to decode Substrait plan: {e}")))?;

        let top_rel = extract_top_rel(&plan)
            .ok_or_else(|| Status::invalid_argument("Plan has no relations"))?;

        match &top_rel.rel_type {
            Some(rel::RelType::Ddl(ddl)) => {
                handle_ddl_rel(&ctx, ddl).await?;
                Ok(-1)
            }
            Some(rel::RelType::Write(write)) => {
                handle_write_rel(&ctx, write).await?;
                let row_count = query_row_count(&ctx, &get_write_table_name(write)?).await;
                Ok(row_count.unwrap_or(-1))
            }
            _ => Err(Status::invalid_argument(
                "do_put_substrait_plan expects a DDL or Write plan",
            )),
        }
    }

    async fn get_flight_info_tables(
        &self,
        _query: CommandGetTables,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        info!("get_flight_info_tables");
        let ctx = self.get_ctx(&request)?;
        let data = self.tables_with_schema(ctx).await;
        let schema = data.schema();

        let uuid = Uuid::new_v4().hyphenated().to_string();
        self.results.insert(uuid.clone(), vec![data]);

        let fetch = FetchResults { handle: uuid };
        let buf = fetch.as_any().encode_to_vec().into();
        let ticket = Ticket { ticket: buf };

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .expect("encoding failed")
            .with_endpoint(FlightEndpoint::new().with_ticket(ticket))
            .with_descriptor(FlightDescriptor {
                r#type: DescriptorType::Cmd.into(),
                cmd: Default::default(),
                path: vec![],
            });
        Ok(Response::new(info))
    }

    async fn do_get_fallback(
        &self,
        _request: Request<Ticket>,
        message: Any,
    ) -> Result<Response<<Self as FlightService>::DoGetStream>, Status> {
        if !message.is::<FetchResults>() {
            Err(Status::unimplemented(format!(
                "do_get: The defined request is invalid: {}",
                message.type_url
            )))?
        }

        let fr: FetchResults = message
            .unpack()
            .map_err(|e| Status::internal(format!("{e:?}")))?
            .ok_or_else(|| Status::internal("Expected FetchResults but got None!"))?;

        let handle = fr.handle;
        info!("getting results for {handle}");
        let result = self.get_result(&handle)?;

        let (schema, batches) = match result.first() {
            None => (Arc::new(Schema::empty()), vec![]),
            Some(batch) => (batch.schema(), result.clone()),
        };

        let batch_stream = futures::stream::iter(batches).map(Ok);
        let stream = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(batch_stream)
            .map_err(Status::from);

        Ok(Response::new(Box::pin(stream)))
    }

    async fn do_action_fallback(
        &self,
        request: Request<Action>,
    ) -> Result<Response<<Self as FlightService>::DoActionStream>, Status> {
        let action = request.into_inner();
        match action.r#type.as_str() {
            "SetSessionOptions" => {
                let req = SetSessionOptionsRequest::decode(action.body)
                    .map_err(|e| Status::invalid_argument(format!("decode SetSessionOptionsRequest: {e}")))?;
                for (name, value) in req.session_options {
                    self.session_options.insert(name, value);
                }
                let stream = futures::stream::empty::<std::result::Result<FlightResult, Status>>();
                Ok(Response::new(Box::pin(stream)))
            }
            "GetSessionOptions" => {
                let options: std::collections::HashMap<String, SessionOptionValue> = self
                    .session_options
                    .iter()
                    .map(|e| (e.key().clone(), e.value().clone()))
                    .collect();
                let result = GetSessionOptionsResult { session_options: options };
                let body = result.encode_to_vec().into();
                let item = FlightResult { body };
                let stream =
                    futures::stream::once(async move { Ok::<FlightResult, Status>(item) });
                Ok(Response::new(Box::pin(stream)))
            }
            _ => Err(Status::unimplemented(format!(
                "do_action: unknown action type: {}",
                action.r#type
            ))),
        }
    }

    async fn register_sql_info(&self, _id: i32, _result: &SqlInfo) {}
}

/// Extract the target table name from a WriteRel.
fn get_write_table_name(write: &substrait::proto::WriteRel) -> Result<String, Status> {
    use substrait::proto::write_rel::WriteType;
    match &write.write_type {
        Some(WriteType::NamedTable(named)) => named
            .names
            .last()
            .cloned()
            .ok_or_else(|| Status::internal("WriteRel NamedTable has no names")),
        _ => Err(Status::internal("WriteRel has no named table write type")),
    }
}

/// Execute a standard query Substrait plan via DataFusion.
async fn execute_query_plan(
    ctx: &SessionContext,
    plan: &substrait::proto::Plan,
) -> Result<Vec<RecordBatch>, Status> {
    let logical_plan = from_substrait_plan(&ctx.state(), plan)
        .await
        .map_err(|e| Status::internal(format!("Failed to consume Substrait plan: {e}")))?;

    let df = DataFrame::new(ctx.state(), logical_plan);
    df.collect()
        .await
        .map_err(|e| Status::internal(format!("Execution error: {e}")))
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FetchResults {
    #[prost(string, tag = "1")]
    pub handle: ::prost::alloc::string::String,
}

impl ProstMessageExt for FetchResults {
    fn type_url() -> &'static str {
        "type.googleapis.com/datafusion.example.com.sql.FetchResults"
    }

    fn as_any(&self) -> Any {
        Any {
            type_url: FetchResults::type_url().to_string(),
            value: ::prost::Message::encode_to_vec(self).into(),
        }
    }
}

pub async fn serve(addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let session_config = SessionConfig::from_env()?.with_information_schema(true);
    let ctx = Arc::new(SessionContext::new_with_config(session_config));
    let service = SubstraitFlightSqlService {
        ctx,
        results: Default::default(),
        session_options: Default::default(),
    };
    let svc = FlightServiceServer::new(service);
    info!("Listening on {addr}");
    Server::builder().add_service(svc).serve(addr).await?;
    Ok(())
}
