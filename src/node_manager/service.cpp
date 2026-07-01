/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <gqe/node_manager/service.hpp>

#include <gqe/catalog.hpp>
#include <gqe/utility/cudf_to_arrow.hpp>
#include <gqe/utility/uuid.hpp>

#include <proto/physical_plan.pb.h>

#include <arrow/api.h>
#include <arrow/builder.h>
#include <arrow/flight/sql/FlightSql.pb.h>
#include <arrow/flight/sql/server.h>
#include <arrow/flight/types.h>
#include <arrow/io/api.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <google/protobuf/any.pb.h>

#include <string_view>
#include <vector>

namespace gqe::node_manager {

namespace {

namespace pb = arrow::flight::protocol;

arrow::Result<std::shared_ptr<arrow::Buffer>> serialize_schema_to_ipc(
  std::shared_ptr<arrow::Schema> const& schema)
{
  ARROW_ASSIGN_OR_RAISE(auto serialized, arrow::ipc::SerializeSchema(*schema));
  return serialized;
}

constexpr std::string_view default_catalog_name   = "gqe-default-catalog";
constexpr std::string_view default_db_schema_name = "gqe-default-schema";

}  // namespace

service::service(gqe::catalog* catalog,
                 execute_query_callback on_query,
                 execute_statement_callback on_statement,
                 execute_physical_plan_callback on_physical_plan,
                 set_option_callback on_set_option,
                 get_option_callback on_get_option)
  : _catalog(catalog),
    _on_query(std::move(on_query)),
    _on_statement(std::move(on_statement)),
    _on_physical_plan(std::move(on_physical_plan)),
    _on_set_option(std::move(on_set_option)),
    _on_get_option(std::move(on_get_option))
{
}

arrow::Status service::GetFlightInfo(arrow::flight::ServerCallContext const& /*context*/,
                                     arrow::flight::FlightDescriptor const& descriptor,
                                     std::unique_ptr<arrow::flight::FlightInfo>* info)
{
  google::protobuf::Any any;
  if (!any.ParseFromArray(descriptor.cmd.data(), static_cast<int>(descriptor.cmd.size()))) {
    return arrow::Status::Invalid("Unable to parse command");
  }

  if (any.Is<proto::PhysicalRelation>()) {
    ARROW_ASSIGN_OR_RAISE(*info, execute_physical_plan_select(any.value(), descriptor));
    return arrow::Status::OK();
  }

  if (any.Is<pb::sql::CommandStatementSubstraitPlan>()) {
    pb::sql::CommandStatementSubstraitPlan cmd;
    if (!any.UnpackTo(&cmd)) {
      return arrow::Status::Invalid("Unable to unpack CommandStatementSubstraitPlan");
    }
    ARROW_ASSIGN_OR_RAISE(*info, execute_substrait_select(cmd.plan().plan(), descriptor));
    return arrow::Status::OK();
  }

  if (any.Is<pb::sql::CommandGetTables>()) {
    ARROW_ASSIGN_OR_RAISE(*info, build_get_tables_info(descriptor));
    return arrow::Status::OK();
  }

  return arrow::Status::NotImplemented("Command not recognized: ", any.type_url());
}

arrow::Status service::DoGet(arrow::flight::ServerCallContext const& /*context*/,
                             arrow::flight::Ticket const& request,
                             std::unique_ptr<arrow::flight::FlightDataStream>* stream)
{
  google::protobuf::Any any;
  if (!any.ParseFromArray(request.ticket.data(), static_cast<int>(request.ticket.size()))) {
    return arrow::Status::Invalid("Unable to parse ticket");
  }

  if (!any.Is<pb::sql::TicketStatementQuery>()) {
    return arrow::Status::NotImplemented("Ticket type not recognized: ", any.type_url());
  }

  pb::sql::TicketStatementQuery parsed;
  if (!any.UnpackTo(&parsed)) {
    return arrow::Status::Invalid("Unable to unpack TicketStatementQuery");
  }

  auto handle = utility::uuid::from_string(parsed.statement_handle());
  std::shared_ptr<arrow::RecordBatch> batch;
  {
    std::lock_guard guard(_results_mutex);
    auto it = _results.find(handle);
    if (it == _results.end()) {
      return arrow::Status::KeyError(fmt::format("Result handle not found: {}", handle));
    }
    batch = it->second;
    _results.erase(it);
  }

  ARROW_ASSIGN_OR_RAISE(auto reader, arrow::RecordBatchReader::Make({batch}));
  *stream = std::make_unique<arrow::flight::RecordBatchStream>(reader);
  return arrow::Status::OK();
}

arrow::Status service::DoPut(arrow::flight::ServerCallContext const& /*context*/,
                             std::unique_ptr<arrow::flight::FlightMessageReader> reader,
                             std::unique_ptr<arrow::flight::FlightMetadataWriter> writer)
{
  auto const& descriptor = reader->descriptor();
  google::protobuf::Any any;
  if (!any.ParseFromArray(descriptor.cmd.data(), static_cast<int>(descriptor.cmd.size()))) {
    return arrow::Status::Invalid("Unable to parse command");
  }

  int64_t record_count = -1;
  if (any.Is<pb::sql::CommandStatementSubstraitPlan>()) {
    pb::sql::CommandStatementSubstraitPlan cmd;
    if (!any.UnpackTo(&cmd)) {
      return arrow::Status::Invalid("Unable to unpack CommandStatementSubstraitPlan");
    }
    ARROW_ASSIGN_OR_RAISE(record_count, execute_substrait_write(cmd.plan().plan()));
  } else {
    return arrow::Status::NotImplemented("Command not recognized: ", any.type_url());
  }

  pb::sql::DoPutUpdateResult result;
  result.set_record_count(record_count);
  auto const buffer = arrow::Buffer::FromString(result.SerializeAsString());
  ARROW_RETURN_NOT_OK(writer->WriteMetadata(*buffer));
  return arrow::Status::OK();
}

arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> service::make_flight_info(
  std::shared_ptr<arrow::RecordBatch> batch, arrow::flight::FlightDescriptor const& descriptor)
{
  auto handle = utility::uuid::generate();
  {
    std::lock_guard guard(_results_mutex);
    _results[handle] = batch;
  }

  ARROW_ASSIGN_OR_RAISE(auto ticket_payload,
                        arrow::flight::sql::CreateStatementQueryTicket(handle.to_string()));
  std::vector<arrow::flight::FlightEndpoint> endpoints{
    arrow::flight::FlightEndpoint{arrow::flight::Ticket{std::move(ticket_payload)},
                                  /*locations=*/{},
                                  /*expiration_time=*/std::nullopt,
                                  /*app_metadata=*/""}};

  ARROW_ASSIGN_OR_RAISE(auto flight_info,
                        arrow::flight::FlightInfo::Make(*batch->schema(),
                                                        descriptor,
                                                        std::move(endpoints),
                                                        /*total_records=*/-1,
                                                        /*total_bytes=*/-1,
                                                        /*ordered=*/false));

  return std::make_unique<arrow::flight::FlightInfo>(std::move(flight_info));
}

arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> service::execute_substrait_select(
  std::string const& plan_bytes, arrow::flight::FlightDescriptor const& descriptor)
{
  std::shared_ptr<arrow::RecordBatch> result_batch;
  try {
    result_batch = _on_query(plan_bytes.data(), plan_bytes.size());
  } catch (std::invalid_argument const& e) {
    return arrow::Status::Invalid(e.what());
  } catch (std::exception const& e) {
    return arrow::Status::UnknownError(e.what());
  }

  if (!result_batch) {
    auto empty_schema = arrow::schema({});
    result_batch =
      arrow::RecordBatch::Make(empty_schema, 0, std::vector<std::shared_ptr<arrow::Array>>{});
  }
  return make_flight_info(std::move(result_batch), descriptor);
}

arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> service::execute_physical_plan_select(
  std::string const& plan_bytes, arrow::flight::FlightDescriptor const& descriptor)
{
  std::shared_ptr<arrow::RecordBatch> result_batch;
  try {
    result_batch = _on_physical_plan(plan_bytes.data(), plan_bytes.size());
  } catch (std::invalid_argument const& e) {
    return arrow::Status::Invalid(e.what());
  } catch (std::exception const& e) {
    return arrow::Status::UnknownError(e.what());
  }

  if (!result_batch) {
    auto empty_schema = arrow::schema({});
    result_batch =
      arrow::RecordBatch::Make(empty_schema, 0, std::vector<std::shared_ptr<arrow::Array>>{});
  }
  return make_flight_info(std::move(result_batch), descriptor);
}

arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> service::build_get_tables_info(
  arrow::flight::FlightDescriptor const& descriptor)
{
  auto table_names = _catalog->table_names();

  arrow::StringBuilder catalog_name_builder;
  arrow::StringBuilder db_schema_name_builder;
  arrow::StringBuilder table_name_builder;
  arrow::StringBuilder table_type_builder;
  arrow::BinaryBuilder table_schema_builder;
  arrow::Int64Builder row_count_builder;

  for (auto const& name : table_names) {
    ARROW_RETURN_NOT_OK(catalog_name_builder.Append(default_catalog_name));
    ARROW_RETURN_NOT_OK(db_schema_name_builder.Append(default_db_schema_name));
    ARROW_RETURN_NOT_OK(table_name_builder.Append(name));
    ARROW_RETURN_NOT_OK(table_type_builder.Append("BASE TABLE"));

    auto const& col_names = _catalog->column_names(name);
    auto col_types        = _catalog->column_types(name);
    auto arrow_schema     = gqe::utility::build_arrow_schema(col_names, col_types);

    ARROW_ASSIGN_OR_RAISE(auto ipc_bytes, serialize_schema_to_ipc(arrow_schema));
    ARROW_RETURN_NOT_OK(
      table_schema_builder.Append(ipc_bytes->data(), static_cast<int32_t>(ipc_bytes->size())));

    auto* stats_mgr   = _catalog->statistics(name);
    int64_t row_count = stats_mgr->statistics().num_rows;
    ARROW_RETURN_NOT_OK(row_count_builder.Append(row_count));
  }

  ARROW_ASSIGN_OR_RAISE(auto catalog_name_array, catalog_name_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto db_schema_name_array, db_schema_name_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto table_name_array, table_name_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto table_type_array, table_type_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto table_schema_array, table_schema_builder.Finish());
  ARROW_ASSIGN_OR_RAISE(auto row_count_array, row_count_builder.Finish());

  auto response_schema = arrow::schema({arrow::field("catalog_name", arrow::utf8()),
                                        arrow::field("db_schema_name", arrow::utf8()),
                                        arrow::field("table_name", arrow::utf8()),
                                        arrow::field("table_type", arrow::utf8()),
                                        arrow::field("table_schema", arrow::binary()),
                                        arrow::field("row_count", arrow::int64())});

  auto batch = arrow::RecordBatch::Make(response_schema,
                                        static_cast<int64_t>(table_names.size()),
                                        {catalog_name_array,
                                         db_schema_name_array,
                                         table_name_array,
                                         table_type_array,
                                         table_schema_array,
                                         row_count_array});

  return make_flight_info(std::move(batch), descriptor);
}

arrow::Result<int64_t> service::execute_substrait_write(std::string const& plan_bytes)
{
  try {
    auto stats = _on_statement(plan_bytes.data(), plan_bytes.size());
    return stats.has_value() ? stats->num_rows : -1;
  } catch (std::invalid_argument const& e) {
    return arrow::Status::Invalid(e.what());
  } catch (std::exception const& e) {
    return arrow::Status::UnknownError(e.what());
  }
}

arrow::Status service::DoAction(arrow::flight::ServerCallContext const& context,
                                arrow::flight::Action const& action,
                                std::unique_ptr<arrow::flight::ResultStream>* result_stream)
{
  std::vector<arrow::flight::Result> results;
  if (action.type == arrow::flight::ActionType::kSetSessionOptions.type) {
    std::string_view body(*action.body);
    ARROW_ASSIGN_OR_RAISE(auto request, arrow::flight::SetSessionOptionsRequest::Deserialize(body));
    ARROW_ASSIGN_OR_RAISE(auto res, SetSessionOptions(context, request));
    ARROW_ASSIGN_OR_RAISE(auto packed, res.SerializeToBuffer());
    results.emplace_back(std::move(packed));
  } else if (action.type == arrow::flight::ActionType::kGetSessionOptions.type) {
    std::string_view body(*action.body);
    ARROW_ASSIGN_OR_RAISE(auto request, arrow::flight::GetSessionOptionsRequest::Deserialize(body));
    ARROW_ASSIGN_OR_RAISE(auto res, GetSessionOptions(context, request));
    ARROW_ASSIGN_OR_RAISE(auto packed, res.SerializeToBuffer());
    results.emplace_back(std::move(packed));
  } else {
    return arrow::Status::NotImplemented("Action not supported: ", action.type);
  }
  *result_stream = std::make_unique<arrow::flight::SimpleResultStream>(std::move(results));
  return arrow::Status::OK();
}

arrow::Result<arrow::flight::SetSessionOptionsResult> service::SetSessionOptions(
  arrow::flight::ServerCallContext const& /*context*/,
  arrow::flight::SetSessionOptionsRequest const& request)
{
  arrow::flight::SetSessionOptionsResult result;
  for (auto const& [name, value] : request.session_options) {
    auto status = _on_set_option(name, value);
    if (!status.ok()) {
      result.errors[name] = {status.IsKeyError()
                               ? arrow::flight::SetSessionOptionErrorValue::kInvalidName
                               : arrow::flight::SetSessionOptionErrorValue::kInvalidValue};
    }
  }
  return result;
}

arrow::Result<arrow::flight::GetSessionOptionsResult> service::GetSessionOptions(
  arrow::flight::ServerCallContext const& /*context*/,
  arrow::flight::GetSessionOptionsRequest const& /*request*/)
{
  ARROW_ASSIGN_OR_RAISE(auto options, _on_get_option());
  return arrow::flight::GetSessionOptionsResult{std::move(options)};
}

}  // namespace gqe::node_manager
