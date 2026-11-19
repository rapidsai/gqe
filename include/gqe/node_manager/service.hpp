/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <gqe/types.hpp>
#include <gqe/utility/uuid.hpp>

#include <arrow/flight/server.h>
#include <arrow/flight/sql/server.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace gqe {
class catalog;
}

namespace gqe::node_manager {

/**
 * @brief Flight server that handles client RPCs for the node manager.
 *
 * Inherits `arrow::flight::FlightServerBase` rather than `FlightSqlServerBase`
 * because the SQL base marks its top-level `GetFlightInfo`/`DoGet`/`DoPut`
 * `final`, which prevents dispatching non-SQL command types. Dispatch is
 * therefore done here via `google::protobuf::Any`.
 */
class service : public arrow::flight::FlightServerBase {
 public:
  /** @brief Callback invoked to execute a Substrait SELECT plan. */
  using execute_query_callback =
    std::function<std::shared_ptr<arrow::RecordBatch>(void const* data, std::size_t size)>;

  /** @brief Callback invoked to execute a Substrait write/DDL plan. */
  using execute_statement_callback =
    std::function<std::optional<gqe::table_statistics>(void const* data, std::size_t size)>;

  /** @brief Callback invoked to execute a serialized `proto::PhysicalRelation` SELECT plan. */
  using execute_physical_plan_callback =
    std::function<std::shared_ptr<arrow::RecordBatch>(void const* data, std::size_t size)>;

  /** @brief Callback invoked to apply a single session option by name and value. */
  using set_option_callback =
    std::function<arrow::Status(std::string const&, arrow::flight::SessionOptionValue const&)>;

  /** @brief Callback invoked to retrieve all current optimization parameters. */
  using get_option_callback =
    std::function<arrow::Result<std::map<std::string, arrow::flight::SessionOptionValue>>()>;

  /**
   * @brief Construct a new Flight service.
   *
   * @param catalog           Catalog used to serve GetTables metadata.
   * @param on_query          Callback for Substrait SELECT plans.
   * @param on_statement      Callback for Substrait write/DDL plans.
   * @param on_physical_plan  Callback for serialized `proto::PhysicalRelation` SELECT plans.
   */
  service(gqe::catalog* catalog,
          execute_query_callback on_query,
          execute_statement_callback on_statement,
          execute_physical_plan_callback on_physical_plan,
          set_option_callback on_set_option,
          get_option_callback on_get_option);

  /** @name Flight RPCs */
  /** @{ */

  [[nodiscard]] arrow::Status GetFlightInfo(
    arrow::flight::ServerCallContext const& context,
    arrow::flight::FlightDescriptor const& descriptor,
    std::unique_ptr<arrow::flight::FlightInfo>* info) override;

  [[nodiscard]] arrow::Status DoGet(
    arrow::flight::ServerCallContext const& context,
    arrow::flight::Ticket const& request,
    std::unique_ptr<arrow::flight::FlightDataStream>* stream) override;

  [[nodiscard]] arrow::Status DoPut(
    arrow::flight::ServerCallContext const& context,
    std::unique_ptr<arrow::flight::FlightMessageReader> reader,
    std::unique_ptr<arrow::flight::FlightMetadataWriter> writer) override;

  [[nodiscard]] arrow::Status DoAction(
    arrow::flight::ServerCallContext const& context,
    arrow::flight::Action const& action,
    std::unique_ptr<arrow::flight::ResultStream>* result) override;

  [[nodiscard]] arrow::Result<arrow::flight::SetSessionOptionsResult> SetSessionOptions(
    arrow::flight::ServerCallContext const& context,
    arrow::flight::SetSessionOptionsRequest const& request);

  [[nodiscard]] arrow::Result<arrow::flight::GetSessionOptionsResult> GetSessionOptions(
    arrow::flight::ServerCallContext const& context,
    arrow::flight::GetSessionOptionsRequest const& request);

  /** @} */

 private:
  /** Build a FlightInfo wrapping a ticket for the cached @p batch. */
  [[nodiscard]] arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> make_flight_info(
    std::shared_ptr<arrow::RecordBatch> batch, arrow::flight::FlightDescriptor const& descriptor);

  /** Execute a Substrait SELECT plan and return a FlightInfo ticket for the result. */
  [[nodiscard]] arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> execute_substrait_select(
    std::string const& plan_bytes, arrow::flight::FlightDescriptor const& descriptor);

  /** Execute a physical-plan SELECT and return a FlightInfo ticket for the result. */
  [[nodiscard]] arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>>
  execute_physical_plan_select(std::string const& plan_bytes,
                               arrow::flight::FlightDescriptor const& descriptor);

  /** Build the GetTables response RecordBatch and return a FlightInfo ticket. */
  [[nodiscard]] arrow::Result<std::unique_ptr<arrow::flight::FlightInfo>> build_get_tables_info(
    arrow::flight::FlightDescriptor const& descriptor);

  /** Execute a Substrait write/DDL plan. Returns rows affected, or -1 for DDL. */
  [[nodiscard]] arrow::Result<int64_t> execute_substrait_write(std::string const& plan_bytes);

  gqe::catalog* _catalog;
  execute_query_callback _on_query;
  execute_statement_callback _on_statement;
  execute_physical_plan_callback _on_physical_plan;
  set_option_callback _on_set_option;
  get_option_callback _on_get_option;

  /**
   * @brief Protects @c _results.
   *
   * Query results are inserted by @c execute_substrait_select(),
   * @c execute_physical_plan_select() and @c build_get_tables_info(), then
   * consumed and erased by @c DoGet(). Multiple Flight handler threads may
   * access the map concurrently.
   *
   * FIXME: The _results currently are only deallocated when retrieved by a client. This is a
   * potentially unbounded memory leak if clients never retrieve results (e.g., due to client
   * failures). The leak could be addressed by adding a TTL (time to live) field, and use the TTL to
   * expire and deallocate results.
   */
  std::mutex _results_mutex;
  std::unordered_map<utility::uuid, std::shared_ptr<arrow::RecordBatch>>
    _results;  ///< Guarded by @c _results_mutex.
};

}  // namespace gqe::node_manager
