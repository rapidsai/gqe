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

// Tests for the fault boundary between the node manager and the task managers. The task-manager
// service's own poisoning tests cover respond_to_operation through real RPCs.

#include <gqe/node_manager/fault_boundary.hpp>

#include <gqe/storage/shared_memory_poison.hpp>

#include <gtest/gtest.h>

#include <boost/interprocess/exceptions.hpp>
#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <arrow/flight/server.h>
#include <arrow/flight/server_middleware.h>

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <tuple>

namespace {

using gqe::node_manager::rank_outcome;

/** @brief Reset the poisoned flag around each test. */
class FaultBoundaryTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
#ifdef GQE_ENABLE_FAULT_INJECTION
    gqe::detail::reset_shared_memory_poisoned_for_testing();
#endif
  }
  void TearDown() override
  {
#ifdef GQE_ENABLE_FAULT_INJECTION
    gqe::detail::reset_shared_memory_poisoned_for_testing();
#endif
  }
};

/**
 * @brief A fixture for the tests that poison the segment.
 *
 * Poisoning is sticky for the process, and only a build with the CMake option
 * `GQE_ENABLE_FAULT_INJECTION` can clear it between tests, so without the option these tests skip.
 */
class FaultBoundaryPoisoningTest : public FaultBoundaryTest {
 protected:
  void SetUp() override
  {
#ifndef GQE_ENABLE_FAULT_INJECTION
    GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#endif
    FaultBoundaryTest::SetUp();
  }
};

// ----- respond_to_operation -----

TEST_F(FaultBoundaryTest, RespondToOperationReportsSuccess)
{
  gqe::proto::ExecutePlanResponse response;
  gqe::node_manager::respond_to_operation(
    response, 3, "test", [&] { response.set_arrow_ipc_result("result"); });
  EXPECT_TRUE(response.success());
  EXPECT_EQ(response.rank(), 3);
  EXPECT_EQ(response.arrow_ipc_result(), "result");
  EXPECT_FALSE(response.rank_fatal());
  EXPECT_FALSE(response.node_fatal());
}

TEST_F(FaultBoundaryTest, RespondToOperationReportsError)
{
  gqe::proto::ExecutePlanResponse response;
  gqe::node_manager::respond_to_operation(
    response, 0, "test", [] { throw std::runtime_error("original failure"); });
  EXPECT_FALSE(response.success());
  EXPECT_EQ(response.error_message(), "original failure");
  EXPECT_FALSE(response.rank_fatal());
  EXPECT_FALSE(response.node_fatal());
}

TEST_F(FaultBoundaryPoisoningTest, RespondToOperationSkipsOperationOnPoisonedSegment)
{
  gqe::mark_shared_memory_poisoned("earlier operation");
  bool ran = false;
  gqe::proto::RefreshStatsResponse response;
  gqe::node_manager::respond_to_operation(response, 0, "test", [&] { ran = true; });
  EXPECT_FALSE(ran);
  EXPECT_FALSE(response.success());
  EXPECT_TRUE(response.node_fatal());
  EXPECT_NE(response.error_message().find("earlier operation"), std::string::npos);
}

TEST_F(FaultBoundaryPoisoningTest, RespondToOperationDiscardsResultsOfPoisonedSuccess)
{
  gqe::proto::ExecutePlanResponse response;
  gqe::node_manager::respond_to_operation(response, 0, "test", [&] {
    response.set_arrow_ipc_result("result");
    gqe::mark_shared_memory_poisoned("cleanup in test");
  });
  EXPECT_FALSE(response.success());
  EXPECT_TRUE(response.node_fatal());
  EXPECT_TRUE(response.arrow_ipc_result().empty());
}

TEST_F(FaultBoundaryPoisoningTest, RespondToOperationRecordsLockException)
{
  gqe::proto::RefreshStatsResponse response;
  gqe::node_manager::respond_to_operation(response, 0, "test", [] {
    throw boost::interprocess::lock_exception(boost::interprocess::not_recoverable);
  });
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
  EXPECT_TRUE(response.node_fatal());
}

// ----- task_manager_client -----

/** @brief A task manager that answers every operation with a scripted status and response. */
class scripted_task_manager final : public gqe::proto::TaskManagerService::Service {
 public:
  grpc::Status status = grpc::Status::OK;
  gqe::proto::ExecutePlanResponse execute_response;
  gqe::proto::RefreshStatsResponse refresh_response;

  grpc::Status ExecutePlan(grpc::ServerContext*,
                           gqe::proto::ExecutePlanRequest const*,
                           gqe::proto::ExecutePlanResponse* response) override
  {
    *response = execute_response;
    return status;
  }

  grpc::Status RefreshStatsTables(grpc::ServerContext*,
                                  gqe::proto::RefreshStatsRequest const*,
                                  gqe::proto::RefreshStatsResponse* response) override
  {
    *response = refresh_response;
    return status;
  }
};

class TaskManagerClientTest : public FaultBoundaryTest {
 protected:
  void SetUp() override
  {
    FaultBoundaryTest::SetUp();
    grpc::ServerBuilder builder;
    builder.RegisterService(&task_manager);
    server = builder.BuildAndStart();
    client =
      std::make_unique<gqe::node_manager::task_manager_client>(server->InProcessChannel({}), 2);
  }

  void TearDown() override
  {
    server->Shutdown();
    FaultBoundaryTest::TearDown();
  }

  gqe::node_manager::rank_result<gqe::proto::ExecutePlanResponse> execute()
  {
    grpc::ClientContext context;
    return client->execute_plan(context, gqe::proto::ExecutePlanRequest{});
  }

  scripted_task_manager task_manager;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<gqe::node_manager::task_manager_client> client;
};

TEST_F(TaskManagerClientTest, ClassifiesSuccess)
{
  task_manager.execute_response.set_success(true);
  task_manager.execute_response.set_arrow_ipc_result("result");
  auto const result = execute();
  EXPECT_EQ(result.outcome, rank_outcome::success);
  EXPECT_TRUE(result.is_confirmed());
  EXPECT_TRUE(result.message.empty());
  EXPECT_EQ(result.response.arrow_ipc_result(), "result");
}

TEST_F(TaskManagerClientTest, ClassifiesError)
{
  task_manager.execute_response.set_error_message("original failure");
  auto const result = execute();
  EXPECT_EQ(result.outcome, rank_outcome::error);
  EXPECT_TRUE(result.is_confirmed());
  EXPECT_EQ(result.message, "Task manager rank 2 failed: original failure");
}

TEST_F(TaskManagerClientTest, ClassifiesRankFatal)
{
  task_manager.execute_response.set_rank_fatal(true);
  EXPECT_EQ(execute().outcome, rank_outcome::rank_fatal);
}

TEST_F(TaskManagerClientTest, NodeFatalTakesPrecedenceOverRankFatal)
{
  task_manager.execute_response.set_rank_fatal(true);
  task_manager.execute_response.set_node_fatal(true);
  EXPECT_EQ(execute().outcome, rank_outcome::node_fatal);
}

TEST_F(TaskManagerClientTest, ClassifiesTransportFailureAsUnconfirmed)
{
  task_manager.status = grpc::Status(grpc::StatusCode::UNAVAILABLE, "busy");
  auto const result   = execute();
  EXPECT_EQ(result.outcome, rank_outcome::transport_failure);
  EXPECT_FALSE(result.is_confirmed());
  EXPECT_EQ(result.message, "Task manager rank 2 failed: busy");
}

TEST_F(TaskManagerClientTest, ClassifiesStatsRefreshNodeFatal)
{
  task_manager.refresh_response.set_node_fatal(true);
  task_manager.refresh_response.set_error_message("unrecoverable");
  grpc::ClientContext context;
  auto const result = client->refresh_stats(context, gqe::proto::RefreshStatsRequest{});
  EXPECT_EQ(result.outcome, rank_outcome::node_fatal);
  EXPECT_EQ(result.message, "Task manager rank 2 stats refresh failed: unrecoverable");
}

TEST_F(TaskManagerClientTest, ClassifiesStatsRefreshRankFatal)
{
  task_manager.refresh_response.set_rank_fatal(true);
  grpc::ClientContext context;
  EXPECT_EQ(client->refresh_stats(context, gqe::proto::RefreshStatsRequest{}).outcome,
            rank_outcome::rank_fatal);
}

// ----- teardown_state -----

TEST_F(FaultBoundaryTest, TeardownKeepsFirstCause)
{
  gqe::node_manager::teardown_state teardown;
  EXPECT_FALSE(teardown.has_begun());
  EXPECT_FALSE(teardown.error().has_value());

  teardown.begin(gqe::node_manager::teardown_reason::shared_memory_unrecoverable, "first");
  teardown.begin(gqe::node_manager::teardown_reason::shared_memory_unrecoverable, "second");
  EXPECT_TRUE(teardown.has_begun());
  EXPECT_EQ(teardown.cause(), "first");
  EXPECT_NE(std::string(teardown.error()->what()).find("first"), std::string::npos);
  EXPECT_NE(std::string(teardown.error()->what())
              .find("shutting down because the shared memory is unrecoverable"),
            std::string::npos);
  EXPECT_THROW(teardown.throw_error(), gqe::shared_memory_unrecoverable_error);
  EXPECT_EQ(teardown.exit_status(), gqe::node_manager::exit_code::shared_memory_unrecoverable);
}

TEST_F(FaultBoundaryTest, TeardownForTaskManagersReportsItsOwnErrorAndExitStatus)
{
  using gqe::node_manager::teardown_reason;
  gqe::node_manager::teardown_state teardown;

  EXPECT_THROW(teardown.begin_and_throw(teardown_reason::task_managers_unavailable, "no bootstrap"),
               gqe::node_manager::task_managers_unavailable_error);
  EXPECT_EQ(teardown.reason(), teardown_reason::task_managers_unavailable);
  EXPECT_EQ(teardown.exit_status(), gqe::node_manager::exit_code::task_managers_unavailable);
  EXPECT_NE(std::string(teardown.error()->what())
              .find("shutting down because the task managers could not be restarted"),
            std::string::npos);
  EXPECT_NE(std::string(teardown.error()->what()).find("no bootstrap"), std::string::npos);

  // Later operations get the same error, not a shared-memory one.
  EXPECT_THROW(teardown.throw_if_begun(), gqe::node_manager::task_managers_unavailable_error);
  EXPECT_THROW(teardown.guard([] { return 7; }),
               gqe::node_manager::task_managers_unavailable_error);
}

TEST_F(FaultBoundaryPoisoningTest, FirstTeardownReasonWinsAcrossReasons)
{
  using gqe::node_manager::teardown_reason;
  gqe::node_manager::teardown_state teardown;

  teardown.begin(teardown_reason::task_managers_unavailable, "first");
  // Poisoning observed afterwards does not change the reason, the exit status or the error type.
  gqe::mark_shared_memory_poisoned("in test");
  EXPECT_THROW(teardown.throw_if_poisoned("node manager"),
               gqe::node_manager::task_managers_unavailable_error);
  EXPECT_EQ(teardown.reason(), teardown_reason::task_managers_unavailable);
  EXPECT_EQ(teardown.cause(), "first");
  EXPECT_EQ(teardown.exit_status(), gqe::node_manager::exit_code::task_managers_unavailable);
}

TEST_F(FaultBoundaryTest, ThrowErrorBeforeTeardownIsALogicError)
{
  gqe::node_manager::teardown_state teardown;
  EXPECT_THROW(teardown.throw_error(), std::logic_error);
  EXPECT_FALSE(teardown.reason().has_value());
}

TEST_F(FaultBoundaryPoisoningTest, ThrowIfBegunBeginsTeardownOnPoisonedSegment)
{
  gqe::node_manager::teardown_state teardown;
  EXPECT_NO_THROW(teardown.throw_if_begun());

  gqe::mark_shared_memory_poisoned("in test");
  EXPECT_THROW(teardown.throw_if_begun(), gqe::shared_memory_unrecoverable_error);
  EXPECT_TRUE(teardown.has_begun());
}

TEST_F(FaultBoundaryTest, GuardPassesThroughResultAndError)
{
  gqe::node_manager::teardown_state teardown;
  EXPECT_EQ(teardown.guard([] { return 7; }), 7);
  EXPECT_THROW(teardown.guard([]() -> int { throw std::invalid_argument("bad"); }),
               std::invalid_argument);
  EXPECT_FALSE(teardown.has_begun());
}

TEST_F(FaultBoundaryPoisoningTest, GuardFailsOperationThatPoisonedTheSegment)
{
  gqe::node_manager::teardown_state teardown;
  EXPECT_THROW(teardown.guard([] {
    gqe::mark_shared_memory_poisoned("cleanup in test");
    return 7;
  }),
               gqe::shared_memory_unrecoverable_error);
  EXPECT_TRUE(teardown.has_begun());
  EXPECT_THROW(teardown.guard([] { return 7; }), gqe::shared_memory_unrecoverable_error);
}

TEST_F(FaultBoundaryPoisoningTest, GuardBeginsTeardownOnLockException)
{
  gqe::node_manager::teardown_state teardown;
  EXPECT_THROW(teardown.guard([]() -> int {
    throw boost::interprocess::lock_exception(boost::interprocess::not_recoverable);
  }),
               gqe::shared_memory_unrecoverable_error);
  EXPECT_TRUE(teardown.has_begun());
}

TEST_F(FaultBoundaryTest, WaitReturnsWhenTeardownBegins)
{
  gqe::node_manager::teardown_state teardown;
  std::stop_source stop;
  auto waiter = std::async(std::launch::async, [&] {
    auto const start = std::chrono::steady_clock::now();
    teardown.wait(stop.get_token(), std::chrono::minutes(1));
    return std::chrono::steady_clock::now() - start;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  teardown.begin(gqe::node_manager::teardown_reason::shared_memory_unrecoverable, "in test");
  EXPECT_LT(waiter.get(), std::chrono::seconds(30));
}

// ----- teardown_gate -----

/** @brief A call context for middleware that does not read it. */
class unused_call_context final : public arrow::flight::ServerCallContext {
 public:
  std::string const& peer_identity() const override { return _empty; }
  std::string const& peer() const override { return _empty; }
  void AddHeader(std::string const&, std::string const&) const override {}
  void AddTrailer(std::string const&, std::string const&) const override {}
  arrow::flight::ServerMiddleware* GetMiddleware(std::string const&) const override
  {
    return nullptr;
  }
  bool is_cancelled() const override { return false; }
  arrow::flight::CallHeaders const& incoming_headers() const override { return _headers; }

 private:
  std::string _empty;
  arrow::flight::CallHeaders _headers;
};

TEST_F(FaultBoundaryTest, GateRejectsCallsOnceTeardownBegins)
{
  gqe::node_manager::teardown_state teardown;
  gqe::node_manager::teardown_gate gate(teardown);
  arrow::flight::CallInfo const info{arrow::flight::FlightMethod::DoAction};
  unused_call_context const context;
  std::shared_ptr<arrow::flight::ServerMiddleware> middleware;

  EXPECT_TRUE(gate.StartCall(info, context, &middleware).ok());

  teardown.begin(gqe::node_manager::teardown_reason::shared_memory_unrecoverable, "in test");
  auto const status = gate.StartCall(info, context, &middleware);
  EXPECT_FALSE(status.ok());
  EXPECT_NE(status.message().find("in test"), std::string::npos);
}

TEST_F(FaultBoundaryTest, GateReportsMissingTaskManagersAsPlainError)
{
  gqe::node_manager::teardown_state teardown;
  gqe::node_manager::teardown_gate gate(teardown);
  arrow::flight::CallInfo const info{arrow::flight::FlightMethod::DoAction};
  unused_call_context const context;
  std::shared_ptr<arrow::flight::ServerMiddleware> middleware;

  teardown.begin(gqe::node_manager::teardown_reason::task_managers_unavailable, "in test");
  auto const status = gate.StartCall(info, context, &middleware);
  ASSERT_FALSE(status.ok());
  // The node exits and does not come back by itself, so the status is not `Unavailable`, which
  // would invite the client to retry against it. The same holds for a shared-memory teardown.
  EXPECT_EQ(arrow::flight::FlightStatusDetail::UnwrapStatus(status), nullptr);

  gqe::node_manager::teardown_state memory_teardown;
  gqe::node_manager::teardown_gate memory_gate(memory_teardown);
  memory_teardown.begin(gqe::node_manager::teardown_reason::shared_memory_unrecoverable, "in test");
  auto const memory_status = memory_gate.StartCall(info, context, &middleware);
  ASSERT_FALSE(memory_status.ok());
  EXPECT_EQ(arrow::flight::FlightStatusDetail::UnwrapStatus(memory_status), nullptr);
}

TEST_F(FaultBoundaryPoisoningTest, GateBeginsTeardownOnPoisonedSegment)
{
  gqe::node_manager::teardown_state teardown;
  gqe::node_manager::teardown_gate gate(teardown);
  arrow::flight::CallInfo const info{arrow::flight::FlightMethod::DoAction};
  unused_call_context const context;
  std::shared_ptr<arrow::flight::ServerMiddleware> middleware;

  gqe::mark_shared_memory_poisoned("in test");
  EXPECT_FALSE(gate.StartCall(info, context, &middleware).ok());
  EXPECT_TRUE(teardown.has_begun());
}

}  // namespace
