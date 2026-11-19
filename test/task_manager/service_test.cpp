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

#include <gqe/task_manager/service.hpp>

#include <gqe/rpc/serialization/uuid.hpp>
#include <gqe/storage/shared_memory.hpp>
#include <gqe/task_manager/owner_thread.hpp>
#include <gqe/task_manager_context.hpp>
#include <gqe/utility/uuid.hpp>

#include <gtest/gtest.h>

#include <boost/interprocess/exceptions.hpp>
#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using gqe::task_manager::owner_thread;
using gqe::task_manager::service;
using gqe::task_manager::task_manager_state;

constexpr int test_rank = 3;

/// Deadline for each RPC and wait, so that a regression that blocks fails instead of hanging.
constexpr auto test_timeout = std::chrono::seconds(30);

/** @brief Sets a deadline of `test_timeout` on a client context. */
void set_test_deadline(grpc::ClientContext& ctx)
{
  ctx.set_deadline(std::chrono::system_clock::now() + test_timeout);
}

/**
 * @brief Runs a task-manager service with stub callbacks on an in-process gRPC server.
 *
 * No callback does GPU work, so these tests need no GPU and no NVSHMEM.
 */
class TaskManagerServiceTest : public ::testing::Test {
 protected:
  TaskManagerServiceTest()
    : svc(owner, [](auto&&...) -> service::execution_result {
        throw std::logic_error("ExecutePlan is not used by these tests");
      })
  {
    svc.on_init_nvshmem([this](int, int, std::string_view) -> service::init_nvshmem_result {
      record_callback_thread();
      if (is_init_failing) { throw std::runtime_error("injected InitNvshmem failure"); }
      return {nullptr, "localhost:1"};
    });
    svc.on_set_peer_info([this](std::vector<gqe::peer_info>) {
      record_callback_thread();
      if (is_set_peer_info_failing) { throw std::runtime_error("injected SetPeerInfo failure"); }
    });
    svc.on_refresh_stats([this](auto&&...) {
      record_callback_thread();
      refresh_started.set_value();
      refresh_release.wait();
    });
    svc.on_fail_stop([this](std::string_view reason) {
      std::scoped_lock guard(mutex);
      fail_stop_reasons.emplace_back(reason);
    });
  }

  void SetUp() override
  {
    grpc::ServerBuilder builder;
    builder.RegisterService(&svc);
    server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    stub = gqe::proto::TaskManagerService::NewStub(server->InProcessChannel({}));
  }

  void TearDown() override
  {
    release_refreshes();  // `Shutdown` waits for running handlers.
    server->Shutdown();
    owner.stop();
  }

  void record_callback_thread()
  {
    std::scoped_lock guard(mutex);
    callback_threads.push_back(std::this_thread::get_id());
  }

  grpc::Status init_nvshmem(gqe::proto::InitNvshmemResponse& response)
  {
    grpc::ClientContext ctx;
    set_test_deadline(ctx);
    gqe::proto::InitNvshmemRequest request;
    request.set_rank(test_rank);
    request.set_num_ranks(test_rank + 1);
    return stub->InitNvshmem(&ctx, request, &response);
  }

  grpc::Status set_peer_info(gqe::proto::SetPeerInfoResponse& response)
  {
    grpc::ClientContext ctx;
    set_test_deadline(ctx);
    gqe::proto::SetPeerInfoRequest request;
    return stub->SetPeerInfo(&ctx, request, &response);
  }

  grpc::Status refresh_stats(gqe::proto::RefreshStatsResponse& response)
  {
    grpc::ClientContext ctx;
    set_test_deadline(ctx);
    gqe::proto::RefreshStatsRequest request;
    *request.mutable_query_id() = gqe::rpc::serialize_uuid(gqe::utility::uuid::generate());
    return stub->RefreshStatsTables(&ctx, request, &response);
  }

  gqe::proto::HealthCheckResponse health_check()
  {
    grpc::ClientContext ctx;
    set_test_deadline(ctx);
    gqe::proto::HealthCheckRequest request;
    gqe::proto::HealthCheckResponse response;
    EXPECT_TRUE(stub->HealthCheck(&ctx, request, &response).ok());
    return response;
  }

  /** @brief Bootstrap the service to `ready`. */
  void bootstrap()
  {
    gqe::proto::InitNvshmemResponse init_response;
    ASSERT_TRUE(init_nvshmem(init_response).ok());
    ASSERT_TRUE(init_response.success());
    gqe::proto::SetPeerInfoResponse peer_response;
    ASSERT_TRUE(set_peer_info(peer_response).ok());
    ASSERT_TRUE(peer_response.success());
    ASSERT_EQ(svc.state(), task_manager_state::ready);
  }

  /** @brief Let refresh callbacks return immediately. Can be called more than once. */
  void release_refreshes()
  {
    std::call_once(release_once, [this] { release_promise.set_value(); });
  }

  owner_thread owner;
  service svc;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<gqe::proto::TaskManagerService::Stub> stub;

  bool is_init_failing          = false;
  bool is_set_peer_info_failing = false;

  std::promise<void> refresh_started;
  std::promise<void> release_promise;
  std::once_flag release_once;
  std::shared_future<void> refresh_release = release_promise.get_future().share();

  std::mutex mutex;
  std::vector<std::thread::id> callback_threads;  ///< Protected by `mutex`.
  std::vector<std::string> fail_stop_reasons;     ///< Protected by `mutex`.
};

}  // namespace

/**
 * @brief Test the state transitions of a successful bootstrap, and that every callback runs on the
 * owner thread.
 */
TEST_F(TaskManagerServiceTest, BootstrapRunsCallbacksOnOwnerThread)
{
  EXPECT_EQ(svc.state(), task_manager_state::starting);
  EXPECT_FALSE(health_check().healthy());

  gqe::proto::InitNvshmemResponse init_response;
  ASSERT_TRUE(init_nvshmem(init_response).ok());
  EXPECT_TRUE(init_response.success());
  EXPECT_EQ(init_response.grpc_address(), "localhost:1");
  EXPECT_EQ(svc.state(), task_manager_state::nvshmem_ready);
  auto health = health_check();
  EXPECT_TRUE(health.healthy());
  EXPECT_EQ(health.rank(), test_rank);

  gqe::proto::SetPeerInfoResponse peer_response;
  ASSERT_TRUE(set_peer_info(peer_response).ok());
  EXPECT_TRUE(peer_response.success());
  EXPECT_EQ(svc.state(), task_manager_state::ready);

  release_refreshes();
  gqe::proto::RefreshStatsResponse refresh_response;
  ASSERT_TRUE(refresh_stats(refresh_response).ok());
  EXPECT_TRUE(refresh_response.success());
  EXPECT_EQ(refresh_response.rank(), test_rank);

  auto const owner_id = [&] {
    std::thread::id id;
    owner.submit([&id] { id = std::this_thread::get_id(); }).get();
    return id;
  }();
  std::scoped_lock guard(mutex);
  ASSERT_EQ(callback_threads.size(), 3);
  for (auto const& id : callback_threads) {
    EXPECT_EQ(id, owner_id);
  }
  EXPECT_TRUE(fail_stop_reasons.empty());
}

/**
 * @brief Test that RPCs sent in the wrong state are rejected without running their callbacks.
 */
TEST_F(TaskManagerServiceTest, RejectsRequestsInWrongState)
{
  gqe::proto::RefreshStatsResponse refresh_response;
  EXPECT_EQ(refresh_stats(refresh_response).error_code(), grpc::StatusCode::UNAVAILABLE);

  gqe::proto::SetPeerInfoResponse peer_response;
  EXPECT_EQ(set_peer_info(peer_response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);

  bootstrap();

  gqe::proto::InitNvshmemResponse init_response;
  EXPECT_EQ(init_nvshmem(init_response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_EQ(set_peer_info(peer_response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);

  grpc::ClientContext ctx;
  set_test_deadline(ctx);
  gqe::proto::GenerateUidRequest uid_request;
  gqe::proto::GenerateUidResponse uid_response;
  EXPECT_EQ(stub->GenerateNvshmemUid(&ctx, uid_request, &uid_response).error_code(),
            grpc::StatusCode::FAILED_PRECONDITION);

  std::scoped_lock guard(mutex);
  EXPECT_EQ(callback_threads.size(), 2);  // Only the bootstrap callbacks ran.
}

/**
 * @brief Test that a second operation is rejected while one is running, that `HealthCheck` still
 * answers, and that the task manager accepts operations again afterwards.
 */
TEST_F(TaskManagerServiceTest, RejectsOperationWhileBusy)
{
  bootstrap();

  gqe::proto::RefreshStatsResponse first_response;
  auto first = std::async(std::launch::async, [&] { return refresh_stats(first_response); });
  // Release the first callback on every exit path, before `first` waits for it in its destructor.
  // Otherwise a failed assertion below would hang the test.
  struct release_guard {
    std::function<void()> release;
    ~release_guard() { release(); }
  } guard{[this] { release_refreshes(); }};
  ASSERT_EQ(refresh_started.get_future().wait_for(test_timeout), std::future_status::ready)
    << "The first RefreshStatsTables callback did not start";

  gqe::proto::RefreshStatsResponse second_response;
  auto const second = refresh_stats(second_response);
  EXPECT_EQ(second.error_code(), grpc::StatusCode::FAILED_PRECONDITION);
  EXPECT_NE(second.error_message().find("still running another operation"), std::string::npos);
  EXPECT_TRUE(health_check().healthy());

  release_refreshes();
  auto const first_status = first.get();
  ASSERT_TRUE(first_status.ok());
  EXPECT_TRUE(first_response.success());

  // `refresh_started` is satisfied, so later refresh callbacks must not set it again.
  svc.on_refresh_stats([](auto&&...) {});
  gqe::proto::RefreshStatsResponse third_response;
  ASSERT_TRUE(refresh_stats(third_response).ok());
  EXPECT_TRUE(third_response.success());
}

/**
 * @brief Test that a failed `InitNvshmem` answers with the error, then fails stop.
 */
TEST_F(TaskManagerServiceTest, FailedInitNvshmemFailsStop)
{
  is_init_failing = true;
  gqe::proto::InitNvshmemResponse response;
  ASSERT_TRUE(init_nvshmem(response).ok());
  EXPECT_FALSE(response.success());
  EXPECT_NE(response.error_message().find("injected InitNvshmem failure"), std::string::npos);

  EXPECT_EQ(svc.state(), task_manager_state::stopping);
  EXPECT_FALSE(health_check().healthy());
  {
    std::scoped_lock guard(mutex);
    ASSERT_EQ(fail_stop_reasons.size(), 1);
    EXPECT_NE(fail_stop_reasons.front().find("injected InitNvshmem failure"), std::string::npos);
  }

  // A retry is rejected: NVSHMEM cannot be initialized again in this process.
  is_init_failing = false;
  EXPECT_EQ(init_nvshmem(response).error_code(), grpc::StatusCode::FAILED_PRECONDITION);
}

/**
 * @brief Test that a failed `SetPeerInfo` fails stop.
 */
TEST_F(TaskManagerServiceTest, FailedSetPeerInfoFailsStop)
{
  gqe::proto::InitNvshmemResponse init_response;
  ASSERT_TRUE(init_nvshmem(init_response).ok());

  is_set_peer_info_failing = true;
  gqe::proto::SetPeerInfoResponse response;
  ASSERT_TRUE(set_peer_info(response).ok());
  EXPECT_FALSE(response.success());
  // The reason reaches the node manager in the response, so that it can report why bootstrap
  // failed.
  EXPECT_NE(response.error_message().find("injected SetPeerInfo failure"), std::string::npos);

  EXPECT_EQ(svc.state(), task_manager_state::stopping);
  std::scoped_lock guard(mutex);
  ASSERT_EQ(fail_stop_reasons.size(), 1);
  EXPECT_NE(fail_stop_reasons.front().find("injected SetPeerInfo failure"), std::string::npos);
}

/**
 * @brief Test that an operation is answered with `UNAVAILABLE` once the owner thread has stopped.
 */
TEST_F(TaskManagerServiceTest, RejectsOperationAfterOwnerStops)
{
  bootstrap();
  owner.stop();

  gqe::proto::RefreshStatsResponse response;
  EXPECT_EQ(refresh_stats(response).error_code(), grpc::StatusCode::UNAVAILABLE);
}

// ----- Unrecoverable shared memory -----

namespace {

/**
 * @brief Runs operations on a bootstrapped service, to test that their responses report an
 * unrecoverable shared memory as `node_fatal`, taking precedence over success and over the
 * operation's own error.
 */
class TaskManagerServicePoisoningTest : public TaskManagerServiceTest {
 protected:
  // Poisoning is sticky for the process, and only a build with the CMake option
  // `GQE_ENABLE_FAULT_INJECTION` can clear it between tests, so without the option these tests
  // skip.
  void SetUp() override
  {
#ifndef GQE_ENABLE_FAULT_INJECTION
    GTEST_SKIP() << "Needs a build configured with -DGQE_ENABLE_FAULT_INJECTION=ON";
#else
    gqe::detail::reset_shared_memory_poisoned_for_testing();
    TaskManagerServiceTest::SetUp();
    bootstrap();
    release_refreshes();
    svc.on_refresh_stats([this](auto&&...) { refresh_behavior(); });
#endif
  }

  void TearDown() override
  {
#ifdef GQE_ENABLE_FAULT_INJECTION
    gqe::detail::reset_shared_memory_poisoned_for_testing();
    TaskManagerServiceTest::TearDown();
#endif
  }

  /// Call RefreshStatsTables with an empty, well-formed request.
  gqe::proto::RefreshStatsResponse call_refresh_stats()
  {
    gqe::proto::RefreshStatsResponse response;
    EXPECT_TRUE(refresh_stats(response).ok());
    return response;
  }

  /// Behavior of the stubbed refresh-stats operation.
  std::function<void()> refresh_behavior = [] {};
};

}  // namespace

TEST_F(TaskManagerServicePoisoningTest, HealthySuccessIsNotNodeFatal)
{
  auto const response = call_refresh_stats();
  EXPECT_TRUE(response.success());
  EXPECT_FALSE(response.node_fatal());
}

TEST_F(TaskManagerServicePoisoningTest, PoisoningDuringSuccessfulOperationIsNodeFatal)
{
  // Models a noexcept cleanup that swallowed a lock failure.
  refresh_behavior = [] { gqe::mark_shared_memory_poisoned("cleanup in test"); };

  auto const response = call_refresh_stats();
  EXPECT_FALSE(response.success());
  EXPECT_TRUE(response.node_fatal());
  EXPECT_NE(response.error_message().find("cleanup in test"), std::string::npos);
}

TEST_F(TaskManagerServicePoisoningTest, LockExceptionIsNodeFatal)
{
  refresh_behavior = [] {
    throw boost::interprocess::lock_exception(boost::interprocess::not_recoverable);
  };

  auto const response = call_refresh_stats();
  EXPECT_FALSE(response.success());
  EXPECT_TRUE(response.node_fatal());
  EXPECT_TRUE(gqe::is_shared_memory_poisoned());
}

TEST_F(TaskManagerServicePoisoningTest, PoisoningTakesPrecedenceOverOperationError)
{
  // Cleanup while the operation was already failing poisoned the segment.
  refresh_behavior = [] {
    gqe::mark_shared_memory_poisoned("cleanup in test");
    throw std::runtime_error("original failure");
  };

  auto const response = call_refresh_stats();
  EXPECT_FALSE(response.success());
  EXPECT_TRUE(response.node_fatal());
  EXPECT_NE(response.error_message().find("cleanup in test"), std::string::npos);
  EXPECT_NE(response.error_message().find("original failure"), std::string::npos);
}

TEST_F(TaskManagerServicePoisoningTest, OrdinaryErrorIsNotNodeFatal)
{
  refresh_behavior = [] { throw std::runtime_error("original failure"); };

  auto const response = call_refresh_stats();
  EXPECT_FALSE(response.success());
  EXPECT_FALSE(response.node_fatal());
  EXPECT_EQ(response.error_message(), "original failure");
}

TEST_F(TaskManagerServicePoisoningTest, ExecutePlanReportsNodeFatalInsteadOfFatal)
{
  gqe::mark_shared_memory_poisoned("earlier operation");

  grpc::ClientContext ctx;
  set_test_deadline(ctx);
  gqe::proto::ExecutePlanRequest request;
  *request.mutable_query_id() = gqe::rpc::serialize_uuid(gqe::utility::uuid::generate());
  gqe::proto::ExecutePlanResponse response;
  ASSERT_TRUE(stub->ExecutePlan(&ctx, request, &response).ok());
  EXPECT_FALSE(response.success());
  EXPECT_TRUE(response.node_fatal());
  EXPECT_FALSE(response.rank_fatal());
  EXPECT_TRUE(response.arrow_ipc_result().empty());
  EXPECT_NE(response.error_message().find("earlier operation"), std::string::npos);
}
