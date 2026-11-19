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

#include <gqe/node_manager/nvshmem_bootstrap.hpp>

#include <gtest/gtest.h>

#include <grpcpp/grpcpp.h>
#include <proto/node_task_manager.grpc.pb.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace proto = gqe::proto;

/**
 * @brief A task manager that answers `InitNvshmem` in one fixed way, without a GPU.
 *
 * A `block` rank behaves like a rank that waits in the NVSHMEM collective for a rank that will
 * never join: it does not answer until its RPC is cancelled or the test releases it. A `fail` rank
 * answers only after every `block` rank has entered its handler, so that the cancellation always
 * reaches a handler that is running.
 */
class fake_task_manager final : public proto::TaskManagerService::Service {
 public:
  enum class behavior { succeed, fail, block };

  /**
   * @param how How this rank answers.
   * @param blockers_entered Counts the `block` ranks that have entered their handler, shared by the
   * ranks of one cluster.
   * @param num_blockers The number of `block` ranks in the cluster.
   */
  fake_task_manager(behavior how, std::atomic<int>& blockers_entered, int num_blockers)
    : _behavior(how), _blockers_entered(blockers_entered), _num_blockers(num_blockers)
  {
  }

  grpc::Status InitNvshmem(grpc::ServerContext* context,
                           proto::InitNvshmemRequest const* request,
                           proto::InitNvshmemResponse* response) override
  {
    switch (_behavior) {
      case behavior::succeed:
        response->set_success(true);
        response->set_pgas_base_ptr(0x1000 + static_cast<std::uint64_t>(request->rank()));
        response->set_grpc_address(std::format("rank-{}:1", request->rank()));
        break;
      case behavior::fail: {
        // Bounded, so that a rank that never enters cannot hang the test.
        auto const give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (_blockers_entered.load() < _num_blockers &&
               std::chrono::steady_clock::now() < give_up) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        response->set_success(false);
        response->set_error_message(std::format("injected failure on rank {}", request->rank()));
        break;
      }
      case behavior::block: {
        _blockers_entered.fetch_add(1);
        // The test releases the rank in the end, so that a regression cannot hang the server.
        while (!context->IsCancelled() && !_is_released.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        _was_cancelled.store(context->IsCancelled());
        response->set_success(false);
        response->set_error_message("blocked rank was released");
        break;
      }
    }
    return grpc::Status::OK;
  }

  /** @brief Let a blocked rank answer. */
  void release() { _is_released.store(true); }

  /** @brief Whether the client cancelled the RPC of a blocked rank. */
  [[nodiscard]] bool was_cancelled() const { return _was_cancelled.load(); }

 private:
  behavior _behavior;
  std::atomic<int>& _blockers_entered;
  int _num_blockers;
  std::atomic<bool> _is_released{false};
  std::atomic<bool> _was_cancelled{false};
};

/** @brief In-process task managers, one per rank, and the channels to them. */
class fake_cluster {
 public:
  using behavior = fake_task_manager::behavior;

  explicit fake_cluster(std::vector<behavior> const& behaviors)
  {
    auto const num_blockers = static_cast<int>(std::ranges::count(behaviors, behavior::block));
    for (auto const how : behaviors) {
      auto& manager = _managers.emplace_back(
        std::make_unique<fake_task_manager>(how, _blockers_entered, num_blockers));
      grpc::ServerBuilder builder;
      builder.RegisterService(manager.get());
      auto& server = _servers.emplace_back(builder.BuildAndStart());
      _channels.push_back(server->InProcessChannel({}));
    }
  }

  ~fake_cluster()
  {
    for (auto& manager : _managers) {
      manager->release();  // `Shutdown` waits for running handlers.
    }
    for (auto& server : _servers) {
      server->Shutdown();
    }
  }

  fake_cluster(fake_cluster const&)            = delete;
  fake_cluster& operator=(fake_cluster const&) = delete;

  [[nodiscard]] std::vector<std::shared_ptr<grpc::Channel>> const& channels() const
  {
    return _channels;
  }

  [[nodiscard]] fake_task_manager& manager(std::size_t rank) { return *_managers[rank]; }

 private:
  std::atomic<int> _blockers_entered{0};  // Declared first: the managers refer to it.
  std::vector<std::unique_ptr<fake_task_manager>> _managers;
  std::vector<std::unique_ptr<grpc::Server>> _servers;
  std::vector<std::shared_ptr<grpc::Channel>> _channels;
};

/// A rank is released after this long even if the test is waiting for it, so that a regression
/// fails its time bound instead of waiting for the 180 s RPC deadline.
constexpr auto safety_release_delay = std::chrono::seconds(20);

/// How long a fail-fast bootstrap may take. A regression waits for the blocked rank.
constexpr auto fail_fast_bound = std::chrono::seconds(10);

/** @brief Release every blocked rank of @p cluster after `safety_release_delay`, unless stopped. */
class safety_release {
 public:
  explicit safety_release(fake_cluster& cluster, std::size_t ranks)
    : _thread([this, &cluster, ranks](std::stop_token token) {
        std::unique_lock lock(_mutex);
        if (_cv.wait_for(lock, token, safety_release_delay, [] { return false; })) { return; }
        for (std::size_t rank = 0; rank < ranks; ++rank) {
          cluster.manager(rank).release();
        }
      })
  {
  }

 private:
  std::mutex _mutex;
  std::condition_variable_any _cv;
  std::jthread _thread;
};

}  // namespace

TEST(InitNvshmemTest, CollectsTheResultOfEveryRank)
{
  fake_cluster cluster({fake_cluster::behavior::succeed, fake_cluster::behavior::succeed});

  auto const peers = gqe::node_manager::init_nvshmem(cluster.channels(), "uid");

  ASSERT_EQ(peers.size(), 2);
  for (int rank = 0; rank < 2; ++rank) {
    EXPECT_EQ(peers[rank].rank, rank);
    EXPECT_EQ(peers[rank].pgas_base_ptr, 0x1000 + static_cast<std::uint64_t>(rank));
    EXPECT_EQ(peers[rank].grpc_address, std::format("rank-{}:1", rank));
  }
}

TEST(InitNvshmemTest, AFailingRankFailsTheBootstrapAtOnceAndCancelsTheOthers)
{
  // Rank 0 waits for rank 1, which fails. Rank 0 comes first, so a bootstrap that waits for the
  // ranks in order would wait for rank 0 before it noticed that rank 1 had failed.
  fake_cluster cluster({fake_cluster::behavior::block, fake_cluster::behavior::fail});
  safety_release release(cluster, 1);

  auto const start = std::chrono::steady_clock::now();
  try {
    std::ignore = gqe::node_manager::init_nvshmem(cluster.channels(), "uid");
    FAIL() << "The bootstrap succeeded although rank 1 failed";
  } catch (std::runtime_error const& error) {
    // The error names the rank that failed, not the cancellation that it caused.
    EXPECT_NE(std::string(error.what()).find("rank 1"), std::string::npos) << error.what();
    EXPECT_NE(std::string(error.what()).find("injected failure on rank 1"), std::string::npos)
      << error.what();
  }
  EXPECT_LT(std::chrono::steady_clock::now() - start, fail_fast_bound)
    << "The bootstrap waited for the blocked rank instead of failing at once";

  // The blocked rank's RPC was cancelled, so it is not left waiting for its own deadline.
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!cluster.manager(0).was_cancelled() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_TRUE(cluster.manager(0).was_cancelled()) << "The blocked rank's RPC was not cancelled";
}

TEST(InitNvshmemTest, ReportsTheFirstFailureNotTheCancellationsItCauses)
{
  // Rank 0 fails, and rank 1 is the blocked one that gets cancelled.
  fake_cluster cluster({fake_cluster::behavior::fail, fake_cluster::behavior::block});
  safety_release release(cluster, 2);

  try {
    std::ignore = gqe::node_manager::init_nvshmem(cluster.channels(), "uid");
    FAIL() << "The bootstrap succeeded although rank 0 failed";
  } catch (std::runtime_error const& error) {
    EXPECT_NE(std::string(error.what()).find("injected failure on rank 0"), std::string::npos)
      << error.what();
    EXPECT_EQ(std::string(error.what()).find("CANCELLED"), std::string::npos) << error.what();
    EXPECT_EQ(std::string(error.what()).find("Cancel"), std::string::npos) << error.what();
  }
}

TEST(InitNvshmemTest, AnUnreachableRankFailsTheBootstrapAndCancelsTheOthers)
{
  fake_cluster cluster({fake_cluster::behavior::block, fake_cluster::behavior::succeed});
  safety_release release(cluster, 1);
  // A channel to nothing: rank 1 cannot be reached.
  auto channels = cluster.channels();
  channels[1]   = grpc::CreateChannel("127.0.0.1:1", grpc::InsecureChannelCredentials());

  auto const start = std::chrono::steady_clock::now();
  EXPECT_THROW(std::ignore = gqe::node_manager::init_nvshmem(channels, "uid"), std::runtime_error);
  EXPECT_LT(std::chrono::steady_clock::now() - start, fail_fast_bound);
}
