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

#include <proto/log_stream.grpc.pb.h>

#include <memory>

namespace gqe::node_manager {

class log_drain;

/**
 * @brief Receive task-manager log streams and forward validated records to a shared log drain.
 *
 * Each stream must begin with one source header followed only by log records. gRPC may execute
 * multiple streams concurrently; the shared @ref log_drain serializes their final output.
 */
class log_stream_service final : public proto::LogStreamService::Service {
 public:
  /**
   * @brief Construct a log-stream service backed by a shared drain.
   *
   * @param drain Destination for records received from every task manager.
   */
  explicit log_stream_service(std::shared_ptr<log_drain> drain);

  /**
   * @brief Consume and validate one task-manager log stream.
   *
   * @param context Context for the incoming gRPC call.
   * @param reader Reader for the header and subsequent log records.
   * @param response Acknowledgment populated with the number of records received.
   * @return `OK` after a valid stream ends, or an error status for malformed or oversized input.
   */
  grpc::Status StreamLogs(grpc::ServerContext* context,
                          grpc::ServerReader<proto::LogStreamMessage>* reader,
                          proto::LogStreamAck* response) override;

 private:
  std::shared_ptr<log_drain> drain_;
};

}  // namespace gqe::node_manager
