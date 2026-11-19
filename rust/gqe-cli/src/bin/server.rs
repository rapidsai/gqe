// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use clap::Parser;

/// Default address the Flight SQL server binds to. Loopback by default so the
/// server is not network-exposed unless an operator opts in via --addr.
const DEFAULT_LISTEN_ADDR: &str = "127.0.0.1:50051";

#[derive(Parser)]
#[command(about = "Flight SQL + Substrait server")]
struct Args {
    /// Address to listen on
    #[arg(long, default_value = DEFAULT_LISTEN_ADDR)]
    addr: std::net::SocketAddr,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();
    gqe_cli::server::serve(args.addr).await
}
