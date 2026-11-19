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

use std::io::Read;

use clap::{ArgGroup, Parser};

#[derive(Parser)]
#[command(about = "Flight SQL + Substrait client")]
#[command(group = ArgGroup::new("input").required(true).multiple(false))]
struct Args {
    /// Path to a SQL file to execute (use "-" for stdin)
    #[arg(long, group = "input")]
    sql_file: Option<String>,

    /// Path to a serialized `gqe.proto.PhysicalRelation` file (use "-" for stdin)
    #[arg(long, group = "input")]
    physical_plan: Option<String>,

    /// URL of the Flight SQL server
    #[arg(long, default_value = "http://localhost:50051")]
    server_url: String,

    /// Output results to a Parquet file
    #[arg(long)]
    parquet: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Args::parse();

    match (&args.sql_file, &args.physical_plan) {
        (Some(path), None) => {
            let sql_text = String::from_utf8(read_input(path)?)?;
            gqe_cli::client::run_against(&args.server_url, &sql_text, args.parquet.as_deref()).await
        }
        (None, Some(path)) => {
            let plan_bytes = read_input(path)?;
            gqe_cli::client::run_physical_plan_against(
                &args.server_url,
                plan_bytes,
                args.parquet.as_deref(),
            )
            .await
        }
        _ => unreachable!(
            "clap's ArgGroup `input` requires exactly one of --sql-file or --physical-plan; \
             this branch is unreachable"
        ),
    }
}

fn read_input(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read(path).map_err(|e| format!("Failed to read {path}: {e}"))?)
    }
}
