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

//! Integration tests for the Flight SQL + Substrait example.
//!
//! Most tests require TPC-H parquet data and reference results.
//! Set the following environment variables to run them:
//!
//!   TPCH_DATA_PATH     - path to TPC-H parquet data directory
//!   TPCH_REF_RESULTS   - path to TPC-H reference results directory
//!
//! Example:
//!   TPCH_DATA_PATH=/scratch/local/tpch/sf10_chunk16m_id64 \
//!   TPCH_REF_RESULTS=/scratch/local/tpch/reference_results/sf10 \
//!   cargo test -p gqe-cli --test integration --release
//!
//! Tests that do not require external data (e.g. SET/SHOW session options)
//! run without any environment variables set.

use datafusion_common_runtime::SpawnedTask;
use gqe_cli::{client, server};

const TPCH_TABLES: &[&str] = &[
    "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
];

const VERIFY_PARQUET_SCRIPT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test/end_to_end/verify_parquet.py"
);

/// Return TPC-H data path from env, or skip the test.
fn tpch_data_path() -> Option<String> {
    std::env::var("TPCH_DATA_PATH").ok()
}

/// Return TPC-H reference results path from env, or None.
fn tpch_ref_results() -> Option<String> {
    std::env::var("TPCH_REF_RESULTS").ok()
}

/// Allocate a random free port by binding to port 0, then immediately releasing it.
fn free_addr() -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

/// Start a server on a random address, returning the address and a spawned task.
/// The task is aborted on drop (cancel-safe via SpawnedTask).
fn start_server() -> (std::net::SocketAddr, SpawnedTask<()>) {
    let addr = free_addr();
    let task = SpawnedTask::spawn(async move {
        let _ = server::serve(addr).await;
    });
    (addr, task)
}

/// Build SQL to create the TPC-H schema and load data from the given path.
/// Reads ci_schema.sql from the data directory and generates COPY statements
/// that point to the per-table subdirectories.
fn tpch_setup_sql(data_path: &str) -> String {
    let schema_path = format!("{data_path}/ci_schema.sql");
    let schema_sql = std::fs::read_to_string(&schema_path)
        .unwrap_or_else(|e| panic!("Failed to read {schema_path}: {e}"));

    let mut copy_stmts = String::new();
    for table in TPCH_TABLES {
        copy_stmts.push_str(&format!(
            "COPY {table} FROM '{data_path}/{table}' (FORMAT 'parquet');\n"
        ));
    }

    format!("{schema_sql}\n{copy_stmts}")
}

fn load_tpch_query(query_num: u32) -> String {
    let query_dir =
        std::env::var("TPCH_QUERIES").unwrap_or_else(|_| "/home/gqe/tpch/queries".to_string());
    let query_path = std::path::PathBuf::from(query_dir).join(format!("q{query_num}.sql"));
    std::fs::read_to_string(&query_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", query_path.display()))
}

macro_rules! tpch_test {
    ($name:ident, $query_num:expr) => {
        #[tokio::test]
        async fn $name() {
            let Some(data_path) = tpch_data_path() else {
                eprintln!("Skipping {}: TPCH_DATA_PATH not set", stringify!($name));
                return;
            };
            let ref_results = tpch_ref_results();

            let (addr, _server_task) = start_server();
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;

            let url = format!("http://{addr}");

            // Create schema and load data
            let setup_sql = tpch_setup_sql(&data_path);
            client::execute(&url, &setup_sql)
                .await
                .expect("schema/data setup failed");

            let sql = load_tpch_query($query_num);

            // Write results to a temp parquet file for validation.
            let tmpdir = tempfile::tempdir().expect("failed to create temp dir");
            let parquet_path = tmpdir.path().join(format!("q{}.parquet", $query_num));
            let parquet_str = parquet_path.to_str().unwrap();

            client::run_against(&url, &sql, Some(parquet_str))
                .await
                .unwrap_or_else(|e| panic!("Q{} failed: {e}", $query_num));

            // Validate against reference results if available.
            if let Some(ref_dir) = &ref_results {
                let ref_file =
                    std::path::Path::new(ref_dir).join(format!("q{}.parquet", $query_num));
                if ref_file.exists() {
                    let status = std::process::Command::new("python3")
                        .arg(VERIFY_PARQUET_SCRIPT)
                        .arg(parquet_str)
                        .arg(&ref_file)
                        .status()
                        .expect("failed to run verify_parquet.py");
                    assert!(status.success(), "Q{} validation failed", $query_num);
                }
            }
        }
    };
}

#[tokio::test]
async fn adhoc_count_lineitem() {
    let Some(data_path) = tpch_data_path() else {
        eprintln!("Skipping adhoc_count_lineitem: TPCH_DATA_PATH not set");
        return;
    };

    let (addr, _server_task) = start_server();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let url = format!("http://{addr}");

    // Create schema and load data
    let setup_sql = tpch_setup_sql(&data_path);
    client::execute(&url, &setup_sql)
        .await
        .expect("schema/data setup failed");

    let result = client::execute(&url, "SELECT COUNT(*) FROM lineitem")
        .await
        .expect("query failed");

    let batches = match result {
        client::ExecuteResult::Query(b) => b,
        _ => panic!("expected Query result from SELECT"),
    };

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("expected Int64Array for COUNT(*)");
    assert!(
        count.value(0) > 0,
        "expected positive row count, got {}",
        count.value(0)
    );
}

#[tokio::test]
async fn drop_table() {
    let Some(data_path) = tpch_data_path() else {
        eprintln!("Skipping drop_table: TPCH_DATA_PATH not set");
        return;
    };

    let (addr, _server_task) = start_server();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let url = format!("http://{addr}");

    // Create schema and load data
    let setup_sql = tpch_setup_sql(&data_path);
    client::execute(&url, &setup_sql)
        .await
        .expect("schema/data setup failed");

    // Verify the table exists
    let tables = client::discover(&url).await.expect("discovery failed");
    assert!(
        tables.iter().any(|t| t.name == "region"),
        "region table should exist before DROP"
    );

    // Drop the table
    client::execute(&url, "DROP TABLE region")
        .await
        .expect("DROP TABLE failed");

    // Verify the table is gone
    let tables = client::discover(&url).await.expect("discovery failed");
    assert!(
        !tables.iter().any(|t| t.name == "region"),
        "region table should not exist after DROP"
    );
}

#[tokio::test]
async fn schema_discovery_after_create() {
    let Some(data_path) = tpch_data_path() else {
        eprintln!("Skipping schema_discovery_after_create: TPCH_DATA_PATH not set");
        return;
    };

    let (addr, _server_task) = start_server();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let url = format!("http://{addr}");

    // Create schema and load data via client
    let setup_sql = tpch_setup_sql(&data_path);
    client::execute(&url, &setup_sql)
        .await
        .expect("schema/data setup failed");

    // Discover tables — all 8 TPC-H tables should be present
    let tables = client::discover(&url).await.expect("discovery failed");

    let expected_tables = [
        "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier",
    ];
    for name in &expected_tables {
        let table = tables
            .iter()
            .find(|t| t.name == *name)
            .unwrap_or_else(|| panic!("table '{name}' not discovered"));
        assert!(table.row_count.is_some(), "table '{name}' has no row count");
        assert!(
            table.row_count.unwrap() > 0,
            "table '{name}' has zero row count"
        );
        assert!(
            !table.schema.fields().is_empty(),
            "table '{name}' has empty schema"
        );
    }
}

#[tokio::test]
async fn set_session_option() {
    let (addr, _server_task) = start_server();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let url = format!("http://{addr}");

    let result = client::execute(&url, "SET join_use_unique_keys TO true")
        .await
        .expect("SET failed");

    match result {
        client::ExecuteResult::Statement(rows) => assert_eq!(rows, 0),
        _ => panic!("expected Statement result from SET"),
    }
}

#[tokio::test]
async fn set_and_show_roundtrip() {
    let (addr, _server_task) = start_server();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let url = format!("http://{addr}");

    client::execute(&url, "SET join_use_unique_keys TO true")
        .await
        .expect("SET failed");

    let result = client::execute(&url, "SHOW join_use_unique_keys")
        .await
        .expect("SHOW failed");

    let batches = match result {
        client::ExecuteResult::Query(b) => b,
        _ => panic!("expected Query result from SHOW"),
    };

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let name_col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("expected StringArray for name column");
    let value_col = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("expected StringArray for value column");

    assert_eq!(name_col.value(0), "join_use_unique_keys");
    assert_eq!(value_col.value(0), "true");
}

#[tokio::test]
async fn show_all() {
    let (addr, _server_task) = start_server();
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let url = format!("http://{addr}");

    client::execute(&url, "SET join_use_unique_keys TO false").await.expect("SET failed");
    client::execute(&url, "SET max_num_workers TO 4").await.expect("SET failed");

    let result = client::execute(&url, "SHOW ALL").await.expect("SHOW ALL failed");

    let batches = match result {
        client::ExecuteResult::Query(b) => b,
        _ => panic!("expected Query result from SHOW ALL"),
    };

    assert_eq!(batches.len(), 1);

    let name_col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .expect("expected StringArray for name column");

    let names: Vec<&str> = name_col.iter().flatten().collect();
    assert!(names.contains(&"join_use_unique_keys"), "expected join_use_unique_keys in SHOW ALL");
    assert!(names.contains(&"max_num_workers"), "expected max_num_workers in SHOW ALL");
    assert!(names.windows(2).all(|w| w[0] <= w[1]), "expected results sorted");
}

tpch_test!(tpch_q1, 1);
tpch_test!(tpch_q2, 2);
tpch_test!(tpch_q3, 3);
tpch_test!(tpch_q4, 4);
tpch_test!(tpch_q5, 5);
tpch_test!(tpch_q6, 6);
tpch_test!(tpch_q7, 7);
tpch_test!(tpch_q8, 8);
tpch_test!(tpch_q9, 9);
tpch_test!(tpch_q10, 10);
// Q11 is skipped: the benchmark SQL uses the SF1 fraction (0.0001) which is
// incorrect for other scale factors. The correct fraction is 0.0001/SF.
// tpch_test!(tpch_q11, 11);
tpch_test!(tpch_q12, 12);
tpch_test!(tpch_q13, 13);
tpch_test!(tpch_q14, 14);
tpch_test!(tpch_q15, 15);
tpch_test!(tpch_q16, 16);
tpch_test!(tpch_q17, 17);
tpch_test!(tpch_q18, 18);
tpch_test!(tpch_q19, 19);
tpch_test!(tpch_q20, 20);
tpch_test!(tpch_q21, 21);
tpch_test!(tpch_q22, 22);
