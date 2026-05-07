/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use datafusion_substrait::serializer;

use datafusion::error::Result;
use datafusion::prelude::*;

use colored::Colorize;
use std::env;
extern crate yaml_rust;
use yaml_rust::{Yaml, YamlLoader};

#[derive(Debug)]
struct TableConfig {
    name: String,
    file_or_directory: String,
}

#[derive(Debug)]
struct QueryConfig {
    sql: String,
    output: String,
    tables: Vec<TableConfig>,
}

fn get_query_config(config_path: &String) -> QueryConfig {
    let s = std::fs::read_to_string(config_path).unwrap();
    let docs = YamlLoader::load_from_str(&s).unwrap();
    let mut tables: Vec<TableConfig> = vec![];
    let mut sql: String = "".to_string();
    let mut output: String = "".to_string();

    let map = &docs[0].as_hash().unwrap();

    for (section, value) in map.iter() {
        match section.as_str().unwrap() {
            "output" => output = value.as_str().unwrap().to_string(),
            "sql" => sql = value.as_str().unwrap().to_string(),
            "tables" => {
                for (name, config) in value.as_hash().unwrap().iter() {
                    let table_name = name.as_str().unwrap().to_string();
                    let table_config = config.as_hash().unwrap();
                    let key_file = Yaml::String("file".to_string());
                    let key_dir = Yaml::String("directory".to_string());
                    if table_config.contains_key(&key_dir) && table_config.contains_key(&key_file) {
                        // The user should not supply both directory and files
                        panic!("Only files OR directory should be specified for each table")
                    }
                    if let Some(file) = table_config.get(&key_file) {
                        tables.push(TableConfig {
                            name: table_name,
                            file_or_directory: file.as_str().unwrap().to_string(),
                        });
                    } else if let Some(directory) = table_config.get(&key_dir) {
                        // TODO: Combine with above when Rust supports if-let chain in the format `if let Some(x) = a || let Some(x) = b`
                        tables.push(TableConfig {
                            name: table_name,
                            file_or_directory: directory.as_str().unwrap().to_string(),
                        });
                    } else {
                        panic!("{}", format!(
                            "Please make sure file or directory section is defined for table {table_name}"
                        ))
                    }
                }
            }
            _ => panic!("Section should be either output, sql or tables"),
        }
    }
    // Return query config
    QueryConfig {
        sql,
        output,
        tables,
    }
}

async fn create_context(table_confs: Vec<TableConfig>) -> Result<SessionContext> {
    let ctx = SessionContext::new();
    for table in table_confs {
        // register parquet file with the execution context
        ctx.register_parquet(
            &table.name,
            &table.file_or_directory,
            ParquetReadOptions::default(),
        )
        .await?;
    }

    Ok(ctx)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("Usage: producer <path_to_config_file>")
    }

    let query_config = get_query_config(&args[1]);
    print!("{}\n{}", format!("[SQL]").blue(), query_config.sql);
    print!(
        "{}\n{:#?}\n",
        format!("[TABLE CONFIGS]").blue(),
        query_config.tables
    );
    print!(
        "{}\n {}\n",
        format!("[OUTPUT PATH]").blue(),
        query_config.output
    );

    let ctx = create_context(query_config.tables).await?;
    let path = query_config.output;
    let sql = query_config.sql;

    // Write substrait plan to file
    serializer::serialize(&sql, &ctx, &path).await?;

    Ok(())
}
