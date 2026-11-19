#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load TPC-H schema and data into a GQE Flight SQL server.

Usage:
  ./load_tpch.py [--server-url URL] [--schema FILENAME] DATA_DIR

DATA_DIR must contain:
  - <schema>               (CREATE TABLE statements; filename selected by --schema,
                            default `schema.sql`. Use `ci_schema.sql` for the
                            handcoded TPC-H fixtures whose physical plans depend
                            on hand-tuned column types.)
  - customer/, lineitem/, nation/, orders/, part/, partsupp/, region/, supplier/
                           (directories with .parquet files)

Example:
  ./load_tpch.py /scratch/local/tpch/sf10_chunk16m_id64
  ./load_tpch.py --schema ci_schema.sql /scratch/local/tpch/sf0.01_id64
  ./load_tpch.py --server-url http://10.0.0.1:50051 /scratch/local/tpch/sf10_chunk16m_id64
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


def resolve_cli() -> Path:
    cli_env = os.environ.get("GQE_CLI")
    if not cli_env:
        print("Error: GQE_CLI environment variable must be set to the path of the gqe-cli binary.",
              file=sys.stderr)
        sys.exit(1)
    cli = Path(cli_env)
    if not (cli.is_file() and os.access(cli, os.X_OK)):
        print(f"Error: GQE_CLI={cli} is not an executable file", file=sys.stderr)
        sys.exit(1)
    return cli


def run(cmd: list, stdin: str | None = None) -> None:
    result = subprocess.run(cmd, input=stdin, text=True)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load TPC-H data into a GQE Flight SQL server.")
    parser.add_argument("--server-url", default="http://localhost:50051")
    parser.add_argument(
        "--schema",
        default="schema.sql",
        help="Filename of the CREATE TABLE statements inside DATA_DIR "
        "(default `schema.sql`; pass `ci_schema.sql` for the handcoded fixtures).",
    )
    parser.add_argument("data_dir", metavar="DATA_DIR", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir
    server_url = args.server_url
    cli = resolve_cli()

    schema_path = data_dir / args.schema
    if not schema_path.is_file():
        print(f"Error: {schema_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Server:  {server_url}")
    print(f"Data:    {data_dir}")
    print(f"Schema:  {schema_path}")
    print(f"Client:  {cli}")
    print()

    # 1. Create schema
    print("=== Creating schema ===")
    run([cli, "--server-url", server_url, "--sql-file", schema_path])

    # 2. Load data (one COPY per table)
    print("=== Loading data ===")
    for table in TABLES:
        table_dir = data_dir / table
        if not table_dir.is_dir():
            print(f"Warning: {table_dir} not found, skipping", file=sys.stderr)
            continue
        print(f"  COPY {table} ...")
        run(
            [cli, "--server-url", server_url, "--sql-file", "-"],
            stdin=f"COPY {table} FROM '{table_dir}' (FORMAT parquet)",
        )

    print()
    print("Done. All tables loaded.")


if __name__ == "__main__":
    main()
