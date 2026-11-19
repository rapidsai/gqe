#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one or more TPC-H queries against a GQE Flight SQL server.

Usage:
  ./run_tpch.py --mode {sql,physical} [--server-url URL] DIR QUERY [QUERY ...]

QUERY can be a query number (e.g. 1) or 'all' to run all 22 queries.
For SQL mode, DIR contains q01.sql ... q22.sql. For physical mode, DIR contains
q1.pb ... q22.pb.

Examples:
  ./run_tpch.py --mode sql /scratch/local/tpch/queries 1
  ./run_tpch.py --mode physical /scratch/local/tpch/physical_plans 1 5 14
  ./run_tpch.py --mode sql /scratch/local/tpch/queries all
  ./run_tpch.py --mode physical --server-url http://10.0.0.1:50051 \\
      /scratch/local/tpch/physical_plans 1 2 3
  ./run_tpch.py --mode sql --validate /scratch/local/tpch/reference_results/sf1 \\
      /scratch/local/tpch/queries all
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ALL_QUERIES = list(range(1, 23))

# Per --mode: (gqe-cli flag accepting the input file, file extension in DIR).
MODE_INFO = {
    "sql": ("--sql-file", "sql"),
    "physical": ("--physical-plan", "pb"),
}


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


def run_query(
    cli: Path,
    server_url: str,
    cli_flag: str,
    input_file: Path,
    quiet: bool = False,
    parquet_output: Path | None = None,
) -> None:
    cmd: list[str | Path] = [cli, "--server-url", server_url, cli_flag, input_file]
    if parquet_output is not None:
        cmd += ["--parquet", parquet_output]
    stdout = subprocess.DEVNULL if quiet else None
    result = subprocess.run(cmd, text=True, stdout=stdout)
    if result.returncode != 0:
        sys.exit(result.returncode)


def query_num(arg: str) -> int | str:
    if arg.lower() == "all":
        return "all"
    try:
        n = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid query number '{arg}' (use 1-22 or 'all')")
    if not 1 <= n <= 22:
        raise argparse.ArgumentTypeError(f"query number {n} out of range (1-22)")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TPC-H queries against a GQE Flight SQL server.")
    parser.add_argument("--mode", required=True, choices=sorted(MODE_INFO.keys()),
                        help="`sql` for SQL text files, `physical` for serialized physical-plan protobufs")
    parser.add_argument("--server-url", default="http://localhost:50051")
    parser.add_argument("input_dir", metavar="DIR", type=Path,
                        help="Directory containing q01.sql ... q22.sql or q1.pb ... q22.pb")
    parser.add_argument("query", nargs="+", metavar="QUERY", type=query_num,
                        help="Query number(s) to run (1-22), or 'all'")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize query execution order")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress query result output")
    parser.add_argument("--validate", metavar="REFERENCE_DIR", type=Path, default=None,
                        help="Validate results against reference parquet files in REFERENCE_DIR "
                             "(expects q1.parquet ... q22.parquet)")
    args = parser.parse_args()

    cli_flag, extension = MODE_INFO[args.mode]
    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.validate and not args.validate.is_dir():
        print(f"Error: reference directory not found: {args.validate}", file=sys.stderr)
        sys.exit(1)

    # Import verify_parquet lazily so the dependency on pandas is only required
    # when --validate is used.
    validate_fn = None
    if args.validate:
        verify_parquet_path = Path(__file__).resolve().parent.parent / "test" / "end_to_end"
        sys.path.insert(0, str(verify_parquet_path))
        from verify_parquet import verify as _verify
        validate_fn = _verify

    # list() copy needed for ALL_QUERIES so random.shuffle doesn't mutate the module-level list;
    # args.query is already a fresh list from argparse.
    query_nums = list(ALL_QUERIES) if args.query == ["all"] else args.query  # type: ignore[assignment]
    if args.randomize:
        random.shuffle(query_nums)

    cli = resolve_cli()

    print(f"Server:     {args.server_url}")
    print(f"Mode:       {args.mode}")
    print(f"Inputs:     {input_dir}")
    print(f"Client:     {cli}")
    print(f"Running:    {' '.join(str(n) for n in query_nums)}")
    if args.validate:
        print(f"Validate:   {args.validate}")
    print()

    failures: list[int] = []
    missing_inputs: list[int] = []

    with tempfile.TemporaryDirectory(prefix="gqe_tpch_") as tmpdir:
        for n in query_nums:
            query_number = f"{n:02d}" if args.mode == "sql" else str(n)
            input_file = input_dir / f"q{query_number}.{extension}"
            if not input_file.is_file():
                print(f"Error: {input_file} not found", file=sys.stderr)
                missing_inputs.append(n)
                continue

            parquet_out = Path(tmpdir) / f"q{n}.parquet" if args.validate else None

            print(f"=== Q{n} ===")
            run_query(cli, args.server_url, cli_flag, input_file,
                      quiet=args.quiet, parquet_output=parquet_out)

            if args.validate:
                ref_file = args.validate / f"q{n}.parquet"
                if not ref_file.is_file():
                    print(f"  Warning: reference {ref_file} not found, skipping validation",
                          file=sys.stderr)
                    continue
                try:
                    validate_fn(str(parquet_out), str(ref_file))
                    print(f"  PASS")
                except AssertionError as e:
                    print(f"  FAIL: {e}", file=sys.stderr)
                    failures.append(n)

    if missing_inputs:
        print(f"\nMissing inputs for queries: {', '.join(f'Q{n}' for n in missing_inputs)}",
              file=sys.stderr)

    if failures:
        print(f"\nValidation failed for queries: {', '.join(f'Q{n}' for n in failures)}",
              file=sys.stderr)

    if missing_inputs or failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
