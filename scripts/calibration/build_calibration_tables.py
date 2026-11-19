#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build src/optimizer/calibration/<profile>.cpp, the calibration profile of one GPU.

Reads the `cr_tp.db3` that `gqe-decompression-report` (gqe-bench) writes and derives:

  SM table         p50 SM decode throughput by nvCOMP data type, format and compression ratio
  DE table         p50 LZ4 decode throughput on the decompression engine (DE)
  bandwidth        total bytes over total time of the large host-to-device copies
  SM advantage     the speedup T_DE / T_SM above which selection picks the SM format, set to a
                   percentile (--sm-advantage-percentile, default p50) over the measured buffers

The inputs are the tables `raw_samples` (one row per buffer, format, backend and repetition),
`h2d_transfers` (one row per host-to-device copy) and `run_info` (the GPU, hosts, dates, GQE
revision and zone map partition size of the runs), which name the GPU and fill the header. Pass
several databases when formats were measured in separate runs.

See docs/tuning/automatic-compression-calibration.md for the definitions and for adding a GPU.

Usage:
  scripts/calibration/build_calibration_tables.py cr_tp.db3 --profile gb300

The profile name is the only required option; every other one has a default.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

NVCOMP_DATA_TYPES = ("CHAR", "INT", "LONGLONG")
SM_FORMATS = ("ans", "bitpacking", "cascaded")
DE_FORMAT = "lz4"
DE_BACKEND = "de"
# The columns that name one measured buffer. A double and a decimal sweep measure the same column
# as two buffers.
BUFFER_KEY = 'source_table, "column", decimal_type, column_part'
INVALID_STATUSES = ("overlapping_ranges",)
MIB = 2**20
GIB = 2**30
COLUMN_LIMIT = 100  # .clang-format column limit, so regenerating is idempotent

LICENSE_HEADER = """\
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
 */"""


def sql_list(values: tuple[str, ...]) -> str:
    """A SQL list of string literals, for an IN clause."""
    return ", ".join(f"'{value}'" for value in values)


def require_percentile_cont(conn: sqlite3.Connection) -> None:
    """Exit when this SQLite has no percentile_cont, which every percentile below is computed with.

    percentile_cont is part of SQLite's percentile extension (https://sqlite.org/percentile.html),
    which not every SQLite package includes. The SQLite of the GQE build container's Python
    (/conda/envs/gqe) does.
    """
    try:
        conn.execute("SELECT percentile_cont(1, 0.5)").fetchone()
    except sqlite3.OperationalError:
        sys.exit(
            f"SQLite {sqlite3.sqlite_version} has no percentile_cont "
            "(https://sqlite.org/percentile.html). Use a Python whose SQLite includes it, such as "
            "the one in the GQE build container."
        )


def open_database(databases: list[Path]) -> sqlite3.Connection:
    """An in-memory database holding the union of the tables of every input database."""
    # uri=True so the read-only "file:" names below are parsed as URIs.
    conn = sqlite3.connect(":memory:", uri=True)
    for position, path in enumerate(databases):
        if not path.is_file():
            sys.exit(f"{path} does not exist")
        conn.execute(f"ATTACH DATABASE ? AS input{position}", (f"file:{path}?mode=ro",))
    for table in ("raw_samples", "h2d_transfers"):
        selects = " UNION ALL ".join(
            f"SELECT * FROM input{position}.{table}" for position in range(len(databases))
        )
        try:
            conn.execute(f"CREATE TEMP TABLE {table} AS {selects}")
        except sqlite3.OperationalError as error:
            sys.exit(f"{', '.join(map(str, databases))} is not a cr_tp.db3 ({error})")
    return conn


def read_run_info(conn: sqlite3.Connection, database_count: int) -> dict[str, str] | None:
    """The conditions of the runs merged over the input databases, or None when one of them has no
    run_info table because an older gqe-decompression-report wrote it."""
    rows = []
    for position in range(database_count):
        try:
            rows += conn.execute(
                "SELECT gpu_name, hostnames, first_date, last_date, gqe_revisions, "
                f"zone_map_partition_sizes FROM input{position}.run_info"
            ).fetchall()
        except sqlite3.OperationalError:
            return None
    gpu_names = sorted({row[0] for row in rows})
    if len(gpu_names) != 1:
        sys.exit(f"the databases must measure one GPU, but they name {gpu_names or 'none'}")

    def merged(column: int, key=None) -> str:
        values = {value.strip() for row in rows for value in row[column].split(",")}
        return ", ".join(sorted(values - {""}, key=key))

    return {
        "gpu_name": gpu_names[0],
        "hostnames": merged(1),
        "first_date": min(row[2] for row in rows),
        "last_date": max(row[3] for row in rows),
        "gqe_revisions": merged(4),
        "zone_map_partition_sizes": merged(5, key=int),
    }


def describe(
    calibrated_on: str,
    run_info: dict[str, str] | None,
    measured_samples: int,
    min_bytes: int,
    sm_advantage_percentile: float,
    note: str | None,
) -> str:
    """What was measured, for the header: the GPU, and the conditions of the runs when known."""
    method = (
        f"p50 of one NVTX range per decompressed buffer of at least {min_bytes / MIB:g} MiB, "
        f"{measured_samples} measured samples per point, SM advantage at "
        f"p{sm_advantage_percentile:g}"
    )
    if run_info is None:
        description = f"{calibrated_on} measurements: {method}"
    else:
        dates = run_info["first_date"]
        if run_info["last_date"] != dates:
            dates += f" to {run_info['last_date']}"
        description = (
            f"{calibrated_on} measurements on {run_info['hostnames']}, {dates}, "
            f"gqe {run_info['gqe_revisions'] or 'unknown'}: {method}, "
            f"zone_map_partition_size {run_info['zone_map_partition_sizes']}"
        )
    return f"{description}, {note}" if note else description


def check_runs_are_valid(conn: sqlite3.Connection) -> None:
    """Exit when no sample was measured or a run's per-buffer attribution does not hold."""
    if conn.execute("SELECT COUNT(*) FROM raw_samples").fetchone()[0] == 0:
        sys.exit("the database has no raw samples")
    invalid = conn.execute(
        f"SELECT DISTINCT decode_status FROM raw_samples WHERE decode_status IN "
        f"({sql_list(INVALID_STATUSES)})"
    ).fetchall()
    if invalid:
        sys.exit(
            f"the run has {sorted(row[0] for row in invalid)} rows, so the per-buffer attribution "
            "does not hold and the run is invalid"
        )


def read_measured_samples(conn: sqlite3.Connection) -> int:
    """The number of measured samples per point: the most repetitions of any buffer, format and
    backend."""
    (measured_samples,) = conn.execute(
        "SELECT MAX(repetitions) FROM (SELECT COUNT(*) AS repetitions FROM raw_samples "
        f"GROUP BY {BUFFER_KEY}, format, backend)"
    ).fetchone()
    return measured_samples


def create_decoded_samples_view(conn: sqlite3.Connection, min_bytes: int) -> None:
    """The view `decoded_samples` of the samples that can be calibrated.

    A sample qualifies when it decoded, its buffer has at least `min_bytes` and it compressed.
    """
    # A view cannot take bound parameters, so min_bytes is inlined as an integer literal.
    conn.execute(
        f"""
        CREATE TEMP VIEW decoded_samples AS
        SELECT nvcomp_physical_type AS nvcomp_data_type, format, backend,
               ROUND(measured_ratio, 4) AS ratio, throughput_GiBps,
               compressed_bytes, uncompressed_bytes, {BUFFER_KEY}
        FROM raw_samples
        WHERE throughput_GiBps IS NOT NULL
          AND nvcomp_physical_type IN ({sql_list(NVCOMP_DATA_TYPES)})
          AND compressed_bytes < uncompressed_bytes
          AND uncompressed_bytes >= {int(min_bytes)}"""
    )


def read_points(conn: sqlite3.Connection, where: str, min_samples: int) -> list[tuple]:
    """The (nvCOMP data type, format, ratio, p50 throughput) buckets of the samples matching
    `where`, sorted by nvCOMP data type, then format, then ratio."""
    return conn.execute(
        f"""
        SELECT nvcomp_data_type, format, ratio, percentile_cont(throughput_GiBps, 0.5)
        FROM decoded_samples
        WHERE {where}
        GROUP BY nvcomp_data_type, format, ratio
        HAVING COUNT(*) >= ?
        ORDER BY nvcomp_data_type, format, ratio""",
        (min_samples,),
    ).fetchall()


SM_WHERE = f"format IN ({sql_list(SM_FORMATS)})"
DE_WHERE = f"format = '{DE_FORMAT}' AND backend = '{DE_BACKEND}'"


def read_bandwidth_gibps(conn: sqlite3.Connection, min_h2d_bytes: int) -> float:
    """Total bytes over total time of the large host-to-device copies, in GiB/s.

    This is the byte-weighted harmonic mean of the copy rates. The cost model charges a buffer
    compressed_bytes / bandwidth, so the sum of the charges over a load only equals the observed
    sum of the copy times if the bandwidth is total bytes over total time.
    """
    row = conn.execute(
        f"""
        SELECT SUM(bytes) / (SUM(duration_ns) / 1e9) / {GIB}
        FROM h2d_transfers WHERE duration_ns > 0 AND bytes >= ?""",
        (min_h2d_bytes,),
    ).fetchone()
    if row[0] is None:
        sys.exit(
            f"no host-to-device copy of at least {min_h2d_bytes / MIB:g} MiB to measure bandwidth"
        )
    return row[0]


def read_sm_advantage(
    conn: sqlite3.Connection, bandwidth_gibps: float, percentile: float
) -> tuple[float, int]:
    """The `percentile` (0 to 100) of T_DE / T_SM over the buffers that both a DE and an SM format
    compress, and how many buffers it ranks. Neither side counts the uncompressed copy, which
    selection compares with the winner separately and without the advantage."""
    row = conn.execute(
        f"""
        WITH format_time AS (
          SELECT {BUFFER_KEY}, format, backend,
                 MAX(compressed_bytes) / {GIB}.0 / :bandwidth
                   + MAX(uncompressed_bytes) / {GIB}.0 / percentile_cont(throughput_GiBps, 0.5)
                   AS seconds
          FROM decoded_samples GROUP BY {BUFFER_KEY}, format, backend),
        de_time AS (
          SELECT {BUFFER_KEY}, MIN(seconds) AS seconds FROM format_time
          WHERE {DE_WHERE} GROUP BY {BUFFER_KEY}),
        sm_time AS (
          SELECT {BUFFER_KEY}, MIN(seconds) AS seconds FROM format_time
          WHERE {SM_WHERE} GROUP BY {BUFFER_KEY})
        SELECT COUNT(*), percentile_cont(de_time.seconds / sm_time.seconds, :fraction)
        FROM de_time JOIN sm_time USING ({BUFFER_KEY})""",
        {"bandwidth": bandwidth_gibps, "fraction": percentile / 100},
    ).fetchone()
    if row is None or row[1] is None:
        sys.exit("no buffer has both a DE and an SM representation to rank")
    return row[1], row[0]


def braced_initializer(fields: list[str]) -> list[str]:
    """One braced initializer, laid out as clang-format would."""
    single = "  {" + ", ".join(fields) + "},"
    if len(single) <= COLUMN_LIMIT:
        return [single]
    lines = [f"  {{{fields[0]},"]
    lines += [f"   {field}," for field in fields[1:-1]]
    lines.append(f"   {fields[-1]}}},")
    return lines


def render_points(name: str, rows: list[tuple]) -> list[str]:
    """A constexpr array of throughput points, in the order of `rows`."""
    lines = [f"constexpr auto {name} = std::to_array<throughput_point>({{"]
    for nvcomp_data_type, fmt, ratio, p50 in rows:
        lines += braced_initializer(
            [
                f"NVCOMP_TYPE_{nvcomp_data_type}",
                f"gqe::compression_format::{fmt}",
                f"{ratio:.4f}",
                f"{p50:.4f}",
            ]
        )
    return lines + ["});", ""]


def render(
    profile: str,
    calibrated_on: str,
    provenance: str,
    sm_rows: list[tuple],
    de_rows: list[tuple],
    bandwidth_gibps: float,
    sm_advantage: float,
) -> str:
    """The C++ source of one profile."""
    header = textwrap.wrap(
        f"Generated by scripts/calibration/build_calibration_tables.py from {provenance}; do not "
        "edit. Throughputs and the bandwidth are in GiB/s.",
        width=COLUMN_LIMIT,
        initial_indent="// ",
        subsequent_indent="// ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(
        [
            LICENSE_HEADER,
            "",
            *header,
            "",
            '#include "tables.hpp"',
            "",
            "#include <array>",
            "",
            "namespace gqe::optimizer::calibration_tables {",
            "",
            "namespace {",
            "",
            *render_points("sm_points", sm_rows),
            *render_points("de_points", de_rows),
            "}  // namespace",
            "",
            f"calibration_profile const& {profile}_profile()",
            "{",
            "  static calibration_profile const profile{",
            f'    "{profile}", "{calibrated_on}", de_points, sm_points, {bandwidth_gibps!r}, '
            f"{sm_advantage!r}}};",
            "  return profile;",
            "}",
            "",
            "}  // namespace gqe::optimizer::calibration_tables",
            "",
        ]
    )


def main() -> None:
    """Derive the profile of one GPU from its measurements and write the C++ source."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "databases", type=Path, nargs="+", help="cr_tp.db3 of the calibration sweep"
    )
    parser.add_argument(
        "--profile", required=True, help="lower case profile name a session selects, e.g. gb300"
    )
    parser.add_argument(
        "--calibrated-on",
        help="device name prefix a session warns about when run elsewhere (default: the GPU in "
        "run_info; required for a database without run_info)",
    )
    parser.add_argument(
        "--provenance",
        help="note appended to the header, e.g. the dataset or the nvCOMP version",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "src" / "optimizer" / "calibration",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=100 * MIB,
        help="drop buffers below this many bytes (default 100 MiB)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=5, help="drop buckets with fewer samples (default 5)"
    )
    parser.add_argument(
        "--min-h2d-bytes",
        type=int,
        default=64 * MIB,
        help="ignore copies below this many bytes when measuring bandwidth (default 64 MiB)",
    )
    parser.add_argument(
        "--sm-advantage-percentile",
        type=float,
        default=50,
        help="percentile (0 to 100) of T_DE / T_SM over the measured buffers that becomes the SM "
        "advantage; about this share of the buffers stays on the DE (default 50)",
    )
    args = parser.parse_args()
    if not 0 <= args.sm_advantage_percentile <= 100:
        parser.error("--sm-advantage-percentile must be between 0 and 100")

    conn = open_database(args.databases)
    require_percentile_cont(conn)
    check_runs_are_valid(conn)
    run_info = read_run_info(conn, len(args.databases))
    calibrated_on = args.calibrated_on or (run_info and run_info["gpu_name"])
    if not calibrated_on:
        sys.exit("the database has no run_info naming the GPU; pass --calibrated-on")
    profile = args.profile
    if not re.fullmatch(r"[a-z][a-z0-9_]*", profile):
        sys.exit(f"the profile name {profile!r} must be a lower case C++ identifier")
    measured_samples = read_measured_samples(conn)
    create_decoded_samples_view(conn, args.min_bytes)
    sm_rows = read_points(conn, SM_WHERE, args.min_samples)
    de_rows = read_points(conn, DE_WHERE, args.min_samples)
    if not sm_rows or not de_rows:
        sys.exit(
            f"no bucket reached {args.min_samples} samples "
            f"(sm: {len(sm_rows)}, de: {len(de_rows)})"
        )
    bandwidth_gibps = read_bandwidth_gibps(conn, args.min_h2d_bytes)
    sm_advantage, buffer_count = read_sm_advantage(
        conn, bandwidth_gibps, args.sm_advantage_percentile
    )

    path = args.output_dir / f"{profile}.cpp"
    path.write_text(
        render(
            profile,
            calibrated_on,
            describe(
                calibrated_on,
                run_info,
                measured_samples,
                args.min_bytes,
                args.sm_advantage_percentile,
                args.provenance,
            ),
            sm_rows,
            de_rows,
            bandwidth_gibps,
            sm_advantage,
        )
    )
    print(f"wrote {len(sm_rows)} SM and {len(de_rows)} DE points to {path}")
    print(f"bandwidth {bandwidth_gibps!r} GiB/s, SM advantage {sm_advantage!r}")
    print(f"the advantage ranks {buffer_count} buffers")


if __name__ == "__main__":
    main()
