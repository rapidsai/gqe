# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_calibration_tables.py on small synthetic cr_tp.db3 databases.

The tests use only the standard library. Run them with a Python whose SQLite has percentile_cont,
such as the one in the GQE build container:

  python -m unittest discover -s scripts/calibration -p "test_*.py"

ctest runs them as `calibration_generator_test`.
"""
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_calibration_tables as build  # noqa: E402

GIB = 2**30
BIG = 200 * 2**20  # above the default 100 MiB floor
ONE_SECOND = 1_000_000_000

RAW_COLUMNS = (
    "source_table, \"column\", table_name, decimal_type, column_kind, sql_type, "
    "nvcomp_physical_type, mapping_reason, format, backend, column_part, repeat_num, "
    "measured_ratio, uncompressed_bytes, compressed_bytes, decode_status, decompress_s, "
    "throughput_GiBps"
)


def setUpModule() -> None:
    """Fail every test with one clear message when this SQLite has no percentile_cont."""
    try:
        sqlite3.connect(":memory:").execute("SELECT percentile_cont(1, 0.5)")
    except sqlite3.OperationalError as error:
        raise RuntimeError(
            f"SQLite {sqlite3.sqlite_version} has no percentile_cont "
            "(https://sqlite.org/percentile.html); run the tests with the GQE build container's "
            "Python"
        ) from error


def sample(
    fmt: str,
    backend: str,
    ratio: float,
    throughput: float | None,
    status: str = "decoded",
    column: str = "c",
    nvcomp_data_type: str = "INT",
) -> dict:
    """One decode sample: a repetition of one buffer in one format and backend."""
    return {
        "column": column,
        "nvcomp_data_type": nvcomp_data_type,
        "fmt": fmt,
        "backend": backend,
        "ratio": ratio,
        "throughput": throughput,
        "status": status,
    }


def make_database(
    path: Path, samples: list[dict], transfers: list[dict], run_info: tuple | None = None
) -> Path:
    """A cr_tp.db3 with the given samples, host-to-device copies and, if given, run_info row.

    Each sample is a dict with the keys of `sample`, each transfer a dict with `bytes` and
    `duration_ns`.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE raw_samples (
          source_table TEXT, "column" TEXT, table_name TEXT, decimal_type TEXT, column_kind TEXT,
          sql_type TEXT, nvcomp_physical_type TEXT, mapping_reason TEXT, format TEXT, backend TEXT,
          column_part TEXT, repeat_num INTEGER, measured_ratio REAL, uncompressed_bytes INTEGER,
          compressed_bytes INTEGER, decode_status TEXT, decompress_s REAL, throughput_GiBps REAL);
        CREATE TABLE h2d_transfers (database TEXT, run_number INTEGER, bytes INTEGER,
                                    duration_ns INTEGER);
        """
    )
    for index, row in enumerate(samples):
        conn.execute(
            f"INSERT INTO raw_samples ({RAW_COLUMNS}) VALUES "
            "('t', :column, 't_c', '', 'fixed_width', 'INT', :nvcomp_data_type, '', :fmt, "
            ":backend, 'value', :repeat_num, :ratio, :uncompressed_bytes, :compressed_bytes, "
            ":status, NULL, :throughput)",
            {
                **row,
                "repeat_num": index,
                "uncompressed_bytes": BIG,
                "compressed_bytes": int(BIG / row["ratio"]),
            },
        )
    conn.executemany(
        "INSERT INTO h2d_transfers VALUES ('db', 1, :bytes, :duration_ns)", transfers
    )
    if run_info is not None:
        conn.execute(
            "CREATE TABLE run_info (gpu_name TEXT, hostnames TEXT, first_date TEXT, "
            "last_date TEXT, gqe_revisions TEXT, zone_map_partition_sizes TEXT)"
        )
        conn.execute("INSERT INTO run_info VALUES (?, ?, ?, ?, ?, ?)", run_info)
    conn.commit()
    conn.close()
    return path


def transfers_database(transfers: list[tuple[int, int]]) -> sqlite3.Connection:
    """An in-memory database with the host-to-device copies `(bytes, duration_ns)`."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE h2d_transfers (database, run_number, bytes, duration_ns)")
    conn.executemany("INSERT INTO h2d_transfers VALUES ('db', 1, ?, ?)", transfers)
    return conn


class DatabaseTestCase(unittest.TestCase):
    """A test case with a temporary directory for its databases and generated files."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.tmp_path = Path(self._directory.name)


class BandwidthTest(unittest.TestCase):
    """The host-to-device bandwidth."""

    def test_bandwidth_is_total_bytes_over_total_time_of_large_copies(self) -> None:
        """Bandwidth is the byte-weighted harmonic mean of the rates of copies above the floor."""
        conn = transfers_database(
            [
                (GIB, ONE_SECOND),  # 1 GiB/s
                (4 * GIB, ONE_SECOND),  # 4 GiB/s
                (GIB, ONE_SECOND // 8),  # 8 GiB/s
                (1000, 1),  # below the size floor
            ]
        )
        # 6 GiB in 2.125 s.
        self.assertAlmostEqual(build.read_bandwidth_gibps(conn, 64 * build.MIB), 6 / 2.125)

    def test_a_slow_copy_pulls_the_bandwidth_down_by_the_time_it_takes(self) -> None:
        """The slow copy has a third of the bytes and most of the time, so the bandwidth follows
        it."""
        conn = transfers_database(
            [
                (10 * GIB, ONE_SECOND),  # 10 GiB/s
                (10 * GIB, ONE_SECOND),  # 10 GiB/s
                (10 * GIB, 10 * ONE_SECOND),  # 1 GiB/s
            ]
        )
        # 30 GiB in 12 s.
        self.assertAlmostEqual(build.read_bandwidth_gibps(conn, 64 * build.MIB), 2.5)

    def test_no_large_copy_exits_with_a_message(self) -> None:
        """Without a copy above the floor there is no bandwidth to derive."""
        conn = transfers_database([(1000, 1)])
        with self.assertRaisesRegex(SystemExit, "no host-to-device copy"):
            build.read_bandwidth_gibps(conn, 64 * build.MIB)


class PointsTest(DatabaseTestCase):
    """The SM and DE tables."""

    def test_de_table_takes_lz4_on_de_and_sm_table_any_backend(self) -> None:
        """The DE table takes only the `de` rows of LZ4; the SM table every backend."""
        samples = []
        for _ in range(6):
            samples += [
                sample("ans", "auto", 2.0, 1000.0),
                sample("bitpacking", "sm", 2.0, 300.0),
                sample("lz4", "de", 2.0, 200.0),
                sample("lz4", "sm", 2.0, 5.0),
            ]
        transfers = [{"bytes": GIB, "duration_ns": 10**7}]
        database = make_database(self.tmp_path / "cr_tp.db3", samples, transfers)
        conn = build.open_database([database])
        build.check_runs_are_valid(conn)
        build.create_decoded_samples_view(conn, 100 * build.MIB)
        sm_rows = build.read_points(conn, build.SM_WHERE, 5)
        de_rows = build.read_points(conn, build.DE_WHERE, 5)

        by_format = {row[1]: row for row in sm_rows}
        self.assertAlmostEqual(by_format["bitpacking"][3], 300.0)
        self.assertAlmostEqual(by_format["ans"][3], 1000.0)
        self.assertEqual(len(de_rows), 1)
        self.assertAlmostEqual(de_rows[0][3], 200.0)  # not the SM backend's 5

    def test_points_are_sorted_by_data_type_format_and_ratio(self) -> None:
        """read_points returns the points in the order the generated table lists them."""
        samples = []
        for nvcomp_data_type, fmt, ratio in [
            ("LONGLONG", "ans", 2.0),
            ("INT", "cascaded", 3.0),
            ("INT", "ans", 4.0),
            ("INT", "ans", 2.0),
            ("CHAR", "bitpacking", 2.0),
        ]:
            samples += [sample(fmt, "auto", ratio, 100.0, nvcomp_data_type=nvcomp_data_type)] * 5
        conn = build.open_database([make_database(self.tmp_path / "cr_tp.db3", samples, [])])
        build.create_decoded_samples_view(conn, 100 * build.MIB)
        self.assertEqual(
            [row[:3] for row in build.read_points(conn, build.SM_WHERE, 5)],
            [
                ("CHAR", "bitpacking", 2.0),
                ("INT", "ans", 2.0),
                ("INT", "ans", 4.0),
                ("INT", "cascaded", 3.0),
                ("LONGLONG", "ans", 2.0),
            ],
        )

    def test_a_bucket_below_the_sample_floor_is_dropped(self) -> None:
        """A (nvCOMP data type, format, ratio) bucket with fewer samples than the floor has no
        point."""
        samples = [sample("ans", "auto", 2.0, 1000.0)] * 4
        transfers = [{"bytes": GIB, "duration_ns": 10**7}]
        database = make_database(self.tmp_path / "cr_tp.db3", samples, transfers)
        conn = build.open_database([database])
        build.create_decoded_samples_view(conn, 100 * build.MIB)
        self.assertEqual(build.read_points(conn, build.SM_WHERE, 5), [])

    def test_overlapping_ranges_invalidate_the_run(self) -> None:
        """Overlapping ranges break the per-buffer attribution, so the run is refused."""
        samples = [sample("ans", "auto", 2.0, None, status="overlapping_ranges")]
        conn = build.open_database([make_database(self.tmp_path / "cr_tp.db3", samples, [])])
        with self.assertRaisesRegex(SystemExit, "invalid"):
            build.check_runs_are_valid(conn)


class SmAdvantageTest(DatabaseTestCase):
    """The SM advantage."""

    def test_advantage_ranks_de_time_over_the_fastest_sm_time(self) -> None:
        """The advantage is the given percentile of T_DE / T_SM over the buffers that both sides
        compress; a buffer the DE does not compress is not ranked."""
        bandwidth = 100.0  # GiB/s
        gib = BIG / GIB
        samples = []
        for column, (lz4_ratio, lz4_throughput, ans_ratio, ans_throughput) in {
            "a": (2.0, 100.0, 4.0, 400.0),
            "b": (2.0, 100.0, 2.0, 100.0),
            "c": (1.0, 100.0, 4.0, 400.0),
        }.items():
            for _ in range(3):
                samples += [
                    sample("lz4", "de", lz4_ratio, lz4_throughput, column=column),
                    sample("ans", "auto", ans_ratio, ans_throughput, column=column),
                ]
        conn = build.open_database([make_database(self.tmp_path / "cr_tp.db3", samples, [])])
        build.create_decoded_samples_view(conn, 100 * build.MIB)
        advantage, buffers = build.read_sm_advantage(conn, bandwidth, 25)

        def total(ratio: float, throughput: float) -> float:
            """Copy plus decode time of one buffer, in seconds."""
            return gib / ratio / bandwidth + gib / throughput

        ratios = sorted(
            [total(2.0, 100.0) / total(4.0, 400.0), total(2.0, 100.0) / total(2.0, 100.0)]
        )
        self.assertEqual(buffers, 2)
        self.assertAlmostEqual(advantage, ratios[0] + (ratios[1] - ratios[0]) * 0.25)
        # p0 sends every buffer to the SMs and p100 every buffer to the DE.
        self.assertAlmostEqual(build.read_sm_advantage(conn, bandwidth, 0)[0], ratios[0])
        self.assertAlmostEqual(build.read_sm_advantage(conn, bandwidth, 100)[0], ratios[1])


class RenderTest(unittest.TestCase):
    """The generated C++ source."""

    def test_render_names_the_profile_before_the_device_it_was_calibrated_on(self) -> None:
        """The profile carries its name, then the device it was calibrated on."""
        source = build.render(
            "gb300",
            "NVIDIA GB300",
            "a test",
            [("INT", "ans", 2.0, 1.0)],
            [("INT", "lz4", 2.0, 1.0)],
            50.0,
            1.5,
        )
        self.assertIn('"gb300", "NVIDIA GB300", de_points, sm_points, 50.0, 1.5', source)
        self.assertIn("calibration_profile const& gb300_profile()", source)


class RunInfoTest(DatabaseTestCase):
    """The GPU and the conditions of the runs."""

    def test_run_info_names_the_gpu_and_the_conditions_of_every_database(self) -> None:
        """The GPU comes from run_info, and the conditions of several databases are merged."""
        first = make_database(
            self.tmp_path / "a.db3",
            [],
            [],
            ("NVIDIA GB200", "node-a", "2026-10-07", "2026-10-07", "9f79dce1", "10000000"),
        )
        second = make_database(
            self.tmp_path / "b.db3",
            [],
            [],
            ("NVIDIA GB200", "node-b", "2026-10-09", "2026-10-09", "9f79dce1", "10000000"),
        )
        conn = build.open_database([first, second])
        run_info = build.read_run_info(conn, 2)
        self.assertEqual(
            run_info,
            {
                "gpu_name": "NVIDIA GB200",
                "hostnames": "node-a, node-b",
                "first_date": "2026-10-07",
                "last_date": "2026-10-09",
                "gqe_revisions": "9f79dce1",
                "zone_map_partition_sizes": "10000000",
            },
        )
        self.assertEqual(
            build.describe("NVIDIA GB200", run_info, 6, 100 * build.MIB, 50, "nvCOMP 5.3.0.16"),
            "NVIDIA GB200 measurements on node-a, node-b, 2026-10-07 to 2026-10-09, gqe 9f79dce1: "
            "p50 of one NVTX range per decompressed buffer of at least 100 MiB, 6 measured samples "
            "per point, SM advantage at p50, zone_map_partition_size 10000000, nvCOMP 5.3.0.16",
        )

    def test_databases_of_different_gpus_or_without_run_info(self) -> None:
        """Two GPUs cannot form one profile; a database without run_info has no GPU to name."""
        gb200 = make_database(
            self.tmp_path / "a.db3", [], [], ("NVIDIA GB200", "h", "d", "d", "r", "1")
        )
        b200 = make_database(
            self.tmp_path / "b.db3", [], [], ("NVIDIA B200", "h", "d", "d", "r", "1")
        )
        with self.assertRaisesRegex(SystemExit, "one GPU"):
            build.read_run_info(build.open_database([gb200, b200]), 2)

        old = make_database(self.tmp_path / "old.db3", [], [])
        self.assertIsNone(build.read_run_info(build.open_database([old]), 1))
        self.assertEqual(
            build.describe("NVIDIA GB200", None, 6, 100 * build.MIB, 25, None),
            "NVIDIA GB200 measurements: p50 of one NVTX range per decompressed buffer of at least "
            "100 MiB, 6 measured samples per point, SM advantage at p25",
        )


class MainTest(DatabaseTestCase):
    """The command line."""

    def calibration_database(self, sample_count: int) -> Path:
        """A database with `sample_count` samples of one buffer in LZ4 on the DE and ANS on the
        SMs, one large copy and run_info."""
        samples = []
        for _ in range(sample_count):
            samples += [sample("lz4", "de", 2.0, 100.0), sample("ans", "auto", 4.0, 400.0)]
        return make_database(
            self.tmp_path / "cr_tp.db3",
            samples,
            [{"bytes": GIB, "duration_ns": ONE_SECOND}],
            ("NVIDIA GB200", "node-a", "2026-10-07", "2026-10-07", "9f79dce1", "10000000"),
        )

    def run_main(self, *arguments: str) -> str:
        """Run main() with `arguments`, writing into the temporary directory; return its output."""
        argv = ["build_calibration_tables.py", *arguments, "--output-dir", str(self.tmp_path)]
        output = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
            build.main()
        return output.getvalue()

    def test_main_writes_the_profile_named_after_the_profile(self) -> None:
        """main() writes <profile>.cpp with the profile and reports what it derived."""
        database = self.calibration_database(6)
        output = self.run_main(str(database), "--profile", "gb200_test")

        source = (self.tmp_path / "gb200_test.cpp").read_text()
        self.assertIn("calibration_profile const& gb200_test_profile()", source)
        self.assertIn('"gb200_test", "NVIDIA GB200", de_points, sm_points, 1.0,', source)
        self.assertIn("SM advantage at p50", source)
        self.assertIn("wrote 1 SM and 1 DE points", output)

    def test_main_rejects_a_profile_name_that_is_not_an_identifier(self) -> None:
        """The profile name becomes a C++ identifier, so other names are refused."""
        database = self.calibration_database(6)
        with self.assertRaisesRegex(SystemExit, "lower case C\\+\\+ identifier"):
            self.run_main(str(database), "--profile", "GB200-NVL4")

    def test_main_exits_when_no_bucket_reaches_the_sample_floor(self) -> None:
        """With fewer samples than --min-samples there is no point to write."""
        database = self.calibration_database(3)
        with self.assertRaisesRegex(SystemExit, "no bucket reached 5 samples"):
            self.run_main(str(database), "--profile", "gb200_test")
        self.assertFalse((self.tmp_path / "gb200_test.cpp").exists())


if __name__ == "__main__":
    unittest.main()
