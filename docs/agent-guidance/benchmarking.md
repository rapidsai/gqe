# Benchmarking guidance

Use [gqe-bench](https://gitlab-master.nvidia.com/Devtech-Compute/gqe-bench) for
the supported benchmark workflow. Do not assume its implementation files exist
in this repository.

Benchmark A/B claims must:

- keep all settings identical except the behavior under test;
- include several measured repeats;
- validate every result against the reference answer;
- measure memory footprint from the live engine's `_gqe_column_stats` and
  `_gqe_table_stats` tables; and
- state what the measurements do not establish.

For TPC-H, use the canonical 22 queries, per-scale-factor Parquet datasets, and
per-scale-factor reference answers. Check scale-dependent literals in query
text. In particular, Q11's fraction is `0.0001 / SF`.
