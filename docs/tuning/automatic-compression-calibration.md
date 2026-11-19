# Calibrating automatic compression selection for a GPU

Automatic compression selection scores each representation of a buffer by the time to copy its compressed bytes to the device plus the time to decode them. The decode time comes from throughput measured on the GPU in use. This page describes how to measure it for a new GPU and what each calibrated value means. All throughputs and bandwidths are in GiB/s.

A calibration profile (`calibration_profile` in `include/gqe/optimizer/decode_throughput.hpp`) holds four things: the SM decode table, the decompression engine (DE) table (empty on a GPU without a DE), the host-to-device copy bandwidth, and the SM advantage. The SM advantage is a multiplier: selection picks an SM format over the fastest DE format only if the SM format is estimated to be at least that many times faster. It accounts for the query kernels that compete with SM decompression for SM cycles, which the cost model does not charge. A profile describes a GPU together with how it is connected to its host, because the bandwidth depends on the connection. The same GPU in another system can need its own profile.

A profile has a name, the `--profile` value of the generator such as `gb300`, and the device name prefix of the GPU it was calibrated on. A session selects its profile by name. Nothing selects a profile from the device name, because the name of a GPU does not tell which system it is in and the same GPU in another system can need another profile, with another name and the same device name. The device name prefix (`calibrated_on`) is only compared with the GPU in use, to tell whether a session uses the profile of another GPU.

## Calibrating a new profile

1. Extract the columns of the dataset and run the decode benchmark of gqe-bench on the target system. Use `--repeat 7` so that each point has six measured samples after the warm-up. Set `zone_map_partition_size` to the value of the configuration the profile is for (our benchmarks use 10,000,000): GQE compresses each zone map partition separately, so the partition size affects the compression ratio and the decode throughput. See `docs/decompression_benchmark.md` in gqe-bench.

   ```bash
   gqe-decompression-extract --dataset-dir <tpch-sf1000> --output-dir <out>/data
   gqe-decompression-bench --data-dir <out>/data --output-dir <out> --repeat 7
   ```

2. The benchmark writes `<out>/report/cr_tp.db3`, a SQLite database with the table `raw_samples` (one row per buffer, format, backend and repetition), the table `h2d_transfers` (every host-to-device copy CUPTI recorded) and the one-row table `run_info` (the GPU, hosts, dates, GQE revision and zone map partition size of the runs). Pass it to the generator. When formats were measured in separate runs, pass every database.

   ```bash
   scripts/calibration/build_calibration_tables.py <out>/report/cr_tp.db3 --profile gb300
   ```

   `--profile` is the only required option: the name sessions select the profile by. Give two systems with the same GPU different names. The profile's `calibrated_on` is the GPU in `run_info`, and the header of the generated file names the GPU, hosts, dates, GQE revision, zone map partition size, buffer floor and samples per point. Using the `--provenance` flag, you can append a note such as the dataset or the nvCOMP version. A database written before `run_info` existed needs `--calibrated-on`, which also overrides the GPU name.

   The script writes `src/optimizer/calibration/<profile>.cpp` and prints the bandwidth and the SM advantage it derived. It computes percentiles with SQLite's [`percentile_cont`](https://sqlite.org/percentile.html), which only SQLite builds that include the percentile extension provide, and many system packages do not. The Python of the GQE build container (`/conda/envs/gqe`) has it. With another Python, the script exits with a message if its SQLite lacks the function.

3. Declare the accessor in `src/optimizer/calibration/tables.hpp`, add the file to `src/CMakeLists.txt`, and append the profile to `calibration_profiles()` in `src/optimizer/decode_throughput.cpp`.

4. Add the profile to the names that `ProfileIsFoundByName` and `UnknownProfileNameIsAnErrorListingTheProfiles` in `test/optimizer/decode_throughput_test.cpp` expect.

A GPU without a DE has no LZ4 rows on the `de` backend, so the generator has nothing to put in the DE table for it. Decide that case before generating.

After changing the generator, run its tests with the Python of the GQE build container: `python -m unittest discover -s scripts/calibration -p "test_*.py"`, or `ctest -R calibration_generator_test` in a build directory.

## Calibration methodology

The calculations are SQL over the tables of the database. We write pXX for the XX-th percentile: the value that XX% of the inputs are at or below.

- **SM table.** Samples are bucketed by nvCOMP data type, format and compression ratio rounded to four decimals, and a bucket needs at least five samples (`--min-samples`). Each bucket keeps the p50 of its throughput.
- **DE table.** The p50 of the LZ4 rows of the `de` backend, bucketed the same way.
- **Bandwidth.** The total bytes over the total time of the host-to-device copies of at least 64 MiB (`--min-h2d-bytes`). This is the byte-weighted harmonic mean of the copy rates. See [Why the bandwidth is total bytes over total time](#why-the-bandwidth-is-total-bytes-over-total-time).
- **SM advantage.** See [The SM advantage](#the-sm-advantage).

At runtime a representation's decode throughput is the point of its (nvCOMP data type, format) curve whose compression ratio is nearest to its own. The lookup does not interpolate, so on a sparse curve the estimate steps at the midpoint between two calibrated ratios. Only CHAR, INT and LONGLONG are calibrated; a column of another nvCOMP type, such as SHORT or an unsigned type, has no curve, gets no automatic candidate and is stored uncompressed.

The generator drops buffers that did not compress, and buffers below 100 MiB (`--min-bytes`). A small buffer's decode time is dominated by fixed launch overhead: at about 38 MiB a format reaches only a small fraction of its large-buffer throughput. The buckets group buffers by compression ratio, not by size, so small buffers would pull down the throughput of the large buffers in the same bucket. A run with `overlapping_ranges` samples is invalid, because the per-buffer attribution of CUPTI activity does not hold.

## The SM advantage

A buffer can be stored in an SM format or in the DE format LZ4. The cost model estimates the time of each as copy time plus decode time, with `B` the bandwidth. For each measured buffer the generator computes both from that buffer's own measurements:

```text
T_DE = compressed_bytes_lz4 / B + uncompressed_bytes / p50 DE throughput of lz4
T_SM = min over the SM formats of ( compressed_bytes / B + uncompressed_bytes / p50 SM throughput )
```

`T_DE / T_SM` is the speedup of the SM format over the DE format. Selection picks the SM format only if this speedup exceeds `sm_advantage`. The generator sets `sm_advantage` to a percentile of `T_DE / T_SM` over the measured buffers, linearly interpolated between ranked values. The percentile is `--sm-advantage-percentile`, by default p50. Only buffers that both a DE format and an SM format compress are ranked, because selection compares the winner with the uncompressed copy separately and without the advantage.

The percentile is the most important tuning parameter of the profile, because it decides how the measured buffers split between the DE and the SMs: at pXX, about XX% of the buffers stay on the DE, namely those where the SMs gain the least, and the rest go to the SMs. At p50 half of the buffers stay on the DE; p0 would send every buffer to the SMs, and p100 every buffer to the DE. Keeping part of the work on the DE keeps it busy alongside the SMs, instead of sending everything to the SMs, where decoding competes with query kernels. Leaving the buffers with the smallest SM gains on the DE gives up the least measured time. A higher percentile leaves buffers with larger SM gains on the DE, which has a fixed throughput ceiling. On the GB200, p50 performed better than p25 in our end-to-end runs. The value is stored with the profile, and a session can override it.

The bandwidth is part of the SM advantage, because `T_DE` and `T_SM` depend on it. Always derive both from the same database, which the generator does.

## Why the bandwidth is total bytes over total time

The cost model charges a buffer `compressed_bytes / B` for its copy. In an end-to-end query the copies add up: what is observed is the sum of the transfer times, `sum(bytes_i / rate_i)`. The sum of the model's charges, `sum(bytes_i) / B`, equals it only for `B = sum(bytes_i) / sum(bytes_i / rate_i)`, the total bytes over the total time. That is the byte-weighted harmonic mean of the rates, which is also what a weighted average of throughputs has to be when the quantity that adds up is time.

A slow copy takes a large share of the total time, and the total-bytes-over-total-time bandwidth follows it, because it costs the same time in a query. Copies below `--min-h2d-bytes` are left out because a small copy is dominated by its launch cost and does not describe the link.
