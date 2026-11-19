# Build, test, and tooling guidance

Follow the canonical setup, build, and test instructions in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#setting-up-your-build-environment).

- For a focused incremental build, use `ninja -C <build-dir> <target>`.
- Treat a successful Ninja build as authoritative when clangd reports `_mm_getcsr`
  errors or stale-index diagnostics under a conda toolchain.
- Follow the [documented pre-commit workflow](../../CONTRIBUTING.md#pre-commit-hooks).
  Formatting hooks may modify files and fail the commit. Restage those files,
  rerun the hooks, and retry the commit.
- Set `GQE_LOG_LEVEL=trace` when verbose engine output is needed.
- GQE boolean environment variables accept `true`, `false`, `on`, `off`, `yes`,
  and `no`. They do not accept `1` as true.
