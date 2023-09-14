#!/bin/bash

# Treat unset variables as an error and exit immediately.
set -u

# Check if the correct number of arguments is provided.
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <tpc_kind> <build_dir> <dataset_dir> <substrait_file_or_pattern>"
  exit 1
fi

# Convert TPC kind to uppercase (for better matching) and assign arguments to variables.
tpc_kind="${1^^}"
build_dir="$2"
dataset_dir="$3"
substrait_pattern="$4"

# Define the query lists for TPC-H and TPC-DS.
tpc_h_queries=(2 3 5 6 7 8 9 10 12 14 15 17 18 19 20)
tpc_ds_queries=()

# Select the correct query list based on the tpc_kind.
case "$tpc_kind" in
    "H")
        queries=("${tpc_h_queries[@]}")
        ;;
    "DS")
        queries=("${tpc_ds_queries[@]}")
        ;;
    *)
        echo "Error: TPC-$tpc_kind is not supported." >&2
        exit 1
        ;;
esac

# Check if the build and dataset directories exist.
for dir in "$build_dir" "$dataset_dir"; do
  if [ ! -d "$dir" ]; then
    echo "Error: Directory does not exist: $dir" >&2
    exit 1
  fi
done

# Counters for total tests and failed tests.
total_tests=0
failed_tests=0

# Function to run a test.
run_test() {
  local substrait_plan="$1"
  if "$build_dir"/benchmark/tpc "${tpc_kind,,}" "$substrait_plan" "$dataset_dir" "parquet_file"; then
    echo "Test SUCCESS: $substrait_plan"
  else
    echo "Test FAILED: $substrait_plan" >&2
    ((failed_tests++))
  fi
}

# Check if substrait pattern has a format specifier (%d) and act accordingly.
if [[ $substrait_pattern == *"%d"* ]]; then
  for query in "${queries[@]}"; do
    ((total_tests++))
    # Generate the filename based on the pattern.
    substrait_plan=$(printf "$substrait_pattern" "$query")
    if [ -f "$substrait_plan" ]; then
      run_test "$substrait_plan"
    else
      echo "Error: File does not exist: $substrait_plan" >&2
      ((failed_tests++))
    fi
  done
else
  ((total_tests++))
  # Run a single substrait plan.
  if [ -f "$substrait_pattern" ]; then
    run_test "$substrait_pattern"
  else
    echo "Error: File does not exist: $substrait_pattern" >&2
    ((failed_tests++))
  fi
fi

# Display test results.
if [ $failed_tests -eq 0 ]; then
  echo "All $total_tests tests PASSED"
else
  echo "$failed_tests/$total_tests tests FAILED"
fi

# Exit with the number of failed tests.
exit $failed_tests
