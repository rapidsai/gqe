#!/bin/bash

# Treat unset variables as an error and exit immediately.
set -u

# Check if the correct number of arguments is provided.
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <tpc_kind> <build_dir> <dataset_dir> <reference_dir_or_file> <substrait_file_or_pattern>"
  exit 1
fi

# Convert TPC kind to uppercase (for better matching) and assign arguments to variables.
tpc_kind="${1^^}"
build_dir="$2"
dataset_dir="$3"
reference_path="$4"
substrait_pattern="$5"

# Define the query lists for TPC-H and TPC-DS.
tpc_h_queries=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22)
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

# Counters for total and failed runs and verifications.
total_verifications=0
failed_verifications=0
total_runs=0
failed_runs=0

# Function to run a test.
run_test() {
  local substrait_plan="$1" reference_file="$2"
  if "$build_dir"/benchmark/tpc "${tpc_kind,,}" "$substrait_plan" "$dataset_dir" "parquet_file" ; then
    echo "Run SUCCESS: $substrait_plan"
    # Extract output file name
    substrait_name=$(basename $substrait_plan | cut -d. -f1)
    output_file="output_${substrait_name}.parquet"
    ((total_verifications++))
    if [ -v output_file ]; then
      # Verify results
      if python $(dirname "$0")/verify_parquet.py "$output_file" "$reference_file"; then
        echo "Verification SUCCESS: $output_file vs. $reference_file"
      else
        echo "Verification FAILED: $output_file vs. $reference_file" >&2
        ((failed_verifications++))
      fi
    else
      echo "Unable to extract output filename from program output"
      ((failed_verifications++))
    fi
  else
    echo "Run FAILED: $substrait_plan" >&2
    ((failed_runs++))
  fi
}

# Check if substrait pattern has a format specifier (%d) and act accordingly.
if [[ $substrait_pattern == *"%d"* ]]; then
  for query in "${queries[@]}"; do
    ((total_runs++))
    # Generate the filename based on the pattern.
    substrait_plan=$(printf "$substrait_pattern" "$query")
    reference_file="${reference_path}/q${query}.parquet"
    if [ -f "$substrait_plan" ]; then
      run_test "$substrait_plan" "$reference_file"
    else
      echo "Error: File does not exist: $substrait_plan" >&2
      ((failed_runs++))
    fi
  done
else
  ((total_runs++))
  # Run a single substrait plan.
  if [ -f "$substrait_pattern" ]; then
    run_test "$substrait_pattern" "$reference_path"
  else
    echo "Error: File does not exist: $substrait_pattern" >&2
    ((failed_runs++))
  fi
fi

# Display test results.
if [ $failed_runs -eq 0 ]; then
  echo "All $total_runs plans ran SUCCESSFULLY"
else
  echo "$failed_runs/$total_runs plans FAILED to run"
fi

if [ $failed_verifications -eq 0 ]; then
  echo "All $total_verifications verifications PASSED"
else
  echo "$failed_verifications/$total_verifications verifications FAILED"
fi

# Exit with the number of failed tests.
exit $(($failed_runs + $failed_verifications))
