# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import decimal
import pyarrow.parquet as pq
import sys
from argparse import ArgumentParser
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype, is_string_dtype
from pandas.testing import assert_frame_equal, assert_series_equal

ABS_TOLERANCE = 1e-6

# FIXME Find a tighter bounds.
ABS_DECIMAL_TOLERANCE = 1e-2


def _is_decimal_series(col: pd.Series) -> bool:
    if col.dtype != object:
        return False
    return col.map(lambda v: isinstance(v, decimal.Decimal)).any()


def _is_missing_value(v) -> bool:
    return v is None or pd.isna(v)


def gqe_assert_decimal_series_equal(lcol: pd.Series, rcol: pd.Series, atol: float):
    left_values = lcol._values
    right_values = rcol._values

    first_different = -1
    num_different = 0
    for i, row in enumerate(left_values):
        left = left_values[i]
        right = right_values[i]
        if _is_missing_value(left) and _is_missing_value(right):
            continue
        if _is_missing_value(left) != _is_missing_value(right):
            num_different += 1
            if first_different == -1:
                first_different = i
            continue
        if abs(left - right) > decimal.Decimal(atol):
            num_different += 1
            if first_different == -1:
                first_different = i

    if num_different > 0:
        # emulate assert_series_equal failure output
        percent_diff = float(num_different) / float(left_values.size) * 100.0
        diff_str = "\nSeries values are different (" + str(percent_diff) + " %)\n"
        indices_str = "[index]: ["
        left_str = "[left]:  ["
        right_str = "[right]: ["
        for i, row in enumerate(left_values):
            indices_str += str(i)
            indices_str += ", "
            left_str += str(left_values[i])
            left_str += ", "
            right_str += str(right_values[i])
            right_str += ", "
        indices_str += ']\n'
        left_str += ']\n'
        right_str += ']\n'
        vals_str = "At positional index " + str(first_different) + ", first diff: " + str(
            left_values[first_different]) + " != " + str(right_values[first_different])
        output_str = diff_str + indices_str + left_str + right_str + vals_str
        raise Exception(output_str)


def gqe_assert_frame_equal(lhs: pd.DataFrame, rhs: pd.DataFrame, atol: float):
    for i, col in enumerate(lhs.columns):
        lcol = lhs._ixs(i, axis=1)
        rcol = rhs._ixs(i, axis=1)

        if _is_decimal_series(lcol) or _is_decimal_series(rcol):
            # use our path since pandas doesn't support approximate decimal comparison
            gqe_assert_decimal_series_equal(lcol, rcol, atol=ABS_DECIMAL_TOLERANCE)
        else:
            # use pandas for everything else
            assert_series_equal(lcol, rcol, atol=atol)


def normalize_type(df1: pd.DataFrame, df2: pd.DataFrame, col: str):
    old_type = df2[col].dtypes.name
    new_type = df1[col].dtypes.name

    # Single-char (ASCII) strings can be compared to their integer representation
    if is_numeric_dtype(df1[col]) and is_string_dtype(df2[col]):
        all_single_char = df2[col].apply(lambda x: len(x) == 1).all()
        if (not all_single_char):
            raise Exception("Can only convert single-char (ASCII) strings to numeric type")
        df2[col] = df2[col].apply(lambda x: ord(x) if isinstance(x, str) and len(x) == 1 else None)

    df2[col] = df2[col].astype(new_type)


def verify(test_file: str, ref_file: str):
    df_gqe = pq.read_table(test_file, use_threads=False).to_pandas(use_threads=False)
    df_ref = pq.read_table(ref_file, use_threads=False).to_pandas(use_threads=False)
    print("Start verification...")

    # normalize column names
    df_gqe.columns = df_ref.columns
    # normalize column types
    for col in df_gqe.columns:
        if is_numeric_dtype(df_gqe[col]) and not is_numeric_dtype(df_ref[col]):
            # if only one column is numeric, covert to numeric type
            normalize_type(df_gqe, df_ref, col)
        elif not is_numeric_dtype(df_gqe[col]) and is_numeric_dtype(df_ref[col]):
            # if only one column is numeric, covert to numeric type
            normalize_type(df_ref, df_gqe, col)
        elif is_float_dtype(df_gqe[col]) and not is_integer_dtype(df_ref[col]):
            # if only one floating point
            normalize_type(df_gqe, df_ref, col)
        elif is_numeric_dtype(df_gqe[col]) and is_numeric_dtype(df_ref[col]):
            # if both are numeric convert to larger type
            gqe_col_item_size = df_gqe[col].dtype.itemsize
            ref_col_item_size = df_ref[col].dtype.itemsize
            if gqe_col_item_size > ref_col_item_size:
                normalize_type(df_gqe, df_ref, col)
            elif gqe_col_item_size < ref_col_item_size:
                normalize_type(df_ref, df_gqe, col)

    # Verify that GQE result is the same as the reference result
    gqe_assert_frame_equal(lhs=df_gqe, rhs=df_ref, atol=ABS_TOLERANCE)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("test", help="result generated by GQE in Parquet format")
    arg_parser.add_argument("reference", help="reference result in Parquet format")
    args = arg_parser.parse_args()

    verify(args.test, args.reference)


if __name__ == "__main__":
    main()
