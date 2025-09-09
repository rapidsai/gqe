/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/executor/like.hpp>
#include <gqe/utility/error.hpp>
#include <gqe/utility/helpers.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <type_traits>
#include <vector>

namespace gqe {

namespace detail {

constexpr char multi_wildcard  = '%';
constexpr char single_wildcard = '_';
// uint32_t has 4 bytes (shift 2), for each char, we need to shift 8,
// but for each uint32_t, we need to shift 6
constexpr int shift_with_uint32 = 6;
// meaning it's asc only with 256 chars
constexpr int charset_bits = 8;
constexpr int num_chars    = 256;

using size_t = std::size_t;

template <typename MASK_TYPE>
__device__ bool check_multi_pattern_match(char const* string_ptr,
                                          size_t string_len,
                                          MASK_TYPE const* pattern_masks,
                                          size_t const* pattern_lengths,
                                          size_t num_middle_patterns,
                                          size_t total_pattern_len)
{
  static_assert(std::is_integral<MASK_TYPE>::value, "MASK_TYPE must be an integral type");
  if (num_middle_patterns == 0) return true;
  MASK_TYPE state       = 0;
  MASK_TYPE const* mask = pattern_masks;
  size_t pattern_num    = 0;  // 0-indexed counter of pattern number
  size_t pattern_len    = pattern_lengths[0];
  MASK_TYPE final_state = 1 << (pattern_len - 1);
  int farthest_start =
    string_len -
    total_pattern_len;  // If we have no pattern overlap and are at this position, we can exit early

  if (string_len >= total_pattern_len) {
    for (size_t i = 0; i < string_len; i++) {
      state        = (state << 1) + 1;
      char const c = string_ptr[i];
      state        = state & mask[(uint8_t)c];
      if ((state & final_state) !=
          0) {  // Matched pattern so switching state and masks to next pattern
        farthest_start +=
          pattern_len;  // Push farthest start by the length of the pattern we just matched to
        pattern_num++;
        if (pattern_num == num_middle_patterns) { return true; }
        // Switch to probing next pattern
        state       = 0;
        pattern_len = pattern_lengths[pattern_num];
        final_state = 1 << (pattern_len - 1);  // reset for WORD2
        mask += num_chars;                     // each mask is 256 elements
      }

      if (i > farthest_start && (state == 0))
        break;  // Exit early if not possible to complete pattern in remaining characters
    }
  }

  return false;
}

__device__ bool check_exact_pattern_match(char const* string_ptr,
                                          char const* pattern,
                                          size_t pattern_len)
{
  for (size_t i = 0; i < pattern_len; i++) {
    char c = string_ptr[i];
    char p = pattern[i];
    if (p != single_wildcard && c != p) { return false; }
  }
  return true;
}

template <typename MASK_TYPE>
__global__ void like_fn_shift_and_kernel(cudf::column_device_view d_strings,
                                         cudf::string_view const d_prefix,
                                         cudf::string_view const d_suffix,
                                         MASK_TYPE const* middle_masks,
                                         size_t const* middle_lengths,
                                         size_t num_middle_patterns,
                                         size_t total_pattern_len,
                                         bool* results)
{
  static_assert(std::is_integral<MASK_TYPE>::value, "MASK_TYPE must be an integral type");
  extern __shared__ unsigned char smem[];

  MASK_TYPE* s_middle_masks = (MASK_TYPE*)smem;
  size_t mask_table_size    = (num_middle_patterns << charset_bits) * sizeof(MASK_TYPE);
  size_t* s_middle_lengths  = (size_t*)(smem + mask_table_size);

  const int tid  = threadIdx.x + blockIdx.x * blockDim.x;
  bool is_active = tid < d_strings.size() && !d_strings.is_null(tid);

  bool result = is_active;

  const char* string_ptr = nullptr;
  size_t string_len      = 0;
  if (is_active) {
    auto const d_str = d_strings.element<cudf::string_view>(tid);
    string_ptr       = d_str.data();
    string_len       = d_str.length();
  }

  // Check for prefix
  size_t prefix_len = d_prefix.length();
  if (prefix_len > 0) {
    char const* prefix = d_prefix.data();
    if (result && (string_len >= prefix_len)) {
      result = check_exact_pattern_match(string_ptr, prefix, prefix_len);
    }
  }

  // Check for suffix
  size_t suffix_len = d_suffix.length();
  if (suffix_len > 0) {
    char const* suffix = d_suffix.data();
    if (result && ((string_len - prefix_len) >= suffix_len)) {
      result = check_exact_pattern_match(string_ptr + string_len - suffix_len, suffix, suffix_len);
    }
  }

  // Check for middle patterns between the prefix & suffix
  if (num_middle_patterns > 0) {
    // Read masks into shared memory
    // Check if already matched prefix & suffix
    uint32_t int32_num_per_mask = sizeof(MASK_TYPE) << shift_with_uint32;
    uint32_t total_int32_read   = int32_num_per_mask * num_middle_patterns;
    for (int i = threadIdx.x; i < total_int32_read; i += blockDim.x) {
      reinterpret_cast<uint32_t*>(s_middle_masks)[i] =
        reinterpret_cast<uint32_t const*>(middle_masks)[i];
    }
    for (int i = threadIdx.x; i < num_middle_patterns; i += blockDim.x) {
      s_middle_lengths[i] = middle_lengths[i];
    }
    __syncthreads();
    if (result) {
      result = check_multi_pattern_match(string_ptr + prefix_len,
                                         string_len - prefix_len - suffix_len,
                                         s_middle_masks,
                                         s_middle_lengths,
                                         num_middle_patterns,
                                         total_pattern_len);
    }
  }

  // Update bitmap when active and the string is not null
  if (is_active) { results[tid] = result; }
}

template <typename MASK_TYPE>
void generate_shift_and_mask_table(std::string const& pattern, MASK_TYPE* out_mask)
{
  static_assert(std::is_integral<MASK_TYPE>::value, "MASK_TYPE must be an integral type");
  for (size_t pattern_idx = 0; pattern_idx < pattern.length(); pattern_idx++) {
    char c = pattern[pattern_idx];
    if (c == single_wildcard) {
      for (size_t i = 0; i < num_chars; i++) {
        out_mask[i] = out_mask[i] | (1 << pattern_idx);
      }
    } else {
      out_mask[(uint8_t)c] = out_mask[(uint8_t)c] | (1 << pattern_idx);
    }
  }
}

template <typename MASK_TYPE>
void run_string_match(cudf::strings_column_view const& input,
                      std::vector<std::string> const& middle_patterns,
                      std::string const& prefix,
                      std::string const& suffix,
                      cudf::mutable_column_view results,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  static_assert(std::is_integral<MASK_TYPE>::value, "MASK_TYPE must be an integral type");
  std::unique_ptr<std::vector<MASK_TYPE>> middle_masks = nullptr;

  size_t num_middle_patterns = middle_patterns.size();
  bool has_prefix            = prefix.length() > 0;
  bool has_suffix            = suffix.length() > 0;
  bool has_middles           = num_middle_patterns > 0;

  if (has_middles) {
    // here num_chars=256 means the character size is 256, only applicable for ASCII-like character
    // set not applicable for Unicode charset
    middle_masks = std::make_unique<std::vector<MASK_TYPE>>(num_chars * num_middle_patterns, 0);
    for (size_t i = 0; i < num_middle_patterns; i++) {
      generate_shift_and_mask_table(middle_patterns[i], middle_masks->data() + num_chars * i);
    }
  }

  std::unique_ptr<rmm::device_buffer> d_prefix_buf = nullptr;
  std::unique_ptr<rmm::device_buffer> d_suffix_buf = nullptr;
  char* d_prefix                                   = nullptr;
  char* d_suffix                                   = nullptr;
  if (has_prefix) {
    d_prefix_buf = std::make_unique<rmm::device_buffer>(prefix.length(), stream, mr);
    d_prefix     = reinterpret_cast<char*>(d_prefix_buf->data());
    GQE_CUDA_TRY(cudaMemcpy(d_prefix, prefix.data(), prefix.length(), cudaMemcpyDefault));
  }
  if (has_suffix) {
    d_suffix_buf = std::make_unique<rmm::device_buffer>(suffix.length(), stream, mr);
    d_suffix     = reinterpret_cast<char*>(d_suffix_buf->data());
    GQE_CUDA_TRY(cudaMemcpy(d_suffix, suffix.data(), suffix.length(), cudaMemcpyDefault));
  }

  // get the length for each middle pattern and put them in middle_lengths
  std::unique_ptr<std::vector<size_t>> middle_lengths = nullptr;
  if (has_middles) {
    middle_lengths = std::make_unique<std::vector<size_t>>(num_middle_patterns, 0);
  }
  size_t total_pattern_len = 0;
  for (size_t i = 0; i < num_middle_patterns; i++) {
    size_t len           = middle_patterns[i].length();
    (*middle_lengths)[i] = len;
    total_pattern_len += (*middle_lengths)[i];
  }

  // ******************* EXECUTE STRING MATCHING KERNEL ******************* //

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  std::unique_ptr<rmm::device_uvector<MASK_TYPE>> d_middle_masks = nullptr;
  std::unique_ptr<rmm::device_uvector<size_t>> d_middle_lengths  = nullptr;
  if (has_middles) {
    d_middle_masks =
      std::make_unique<rmm::device_uvector<MASK_TYPE>>(num_chars * num_middle_patterns, stream, mr);
    GQE_CUDA_TRY(cudaMemcpy(d_middle_masks->data(),
                            middle_masks->data(),
                            middle_masks->size() * sizeof(MASK_TYPE),
                            cudaMemcpyDefault));

    d_middle_lengths =
      std::make_unique<rmm::device_uvector<size_t>>(num_middle_patterns, stream, mr);
    GQE_CUDA_TRY(cudaMemcpy(d_middle_lengths->data(),
                            middle_lengths->data(),
                            middle_lengths->size() * sizeof(size_t),
                            cudaMemcpyDefault));
  }

  size_t block_size = 256;
  size_t grid_size  = gqe::utility::divide_round_up(d_strings->size(), block_size);
  size_t smem_size =
    num_middle_patterns * sizeof(MASK_TYPE) * num_chars + sizeof(size_t) * num_middle_patterns;
  // performs the like pattern matching on a single string per thread
  like_fn_shift_and_kernel<<<grid_size, block_size, smem_size, stream>>>(
    *d_strings,
    d_prefix != nullptr ? cudf::string_view{d_prefix, (cudf::size_type)prefix.length()}
                        : cudf::string_view{},
    d_suffix != nullptr ? cudf::string_view{d_suffix, (cudf::size_type)suffix.length()}
                        : cudf::string_view{},
    d_middle_masks != nullptr ? d_middle_masks->data() : nullptr,
    d_middle_lengths != nullptr ? d_middle_lengths->data() : nullptr,
    num_middle_patterns,
    total_pattern_len,
    results.data<bool>());
}

// Check if the max length of the middle patterns is <= 64,
// Basically, this function is to find the prefix, suffix as well as strings in the middle
// e.g. hello%world%good%morning: the prefix is "hello", suffix is "morning"
// and strings in the middle are ["world", "good"]
// TODO: Future optimization is to use the smallest possible table size for each pattern.
void preprocess_like(std::string const& pattern,
                     std::string& prefix,
                     std::string& suffix,
                     std::vector<std::string>& middle_patterns,
                     size_t& max_mid_pattern_len,
                     char const escape_character)
{
  size_t const pattern_len = pattern.length();

  std::vector<std::string> all_patterns;
  // for holding the temporary string that's processed in the middle
  std::string tmp_string;

  bool escaped = false;
  size_t curr  = 0;
  while (curr < pattern_len) {
    char curr_char = pattern[curr];
    escaped        = (curr_char == escape_character);
    // get the next char by ignoring the escaped char
    auto const pattern_char = escaped && (curr < pattern_len - 1) ? pattern[++curr] : curr_char;

    if (escaped || pattern_char != multi_wildcard) {
      // push back the pattern_char into tmp_string when
      // 1) its previous char is escaped char
      // or 2) its previous char isn't escaped char and it's not multi_wildcard
      tmp_string.push_back(pattern_char);
      if (curr == pattern_len - 1) { all_patterns.push_back(tmp_string); }
    } else if (tmp_string.length() > 0) {
      all_patterns.push_back(tmp_string);
      tmp_string = "";
    }

    curr++;
  }

  if (all_patterns.size() > 0 && pattern[0] != multi_wildcard) {
    prefix = all_patterns[0];
    all_patterns.erase(all_patterns.begin());
  }

  if (all_patterns.size() > 0 &&
      (pattern[pattern_len - 1] != multi_wildcard ||
       (pattern_len > 1 && pattern[pattern_len - 2] == escape_character))) {
    suffix = all_patterns[all_patterns.size() - 1];
    all_patterns.pop_back();
  }

  for (auto pattern : all_patterns) {
    max_mid_pattern_len = max(max_mid_pattern_len, pattern.length());
  }

  middle_patterns =
    all_patterns;  // set middle_patterns to all the patterns, minus the prefix and suffix
}

}  // end of namespace detail

std::unique_ptr<cudf::column> like(cudf::strings_column_view const& input,
                                   std::string const& pattern,
                                   cudf::string_scalar const& escape_character,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  // sanity check just like cudf::strings::like
  CUDF_EXPECTS(escape_character.is_valid(stream),
               "Parameter escape_character must be valid",
               std::invalid_argument);

  auto const d_escape = escape_character.value(stream);
  CUDF_EXPECTS(d_escape.size_bytes() <= 1,
               "Parameter escape_character must be a single character",
               std::invalid_argument);

  auto results = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                           input.size(),
                                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                           input.null_count(),
                                           stream,
                                           mr);
  if (input.is_empty()) { return results; }

  // preprocess the pattern
  std::string prefix = "";
  std::string suffix = "";
  std::vector<std::string> middle_patterns;
  size_t max_mid_pattern_len = 0;
  char const esc_char        = d_escape.empty() ? 0 : d_escape.data()[0];
  // extract the pattern into prefix, suffix, and middle patterns here
  gqe::detail::preprocess_like(
    pattern, prefix, suffix, middle_patterns, max_mid_pattern_len, esc_char);

  // run string match based on different max length of mid pattern strings
  if (max_mid_pattern_len <= 8) {
    gqe::detail::run_string_match<uint8_t>(
      input, middle_patterns, prefix, suffix, results->mutable_view(), stream, mr);
  } else if (max_mid_pattern_len <= 16) {
    gqe::detail::run_string_match<uint16_t>(
      input, middle_patterns, prefix, suffix, results->mutable_view(), stream, mr);
  } else if (max_mid_pattern_len <= 32) {
    gqe::detail::run_string_match<uint32_t>(
      input, middle_patterns, prefix, suffix, results->mutable_view(), stream, mr);
  } else if (max_mid_pattern_len <= 64) {
    gqe::detail::run_string_match<uint64_t>(
      input, middle_patterns, prefix, suffix, results->mutable_view(), stream, mr);
  } else {
    // call cudf::strings::like - slow path
    cudf::string_scalar const d_pattern{pattern};
    return cudf::strings::like(input, d_pattern, escape_character);
  }

  results->set_null_count(input.null_count());
  return results;
}

}  // end of namespace gqe
