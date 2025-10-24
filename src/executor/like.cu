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

#include <cuco/extent.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_map.cuh>
#include <cuco/storage.cuh>

#include <cooperative_groups.h>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/contains.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gqe {

namespace detail {

// Encoding type for compile-time dispatch
enum class EncodingTag { ascii, utf8Bytewise };

constexpr char multi_wildcard  = '%';
constexpr char single_wildcard = '_';
// uint32_t has 4 bytes (shift 2), for each char, we need to shift 8,
// but for each uint32_t, we need to shift 6
constexpr int shift_with_uint32 = 6;
// meaning it's asc only with 256 chars
constexpr int charset_bits = 8;
constexpr int num_chars    = 256;

using size_t = std::size_t;

/**
 * @brief Check if the string matches the exact pattern
 * @param string_ptr The pointer to the string
 * @param pattern The pattern to match
 * @param wildcard_flags The wildcard flags for the pattern
 * @param pattern_len The length of the pattern
 * @return True if the string matches the exact pattern, false otherwise
 */
__device__ bool check_exact_pattern_match(char const* string_ptr,
                                          char const* pattern,
                                          bool const* wildcard_flags,
                                          size_t pattern_len)
{
  for (size_t i = 0; i < pattern_len; i++) {
    char c = string_ptr[i];
    char p = pattern[i];
    // When wildcard_flags[i] is false, this position is a regular character
    if (!wildcard_flags[i] && c != p) { return false; }
    // If wildcard_flags[i] is true, this position matches any character
  }
  return true;
}

/**
 * @brief Check if the string matches the multi-pattern
 * @tparam MASK_TYPE The type of the mask
 * @param string_ptr The pointer to the string
 * @param string_len The length of the string
 * @param pattern_masks The masks for the patterns
 * @param pattern_lengths The lengths of the patterns
 * @param num_middle_patterns The number of middle patterns
 * @param total_pattern_len The total length of the pattern
 * @return True if the string matches the multi-pattern, false otherwise
 */
template <typename MASK_TYPE>
__device__ bool check_multi_pattern_match_bytewise(char const* string_ptr,
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
  // If we have no pattern overlap and are at this position, we can exit early
  int farthest_start = string_len - total_pattern_len;

  // This is still applicable for UTF-8 bytewise comparison since it's still a bytewise comparison
  // and also UTF-8 is fully compatible with ASCII.
  if (string_len >= total_pattern_len) {
    size_t i                 = 0;
    size_t const num_batches = string_len / 4;

    // Process 4 bytes at a time using uchar4 (no alignment requirement, no divergence)
    for (size_t batch_nr = 0; batch_nr < num_batches; ++batch_nr) {
      size_t const base_idx = batch_nr * 4;
      // Read 4 individual bytes and pack into uchar4 (alignment-safe)
      uchar4 char_batch = make_uchar4((unsigned char)string_ptr[base_idx],
                                      (unsigned char)string_ptr[base_idx + 1],
                                      (unsigned char)string_ptr[base_idx + 2],
                                      (unsigned char)string_ptr[base_idx + 3]);

      for (int j = 0; j < 4; ++j) {
        unsigned char c = reinterpret_cast<unsigned char*>(&char_batch)[j];
        state           = (state << 1) + 1;
        state           = state & mask[c];
        if ((state & final_state) != 0) {
          farthest_start += pattern_len;
          pattern_num++;
          if (pattern_num == num_middle_patterns) { return true; }
          state       = 0;
          pattern_len = pattern_lengths[pattern_num];
          final_state = 1 << (pattern_len - 1);
          mask += num_chars;
        }
        if (i > farthest_start && (state == 0)) { return false; }
        i++;
      }
    }

    // Handle remaining bytes (0-3 bytes)
    // The code below repeats the logic of the loop above, but keep as is
    // Tried to use refactor to a lambda function to dedup the logic, but
    // the perf hurts more, the kernel time increases more than 30%
    for (; i < string_len; i++) {
      state        = (state << 1) + 1;
      char const c = string_ptr[i];
      state        = state & mask[(uint8_t)c];
      if ((state & final_state) != 0) {
        farthest_start += pattern_len;
        pattern_num++;
        if (pattern_num == num_middle_patterns) { return true; }
        state       = 0;
        pattern_len = pattern_lengths[pattern_num];
        final_state = 1 << (pattern_len - 1);  // reset for next word
        mask += num_chars;                     // each mask is 256 elements
      }
      // Exit early if not possible to complete pattern in remaining characters
      if (i > farthest_start && (state == 0)) break;
    }
  }

  return false;
}

/**
 * @brief Kernel function for the like pattern matching
 * @tparam MASK_TYPE The type of the mask
 * @param d_strings The column of strings to evaluate
 * @param d_prefix The prefix of the pattern
 * @param d_suffix The suffix of the pattern
 * @param prefix_wildcard_flags The wildcard flags for the prefix
 * @param suffix_wildcard_flags The wildcard flags for the suffix
 * @param middle_masks The masks for the middle patterns
 * @param middle_lengths The lengths of the middle patterns
 * @param num_middle_patterns The number of middle patterns
 * @param total_pattern_len The total length of the pattern
 * @param results The results of the pattern matching
 */
template <typename MASK_TYPE>
__global__ void like_fn_shift_and_kernel(cudf::column_device_view d_strings,
                                         cudf::string_view const d_prefix,
                                         cudf::string_view const d_suffix,
                                         bool const* prefix_wildcard_flags,
                                         bool const* suffix_wildcard_flags,
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
    // Use size_bytes() for ASCII charset, it's much faster than length()
    // since cudf::string_view is UTF-8 compatible
    string_len = d_str.size_bytes();
  }

  // Check for prefix
  size_t prefix_len = d_prefix.length();
  if (prefix_len > 0) {
    char const* prefix = d_prefix.data();
    if (result && (string_len >= prefix_len)) {
      result = check_exact_pattern_match(string_ptr, prefix, prefix_wildcard_flags, prefix_len);
    }
  }

  // Check for suffix
  size_t suffix_len = d_suffix.length();
  if (suffix_len > 0) {
    char const* suffix = d_suffix.data();
    if (result && ((string_len - prefix_len) >= suffix_len)) {
      result = check_exact_pattern_match(
        string_ptr + string_len - suffix_len, suffix, suffix_wildcard_flags, suffix_len);
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
      result = check_multi_pattern_match_bytewise(string_ptr + prefix_len,
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

/**
 * @brief Generate shift-and mask tables for all middle patterns
 * @tparam MASK_TYPE The type of the mask
 * @tparam encoding The encoding type for compile-time dispatch
 * @param middle_patterns The middle patterns to generate masks for
 * @param middle_wildcard_flags The wildcard flags for each pattern (only used for ASCII encoding)
 * @return A vector containing all mask tables concatenated (num_chars * num_patterns elements)
 */
template <typename MASK_TYPE, EncodingTag encoding>
std::vector<MASK_TYPE> generate_shift_and_mask_tables(
  std::vector<std::string> const& middle_patterns,
  std::vector<std::vector<uint8_t>> const& middle_wildcard_flags)
{
  static_assert(std::is_integral<MASK_TYPE>::value, "MASK_TYPE must be an integral type");

  size_t num_middle_patterns = middle_patterns.size();
  if (num_middle_patterns == 0) { return {}; }

  // Allocate mask table: num_chars entries per pattern
  std::vector<MASK_TYPE> mask_tables(num_chars * num_middle_patterns, 0);

  for (size_t i = 0; i < num_middle_patterns; i++) {
    std::string const& pattern = middle_patterns[i];
    MASK_TYPE* out_mask        = mask_tables.data() + num_chars * i;

    if constexpr (encoding == EncodingTag::utf8Bytewise) {
      // UTF-8 bytewise: simple byte-by-byte matching (no wildcard support)
      for (size_t pattern_idx = 0; pattern_idx < pattern.length(); pattern_idx++) {
        char c               = pattern[pattern_idx];
        out_mask[(uint8_t)c] = out_mask[(uint8_t)c] | (1 << pattern_idx);
      }
    } else {
      // ASCII: support single-character wildcards
      std::vector<uint8_t> const& wildcard_flags = middle_wildcard_flags[i];
      for (size_t pattern_idx = 0; pattern_idx < pattern.length(); pattern_idx++) {
        char c = pattern[pattern_idx];
        // Use pre-computed wildcard flags from preprocess_like
        if (wildcard_flags[pattern_idx]) {
          // This position is an unescaped single wildcard
          for (size_t j = 0; j < num_chars; j++) {
            out_mask[j] = out_mask[j] | (1 << pattern_idx);
          }
        } else {
          // This is a regular character (or escaped underscore)
          out_mask[(uint8_t)c] = out_mask[(uint8_t)c] | (1 << pattern_idx);
        }
      }
    }
  }

  return mask_tables;
}

/**
 * @brief Run the string matching kernel
 * @tparam MASK_TYPE The type of the mask
 * @tparam encoding The encoding type
 * @param input The input column
 * @param middle_patterns The middle patterns
 * @param middle_wildcard_flags The wildcard flags for the middle patterns
 * @param prefix The prefix of the pattern
 * @param suffix The suffix of the pattern
 * @param prefix_wildcard_flags The wildcard flags for the prefix
 * @param suffix_wildcard_flags The wildcard flags for the suffix
 * @param results The results of the pattern matching
 * @param stream The CUDA stream
 * @param mr The device memory resource
 */
template <typename MASK_TYPE, EncodingTag encoding = EncodingTag::ascii>
void run_string_match(cudf::strings_column_view const& input,
                      std::vector<std::string> const& middle_patterns,
                      std::vector<std::vector<uint8_t>> const& middle_wildcard_flags,
                      std::string const& prefix,
                      std::string const& suffix,
                      std::vector<uint8_t> const& prefix_wildcard_flags,
                      std::vector<uint8_t> const& suffix_wildcard_flags,
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
    assert(middle_wildcard_flags.size() == num_middle_patterns);
    // Generate mask tables for all middle patterns
    // here num_chars=256 means the character size is 256, only applicable for ASCII-like character
    // set not applicable for Unicode charset
    auto mask_tables =
      generate_shift_and_mask_tables<MASK_TYPE, encoding>(middle_patterns, middle_wildcard_flags);
    middle_masks = std::make_unique<std::vector<MASK_TYPE>>(std::move(mask_tables));
  }

  // Allocate device buffers for prefix, suffix, and wildcard flags
  std::unique_ptr<rmm::device_buffer> d_prefix_buf                = nullptr;
  std::unique_ptr<rmm::device_buffer> d_suffix_buf                = nullptr;
  std::unique_ptr<rmm::device_buffer> d_prefix_wildcard_flags_buf = nullptr;
  std::unique_ptr<rmm::device_buffer> d_suffix_wildcard_flags_buf = nullptr;
  // Pointers to device memory for prefix, suffix, and wildcard flags
  char* d_prefix                = nullptr;
  char* d_suffix                = nullptr;
  bool* d_prefix_wildcard_flags = nullptr;
  bool* d_suffix_wildcard_flags = nullptr;
  if (has_prefix) {
    d_prefix_buf = std::make_unique<rmm::device_buffer>(prefix.length(), stream, mr);
    d_prefix     = reinterpret_cast<char*>(d_prefix_buf->data());
    GQE_CUDA_TRY(cudaMemcpy(d_prefix, prefix.data(), prefix.length(), cudaMemcpyDefault));
    d_prefix_wildcard_flags_buf =
      std::make_unique<rmm::device_buffer>(prefix.length() * sizeof(bool), stream, mr);
    d_prefix_wildcard_flags = reinterpret_cast<bool*>(d_prefix_wildcard_flags_buf->data());
    GQE_CUDA_TRY(cudaMemcpy(d_prefix_wildcard_flags,
                            prefix_wildcard_flags.data(),
                            prefix.length() * sizeof(bool),
                            cudaMemcpyDefault));
  }
  if (has_suffix) {
    d_suffix_buf = std::make_unique<rmm::device_buffer>(suffix.length(), stream, mr);
    d_suffix     = reinterpret_cast<char*>(d_suffix_buf->data());
    GQE_CUDA_TRY(cudaMemcpy(d_suffix, suffix.data(), suffix.length(), cudaMemcpyDefault));
    d_suffix_wildcard_flags_buf =
      std::make_unique<rmm::device_buffer>(suffix.length() * sizeof(bool), stream, mr);
    d_suffix_wildcard_flags = reinterpret_cast<bool*>(d_suffix_wildcard_flags_buf->data());
    GQE_CUDA_TRY(cudaMemcpy(d_suffix_wildcard_flags,
                            suffix_wildcard_flags.data(),
                            suffix.length() * sizeof(bool),
                            cudaMemcpyDefault));
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
    total_pattern_len += len;
  }

  // Execute the string matching kernel
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
    d_prefix_wildcard_flags,
    d_suffix_wildcard_flags,
    d_middle_masks != nullptr ? d_middle_masks->data() : nullptr,
    d_middle_lengths != nullptr ? d_middle_lengths->data() : nullptr,
    num_middle_patterns,
    total_pattern_len,
    results.data<bool>());
}

/**
 * @brief Preprocess the pattern
 * Check if the max length of the middle patterns is <= 64,
 * Basically, this function is to find the prefix, suffix as well as strings in the middle
 * e.g. hello%world%good%morning: the prefix is "hello", suffix is "morning"
 * and strings in the middle are ["world", "good"]
 * TODO: Future optimization is to use the smallest possible table size for each pattern.
 * @param pattern The pattern to preprocess
 * @param prefix The prefix of the pattern
 * @param suffix The suffix of the pattern
 * @param prefix_wildcard_flags The wildcard flags for the prefix
 * @param suffix_wildcard_flags The wildcard flags for the suffix
 * @param middle_patterns The middle patterns
 * @param middle_wildcard_flags The wildcard flags for the middle patterns
 * @param max_mid_pattern_len The maximum length of the middle patterns
 * @param escape_character The escape character
 * @param has_single_wildcard Whether the pattern contains a single wildcard
 */
void preprocess_like(std::string const& pattern,
                     std::string& prefix,
                     std::string& suffix,
                     std::vector<uint8_t>& prefix_wildcard_flags,
                     std::vector<uint8_t>& suffix_wildcard_flags,
                     std::vector<std::string>& middle_patterns,
                     std::vector<std::vector<uint8_t>>& middle_wildcard_flags,
                     size_t& max_mid_pattern_len,
                     char const escape_character,
                     bool& has_single_wildcard)
{
  size_t const pattern_len = pattern.length();

  std::vector<std::string> all_patterns;
  std::vector<std::vector<uint8_t>> all_wildcard_flags;
  // for holding the temporary string that's processed in the middle
  std::string tmp_string;
  std::vector<uint8_t> tmp_wildcard_flags;

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
      // Remember if this position is an unescaped single wildcard
      tmp_wildcard_flags.push_back(!escaped && pattern_char == single_wildcard);
      // check if the pattern contains single wildcard
      if (!has_single_wildcard && !escaped && pattern_char == single_wildcard) {
        has_single_wildcard = true;
      }
      if (curr == pattern_len - 1) {
        all_patterns.push_back(tmp_string);
        all_wildcard_flags.push_back(tmp_wildcard_flags);
      }
    } else if (tmp_string.length() > 0) {
      all_patterns.push_back(tmp_string);
      all_wildcard_flags.push_back(tmp_wildcard_flags);
      tmp_string = "";
      tmp_wildcard_flags.clear();
    }

    curr++;
  }

  // Check the prefix, we only need to consider if the first char is not multi_wildcard
  if (all_patterns.size() > 0 && pattern[0] != multi_wildcard) {
    prefix                = all_patterns[0];
    prefix_wildcard_flags = all_wildcard_flags[0];
    all_patterns.erase(all_patterns.begin());
    all_wildcard_flags.erase(all_wildcard_flags.begin());
  }

  // To check the suffix, we also need to consider the escaped character case
  // when the last char is multi_wildcard
  if (all_patterns.size() > 0 &&
      (pattern[pattern_len - 1] != multi_wildcard ||
       (pattern_len > 1 && pattern[pattern_len - 2] == escape_character))) {
    suffix                = all_patterns[all_patterns.size() - 1];
    suffix_wildcard_flags = all_wildcard_flags[all_wildcard_flags.size() - 1];
    all_patterns.pop_back();
    all_wildcard_flags.pop_back();
  }

  for (auto pattern : all_patterns) {
    max_mid_pattern_len = max(max_mid_pattern_len, pattern.length());
  }

  // set middle_patterns to all the patterns, minus the prefix and suffix
  middle_patterns       = all_patterns;
  middle_wildcard_flags = all_wildcard_flags;
}

}  // end of namespace detail

// Internal implementation that handles both ASCII and UTF-8 bytewise matching
template <detail::EncodingTag encoding>
std::unique_ptr<cudf::column> like_impl(cudf::strings_column_view const& input,
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
  // If there is no character after an escape character in the LIKE pattern,
  // the pattern isn't valid and the LIKE returns FALSE.
  // REF:
  // https://learn.microsoft.com/en-us/sql/t-sql/language-elements/like-transact-sql?view=sql-server-ver17
  auto const h_escape = escape_character.to_string(stream);
  if (!h_escape.empty() && pattern.back() == h_escape.data()[0]) { return results; }

  // preprocess the pattern
  std::string prefix = "";
  std::string suffix = "";
  std::vector<uint8_t> prefix_wildcard_flags;
  std::vector<uint8_t> suffix_wildcard_flags;
  std::vector<std::string> middle_patterns;
  std::vector<std::vector<uint8_t>> middle_wildcard_flags;
  size_t max_mid_pattern_len = 0;
  char const esc_char        = h_escape.empty() ? 0 : h_escape.data()[0];

  // For UTF-8 bytewise: check if pattern contains '_' wildcard (which requires character-aware
  // matching) For ASCII: skip the check since ASCII can handle single-byte wildcards
  bool has_single_wildcard = (encoding != detail::EncodingTag::utf8Bytewise);

  // extract the pattern into prefix, suffix, and middle patterns here
  gqe::detail::preprocess_like(pattern,
                               prefix,
                               suffix,
                               prefix_wildcard_flags,
                               suffix_wildcard_flags,
                               middle_patterns,
                               middle_wildcard_flags,
                               max_mid_pattern_len,
                               esc_char,
                               has_single_wildcard);

  // For UTF-8 bytewise only: if single wildcard detected, fall back to character-aware matching
  if constexpr (encoding == detail::EncodingTag::utf8Bytewise) {
    if (has_single_wildcard) {
      // Fall back to cudf::strings::like for UTF-8 character-aware wildcard matching
      cudf::string_scalar const d_pattern{pattern};
      return cudf::strings::like(input, d_pattern, escape_character);
    }
  }

  // run string match based on different max length of mid pattern strings
  if (max_mid_pattern_len <= 8) {
    gqe::detail::run_string_match<uint8_t, encoding>(input,
                                                     middle_patterns,
                                                     middle_wildcard_flags,
                                                     prefix,
                                                     suffix,
                                                     prefix_wildcard_flags,
                                                     suffix_wildcard_flags,
                                                     results->mutable_view(),
                                                     stream,
                                                     mr);
  } else if (max_mid_pattern_len <= 16) {
    gqe::detail::run_string_match<uint16_t, encoding>(input,
                                                      middle_patterns,
                                                      middle_wildcard_flags,
                                                      prefix,
                                                      suffix,
                                                      prefix_wildcard_flags,
                                                      suffix_wildcard_flags,
                                                      results->mutable_view(),
                                                      stream,
                                                      mr);
  } else if (max_mid_pattern_len <= 32) {
    gqe::detail::run_string_match<uint32_t, encoding>(input,
                                                      middle_patterns,
                                                      middle_wildcard_flags,
                                                      prefix,
                                                      suffix,
                                                      prefix_wildcard_flags,
                                                      suffix_wildcard_flags,
                                                      results->mutable_view(),
                                                      stream,
                                                      mr);
  } else if (max_mid_pattern_len <= 64) {
    gqe::detail::run_string_match<uint64_t, encoding>(input,
                                                      middle_patterns,
                                                      middle_wildcard_flags,
                                                      prefix,
                                                      suffix,
                                                      prefix_wildcard_flags,
                                                      suffix_wildcard_flags,
                                                      results->mutable_view(),
                                                      stream,
                                                      mr);
  } else {
    // call cudf::strings::like - slow path
    cudf::string_scalar const d_pattern{pattern};
    return cudf::strings::like(input, d_pattern, escape_character);
  }

  results->set_null_count(input.null_count());
  return results;
}

std::unique_ptr<cudf::column> like_utf8_bytewise(cudf::strings_column_view const& input,
                                                 std::string const& pattern,
                                                 cudf::string_scalar const& escape_character,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  return like_impl<detail::EncodingTag::utf8Bytewise>(input, pattern, escape_character, stream, mr);
}

std::unique_ptr<cudf::column> like(cudf::strings_column_view const& input,
                                   std::string const& pattern,
                                   cudf::string_scalar const& escape_character,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  return like_impl<detail::EncodingTag::ascii>(input, pattern, escape_character, stream, mr);
}

}  // end of namespace gqe
