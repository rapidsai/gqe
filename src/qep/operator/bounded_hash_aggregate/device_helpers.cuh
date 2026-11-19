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
 */

#pragma once

#include "map_layout.hpp"

#include <gqe/utility/cuda.hpp>
#include <gqe/utility/helpers.hpp>

#include <cuda/atomic>
#include <cuda/numeric>
#include <cuda/std/bit>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/version>
#include <cudf/aggregation.hpp>

namespace gqe {
namespace qep {
namespace bounded_hash_aggregate::detail {

/**
 * @brief Device-side helpers shared by the insert and scan kernels.
 *
 * Relaxed device-scope atomics, in-register value combination, and launch-geometry helpers.
 */

/**
 * @brief Device-scoped atomic reference for the shared map accesses.
 *
 * All shared map accesses go through it, except the bucket key loads of the insert probe.
 * `load_bucket_keys` issues those as relaxed device-scope vector loads in inline PTX, because
 * `cuda::atomic_ref` has no element-wise vector load.
 *
 * Every operation uses relaxed ordering. Atomicity per slot field is all the kernels need,
 * because cross-stream visibility comes from the executor's stream ordering, not from fences.
 */
template <typename T>
using device_atomic_ref = cuda::atomic_ref<T, cuda::thread_scope_device>;

/** @brief CAS a slot key. Returns the observed key, which equals `expected` when the CAS filled the
 * slot. */
template <typename Key>
__device__ __forceinline__ Key atomic_cas_key(Key* address, Key expected, Key desired)
{
  device_atomic_ref<Key>{*address}.compare_exchange_strong(
    expected, desired, cuda::memory_order_relaxed);
  return expected;
}

/**
 * @brief Two-limb 128-bit atomic accumulate: low-limb fetch_add, carry into the high limb.
 *
 * Correct because addition is commutative and associative, and no map reads happen until all
 * inserts quiesce, so torn intermediate limb states are unobservable.
 */
__device__ __forceinline__ void atomic_add_value(two_limb* address, two_limb addend)
{
  unsigned long long const old = device_atomic_ref<unsigned long long>{address->lo}.fetch_add(
    addend.lo, cuda::memory_order_relaxed);
  unsigned long long const carry = (old + addend.lo < old) ? 1ULL : 0ULL;
  unsigned long long const high  = carry + static_cast<unsigned long long>(addend.hi);
  if (high != 0ULL) {
    device_atomic_ref<long long>{address->hi}.fetch_add(static_cast<long long>(high),
                                                        cuda::memory_order_relaxed);
  }
}

/**
 * @brief Add a DECIMAL128 contribution to a split sum.
 *
 * The inline part takes the contribution's low 64 bits with one wrapping `fetch_add`. Its exact
 * predecessor detects each wrap past +-2^63 in exactly one addition, even under concurrent
 * additions. The wrap counter takes these wraps and the contribution's high part, so it is only
 * touched for wraps and for contributions outside the int64 range.
 */
__device__ __forceinline__ void split_sum_add(split_sum128* slot,
                                              int64_t* wrap_counter,
                                              two_limb addend)
{
  auto const low = cuda::std::bit_cast<int64_t>(addend.lo);
  // The carry wraps modulo 2^64 like the high part of the 128-bit sum. It is unsigned, because
  // a contribution that already wrapped the 128-bit range has a high part near the int64
  // limits, and signed overflow is undefined.
  uint64_t carry = static_cast<uint64_t>(addend.hi) + (addend.lo >> 63);
  // Atomic signed arithmetic wraps in two's complement, so no unsigned detour is needed.
  int64_t const old =
    device_atomic_ref<int64_t>{slot->lo}.fetch_add(low, cuda::memory_order_relaxed);
  if (cuda::add_overflow(old, low).overflow) { carry += low < 0 ? ~uint64_t{0} : uint64_t{1}; }
  if (carry != 0) {
    device_atomic_ref<int64_t>{*wrap_counter}.fetch_add(static_cast<int64_t>(carry),
                                                        cuda::memory_order_relaxed);
  }
}

/** @brief Assemble the DECIMAL128 value of a split sum from its inline part and wrap counter. */
__host__ __device__ __forceinline__ two_limb split_sum_value(split_sum128 slot,
                                                             int64_t wrap_counter)
{
  // A negative inline part borrows 2^64 from the high part, which wraps modulo 2^64.
  uint64_t const borrow = slot.lo < 0 ? 1 : 0;
  return two_limb{static_cast<unsigned long long>(slot.lo),
                  static_cast<long long>(static_cast<uint64_t>(wrap_counter) - borrow)};
}

/** @brief Atomic add for every accumulator and count type other than `two_limb`. */
template <typename Value>
__device__ __forceinline__ void atomic_add_value(Value* address, Value value)
{
  device_atomic_ref<Value>{*address}.fetch_add(value, cuda::memory_order_relaxed);
}

/**
 * @brief Atomic multiply through a CAS loop.
 *
 * No multiply atomic exists in hardware. The loop converges because each iteration either
 * publishes its factor or observes a strictly newer accumulator, and every competing thread
 * publishes exactly once.
 */
template <typename Value>
__device__ __forceinline__ void atomic_mul_value(Value* address, Value value)
{
  device_atomic_ref<Value> ref{*address};
  Value expected = ref.load(cuda::memory_order_relaxed);
  while (!ref.compare_exchange_weak(expected, expected * value, cuda::memory_order_relaxed)) {}
}

template <typename Value, cudf::aggregation::Kind K>
__device__ __forceinline__ Value combine_values(Value lhs, Value rhs);

/**
 * @brief 128-bit relaxed device-scope CAS on a 16-byte-aligned `two_limb` (sm_90+).
 *
 * On success returns `true`. On failure returns `false` and replaces `expected` with the
 * observed value, like `compare_exchange`. The mirrored bucket layout provides the required
 * 16-byte alignment.
 *
 * CCCL exposes the operation through `cuda::atomic_ref` since v3.2.0. Earlier versions ship
 * b128 compare-exchange wrappers that do not compile (fixed by NVIDIA/cccl#8403), so below
 * that version the CAS wraps `atom.relaxed.gpu.cas.b128` in hand-written PTX.
 *
 * @pre The device has compute capability 9.0 or higher, enforced by the build task's
 *      `initialize` before any kernel runs. Older devices trap.
 *
 * @tparam Value A 16-byte, 16-byte-aligned mirrored accumulator (`two_limb` or `arg_pair`).
 */
template <typename Value>
  requires is_mirrored_value<Value>
__device__ __forceinline__ bool atomic_cas_128(Value* address, Value& expected, Value desired)
{
#if __CUDA_ARCH__ >= 900 && CCCL_VERSION >= 3002000
  device_atomic_ref<Value> ref{*address};
  return ref.compare_exchange_strong(expected, desired, cuda::memory_order_relaxed);
#elif __CUDA_ARCH__ >= 900
  two_limb const expected_bits = cuda::std::bit_cast<two_limb>(expected);
  two_limb const desired_bits  = cuda::std::bit_cast<two_limb>(desired);
  two_limb observed;
  asm volatile(
    "{\n\t"
    ".reg .b128 _cmp, _new, _old;\n\t"
    "mov.b128 _cmp, {%2, %3};\n\t"
    "mov.b128 _new, {%4, %5};\n\t"
    "atom.relaxed.gpu.cas.b128 _old, [%6], _cmp, _new;\n\t"
    "mov.b128 {%0, %1}, _old;\n\t"
    "}"
    : "=l"(observed.lo), "=l"(observed.hi)
    : "l"(expected_bits.lo),
      "l"(static_cast<unsigned long long>(expected_bits.hi)),
      "l"(desired_bits.lo),
      "l"(static_cast<unsigned long long>(desired_bits.hi)),
      "l"(address)
    : "memory");
  bool const matched = observed.lo == expected_bits.lo && observed.hi == expected_bits.hi;
  expected           = cuda::std::bit_cast<Value>(observed);
  return matched;
#else
  __trap();
  return false;
#endif
}

/**
 * @brief CAS-loop combine for orderings no hardware atomic implements.
 *
 * Used by floating-point MIN/MAX, whose NaN-aware order differs from the `value < old` rule
 * of libcu++'s fetch_min/fetch_max fallback (that rule would never replace a NaN
 * accumulator). Exits without a CAS once the combined value equals the observed one bitwise.
 */
template <typename Value, cudf::aggregation::Kind K>
__device__ __forceinline__ void atomic_combine_cas(Value* address, Value value)
{
  using bits = utility::make_unsigned_t<Value>;
  device_atomic_ref<Value> ref{*address};
  Value expected = ref.load(cuda::memory_order_relaxed);
  while (true) {
    Value const combined = combine_values<Value, K>(expected, value);
    if (cuda::std::bit_cast<bits>(combined) == cuda::std::bit_cast<bits>(expected)) { return; }
    if (ref.compare_exchange_weak(expected, combined, cuda::memory_order_relaxed)) { return; }
  }
}

/**
 * @brief Combine `value` into `*address` with the aggregation's device atomic.
 *
 * Integral MIN/MAX lower to native atomics. Floating-point MIN/MAX go through the CAS loop
 * with the NaN-aware order. PRODUCT goes through the CAS loop, since no multiply atomic
 * exists.
 */
template <typename Value, cudf::aggregation::Kind K>
__device__ __forceinline__ void atomic_combine(Value* address, Value value)
{
  if constexpr (K == cudf::aggregation::MIN || K == cudf::aggregation::MAX ||
                K == cudf::aggregation::ARGMIN || K == cudf::aggregation::ARGMAX) {
    if constexpr (is_mirrored_value<Value>) {
      // A possibly torn initial read only costs one extra iteration: the CAS validates all
      // 16 bytes and refreshes `expected` on failure.
      auto* limbs = reinterpret_cast<two_limb*>(address);
      two_limb const first{
        device_atomic_ref<unsigned long long>{limbs->lo}.load(cuda::memory_order_relaxed),
        device_atomic_ref<long long>{limbs->hi}.load(cuda::memory_order_relaxed)};
      Value expected = cuda::std::bit_cast<Value>(first);
      while (true) {
        Value const combined = combine_values<Value, K>(expected, value);
        if (cuda::std::bit_cast<two_limb>(combined).lo ==
              cuda::std::bit_cast<two_limb>(expected).lo &&
            cuda::std::bit_cast<two_limb>(combined).hi ==
              cuda::std::bit_cast<two_limb>(expected).hi) {
          return;
        }
        if (atomic_cas_128(address, expected, combined)) { return; }
      }
    } else if constexpr (cuda::std::is_floating_point_v<Value>) {
      atomic_combine_cas<Value, K>(address, value);
    } else {
      // The early exit skips the atomic when the stored value already wins, like the CAS
      // loop's bitwise exit. It is measured 18% faster on a MIN over few distinct values.
      // Racing updates only move the stored value further in the winning direction, so a
      // stale read at worst issues a redundant atomic.
      device_atomic_ref<Value> ref{*address};
      if constexpr (K == cudf::aggregation::MIN) {
        if (value < ref.load(cuda::memory_order_relaxed)) {
          ref.fetch_min(value, cuda::memory_order_relaxed);
        }
      } else {
        if (value > ref.load(cuda::memory_order_relaxed)) {
          ref.fetch_max(value, cuda::memory_order_relaxed);
        }
      }
    }
  } else if constexpr (K == cudf::aggregation::PRODUCT) {
    atomic_mul_value(address, value);
  } else {
    // SUM, SUM_OF_SQUARES, and both COUNT kinds combine by addition.
    atomic_add_value(address, value);
  }
}

/** @brief The aggregation identity: the accumulator's initial value. */
template <typename Value, cudf::aggregation::Kind K>
__host__ __device__ __forceinline__ constexpr Value identity_value()
{
  if constexpr (cuda::std::is_same_v<Value, two_limb>) {
    if constexpr (K == cudf::aggregation::MIN) {
      // The largest 128-bit value: any real value replaces it.
      return two_limb{~0ULL, cuda::std::numeric_limits<long long>::max()};
    } else if constexpr (K == cudf::aggregation::MAX) {
      return two_limb{0ULL, cuda::std::numeric_limits<long long>::min()};
    } else {
      return two_limb{0ULL, 0LL};
    }
  } else if constexpr (K == cudf::aggregation::ARGMIN || K == cudf::aggregation::ARGMAX) {
    if constexpr (is_arg_pair<Value>) {
      // The ordering half carries the matching MIN/MAX identity. The payload half is
      // arbitrary until a real row wins.
      constexpr auto inner =
        K == cudf::aggregation::ARGMIN ? cudf::aggregation::MIN : cudf::aggregation::MAX;
      return Value{identity_value<decltype(Value{}.value), inner>(), 0ULL};
    } else {
      // Unreachable: ARGMIN/ARGMAX dispatch only onto pair accumulators.
      return Value{};
    }
  } else if constexpr (K == cudf::aggregation::MIN || K == cudf::aggregation::MAX) {
    if constexpr (cuda::std::is_floating_point_v<Value>) {
      // NaN is the weakest element of the NaN-aware order: any real value replaces it, and an
      // all-NaN group keeps it, matching the cuDF hash groupby.
      return cuda::std::numeric_limits<Value>::quiet_NaN();
    } else if constexpr (K == cudf::aggregation::MIN) {
      return cuda::std::numeric_limits<Value>::max();
    } else {
      return cuda::std::numeric_limits<Value>::lowest();
    }
  } else if constexpr (K == cudf::aggregation::PRODUCT) {
    return Value{1};
  } else {
    return Value{0};
  }
}

/** @brief Widen one input element into the accumulator type. */
template <typename Input, typename Value>
__device__ __forceinline__ Value widen(Input value)
{
  if constexpr (cuda::std::is_same_v<Value, two_limb>) {
    if constexpr (cuda::std::is_same_v<Input, two_limb>) {
      return value;
    } else {
      auto const wide = static_cast<int64_t>(value);
      return two_limb{static_cast<unsigned long long>(wide), wide < 0 ? -1LL : 0LL};
    }
  } else {
    return static_cast<Value>(value);
  }
}

/** @brief Combine two in-register values with the aggregation. */
template <typename Value, cudf::aggregation::Kind K>
__device__ __forceinline__ Value combine_values(Value lhs, Value rhs)
{
  if constexpr (K == cudf::aggregation::ARGMIN || K == cudf::aggregation::ARGMAX) {
    // The winner is decided by the ordering half alone, with the same NaN-aware selection
    // as MIN/MAX. The payload rides along.
    using ordering = decltype(Value{}.value);
    constexpr auto inner =
      K == cudf::aggregation::ARGMIN ? cudf::aggregation::MIN : cudf::aggregation::MAX;
    auto const winner = combine_values<ordering, inner>(lhs.value, rhs.value);
    // Compare against the right-hand ordering value: on bitwise ties (including the NaN
    // identity against a NaN row) the incoming right-hand pair wins, so a real row always
    // replaces the identity's arbitrary payload.
    bool const rhs_wins = cuda::std::bit_cast<uint64_t>(cuda::std::bit_cast<ordering>(winner)) ==
                          cuda::std::bit_cast<uint64_t>(rhs.value);
    return rhs_wins ? rhs : lhs;
  } else if constexpr (cuda::std::is_same_v<Value, two_limb>) {
    if constexpr (K == cudf::aggregation::MIN || K == cudf::aggregation::MAX) {
      // Signed 128-bit comparison: the high limb decides, the unsigned low limb breaks ties.
      bool const rhs_less = rhs.hi < lhs.hi || (rhs.hi == lhs.hi && rhs.lo < lhs.lo);
      if constexpr (K == cudf::aggregation::MIN) {
        return rhs_less ? rhs : lhs;
      } else {
        return rhs_less ? lhs : rhs;
      }
    } else {
      // SUM: 128-bit add with in-register carry. The high limbs add unsigned, because the sum
      // wraps modulo 2^128 and signed overflow is undefined.
      uint64_t const lo    = lhs.lo + rhs.lo;
      uint64_t const carry = lo < lhs.lo ? 1 : 0;
      return two_limb{lo,
                      static_cast<long long>(static_cast<uint64_t>(lhs.hi) +
                                             static_cast<uint64_t>(rhs.hi) + carry)};
    }
  } else if constexpr (K == cudf::aggregation::MIN) {
    // The self-compare demotes a NaN `lhs`: NaN never wins against a real value, so only an
    // all-NaN group stays NaN. This makes the reduction order-independent and matches the
    // cuDF hash groupby. Integral types fold the dead branch away.
    return rhs < lhs ? rhs : (lhs != lhs ? rhs : lhs);
  } else if constexpr (K == cudf::aggregation::MAX) {
    return lhs < rhs ? rhs : (lhs != lhs ? rhs : lhs);
  } else if constexpr (K == cudf::aggregation::PRODUCT) {
    return lhs * rhs;
  } else {
    return lhs + rhs;
  }
}

// -----------------------------------------------------------------------------
// Launch geometry
// -----------------------------------------------------------------------------

/**
 * @brief Threads per block of every kernel launch. Any warp multiple works. The value is not
 * GPU-specific, and the occupancy-derived grid adapts to the device.
 */
inline constexpr int launch_block_size = 128;

}  // namespace bounded_hash_aggregate::detail
}  // namespace qep
}  // namespace gqe
