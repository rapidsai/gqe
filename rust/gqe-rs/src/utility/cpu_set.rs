/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

use std::mem::size_of_val;

/// CPU set to create CPU and NUMA node masks.
///
/// Inspired by Linux's `cpu_set_t`, see the `cpu_set` manual page.
///
/// Limitations
/// ===========
///
/// The set is currently restricted to a 16 64-bit integers. Therefore, IDs
/// must be smaller or equal to 1023.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct CpuSet {
    mask: [u64; Self::MASK_LEN as usize],
}

impl CpuSet {
    const MASK_LEN: u16 = 16;
    const ENTRY_LEN: u16 = 64;
    const MAX_LEN: u16 = Self::MASK_LEN * Self::ENTRY_LEN;

    /// Create an empty CPU set.
    pub fn new() -> Self {
        Self {
            mask: [0; Self::MASK_LEN as usize],
        }
    }

    /// Add an ID to the set.
    pub fn add(&mut self, id: u16) {
        assert!(id < Self::MAX_LEN);

        let entry = &mut self.mask[(id / Self::ENTRY_LEN) as usize];
        let pos = id % Self::ENTRY_LEN;
        *entry |= 1 << pos;
    }

    /// Remove an ID from the set.
    pub fn remove(&mut self, id: u16) {
        assert!(id < Self::MAX_LEN);

        let entry = &mut self.mask[(id / Self::ENTRY_LEN) as usize];
        let pos = id % Self::ENTRY_LEN;
        *entry &= !(1 << pos);
    }

    /// Query if an ID is included in the set.
    pub fn is_set(&self, id: u16) -> bool {
        assert!(id < Self::MAX_LEN);

        let entry = &self.mask[(id / Self::ENTRY_LEN) as usize];
        let pos = id % Self::ENTRY_LEN;
        (*entry & (1 << pos)) != 0
    }

    /// Returns the number of IDs in the set.
    pub fn count(&self) -> usize {
        self.mask.iter().map(|e| e.count_ones() as usize).sum()
    }

    /// Returns the size of the set in bytes.
    pub fn bytes(&self) -> usize {
        size_of_val(&self.mask)
    }

    /// Reset the set to zero.
    pub fn zero(&mut self) {
        self.mask.fill(0);
    }

    /// Query the maximum possible number of IDs currently in the set.
    pub fn max_id(&self) -> u16 {
        let leading_zeros: u16 = self
            .mask
            .iter()
            .rev()
            .scan(true, |take_next, e| {
                if *take_next {
                    let lzs = e.leading_zeros() as u16;
                    if lzs != Self::ENTRY_LEN {
                        *take_next = false;
                    }
                    Some(lzs)
                } else {
                    None
                }
            })
            .sum();

        Self::MAX_LEN - leading_zeros + 1
    }

    /// Get the set as a slice.
    pub fn as_slice(&self) -> &[u64] {
        &self.mask
    }

    /// Get the set as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        &mut self.mask
    }
}

impl std::ops::BitAnd for CpuSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut mask = self.mask;
        mask.iter_mut()
            .zip(rhs.mask.iter())
            .for_each(|(l, r)| *l &= *r);

        Self { mask }
    }
}

impl std::ops::BitOr for CpuSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut mask = self.mask;
        mask.iter_mut()
            .zip(rhs.mask.iter())
            .for_each(|(l, r)| *l |= *r);

        Self { mask }
    }
}

impl std::ops::BitXor for CpuSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut mask = self.mask;
        mask.iter_mut()
            .zip(rhs.mask.iter())
            .for_each(|(l, r)| *l ^= *r);

        Self { mask }
    }
}
