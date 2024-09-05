/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gqe/utility/timer.hpp>

namespace gqe::utility {

bandwidth_timer::bandwidth_timer(std::string name) : total_elapsed_time(0), name(std::move(name)) {}

void bandwidth_timer::start()
{
  start_time  = std::chrono::high_resolution_clock::now();
  input_bytes = 0;
  units       = 0;
}

void bandwidth_timer::end()
{
  auto end_time = std::chrono::high_resolution_clock::now();
  elapsed_time  = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  total_input_bytes += input_bytes;
  total_units += units;
  total_elapsed_time += elapsed_time;
  data_history.emplace_back(input_bytes, units, elapsed_time.count());
  input_bytes = 0;
  units       = 0;
}

void bandwidth_timer::add(uint64_t bytes, uint64_t units)
{
  this->input_bytes += bytes;
  this->units += units;
}

std::string bandwidth_timer::to_string() const
{
  std::string bw_str = "{\n\"Bandwidth Timer\" : {\n";
  // Output name
  bw_str += "\t\"name\" : \"" + name + "\",\n";
  // Data history
  bw_str += "\t\"instances\" : [\n";
  for (const auto& [bytes, units, elapsed_time] : data_history) {
    bw_str += "\t\t{\"size mb\" : " + std::to_string(bytes / 1e6) +
              ", \"time ms\" : " + std::to_string(elapsed_time / 1e3) +
              ", \"bandwidth\" : " + std::to_string((double)bytes / elapsed_time);
    if (units) {
      bw_str += ", \"units\" : " + std::to_string(units) +
                ", \"size per unit\" : " + std::to_string((double)bytes / units) + "},\n";
    } else {
      bw_str += ", \"units\" : 0, \"size per unit\" : 0},\n";
    }
  }
  bw_str += "\t],\n";
  // Total bandwidth
  bw_str += "\t\"sum\" : { ";
  if (total_input_bytes) {
    bw_str += "\"size mb\" : " + std::to_string(total_input_bytes / 1e6) +
              ", \"time ms\" : " + std::to_string(total_elapsed_time.count() / 1e3) +
              ", \"bandwidth\" : " + std::to_string(total_input_bytes / total_elapsed_time.count());
    if (total_units.load()) {
      bw_str += ", \"units\" : " + std::to_string(total_units.load()) + ", \"size per unit\" : " +
                std::to_string((double)total_input_bytes / total_units.load()) + "\t}\n";
    } else {
      bw_str += ", \"units\" : 0, \"size per unit\" : 0\t}\n";
    }
  } else {
    bw_str +=
      "\t\t\"size mb\" : 0, \"time ms\" : 0, \"bandwidth\" : 0, \"units\" : 0, \"size per unit\" : "
      "0\n\t}\n";
  }
  bw_str += "}\n";
  return bw_str;
}

}  // namespace gqe::utility