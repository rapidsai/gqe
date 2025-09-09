#pragma once

#include <mpi.h>

#include <gqe/utility/error.hpp>

#include <string>
#include <type_traits>
#include <vector>

namespace gqe::utility {
namespace multi_process {

template <typename T, std::enable_if_t<std::is_pointer_v<T>>* = nullptr>
std::vector<T> all_gather_ptr(MPI_Comm comm, T send_ptr)
{
  int world_size = 0;
  GQE_MPI_TRY(MPI_Comm_size(comm, &world_size));

  // All gather PGAS base pointer allocations
  auto base_ptrs = std::vector<T>(world_size);
  GQE_MPI_TRY(
    MPI_Allgather(&send_ptr, sizeof(T), MPI_CHAR, base_ptrs.data(), sizeof(T), MPI_CHAR, comm));
  return base_ptrs;
}

std::vector<int> all_gather_int(MPI_Comm comm, int num);

std::vector<std::string> all_gather_string(MPI_Comm comm, std::string const& send_str);
}  // namespace multi_process
}  // namespace gqe::utility