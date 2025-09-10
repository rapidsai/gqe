#pragma once

#include <cudf/column/column_view.hpp>

namespace libperfect {

std::tuple<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>>
unique_indices(std::vector<cudf::column_view> const& keys, cudf::column_view const& mask);

}  // namespace libperfect
