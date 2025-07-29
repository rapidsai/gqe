#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>

std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
          std::unique_ptr<rmm::device_uvector<cudf::size_type>>>
perfect_join(const cudf::table_view& left_keys, const cudf::table_view& right_keys);
