/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gqe/rpc/serialization/optimization_parameters.hpp>

#include <boost/pfr.hpp>

#include <format>
#include <type_traits>

namespace gqe::rpc {

namespace {

template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_optional_v = is_optional<std::remove_cvref_t<T>>::value;

proto::SessionOptionValue to_proto_value(arrow::flight::SessionOptionValue const& v)
{
  proto::SessionOptionValue out;
  std::visit(
    [&](auto const& val) {
      using T = std::decay_t<decltype(val)>;
      if constexpr (std::is_same_v<T, bool>) {
        out.set_bool_value(val);
      } else if constexpr (std::is_same_v<T, int64_t>) {
        out.set_int64_value(val);
      } else if constexpr (std::is_same_v<T, double>) {
        out.set_double_value(val);
      } else if constexpr (std::is_same_v<T, std::string>) {
        out.set_string_value(val);
      }
      // std::monostate: leave oneof unset
    },
    v);
  return out;
}

arrow::flight::SessionOptionValue from_proto_value(proto::SessionOptionValue const& v)
{
  switch (v.option_value_case()) {
    case proto::SessionOptionValue::kBoolValue: return v.bool_value();
    case proto::SessionOptionValue::kInt64Value: return v.int64_value();
    case proto::SessionOptionValue::kDoubleValue: return v.double_value();
    case proto::SessionOptionValue::kStringValue: return v.string_value();
    default: return std::monostate{};
  }
}

}  // namespace

std::map<std::string, arrow::flight::SessionOptionValue> optimization_parameters_to_session_options(
  gqe::optimization_parameters const& p)
{
  using V = arrow::flight::SessionOptionValue;
  std::map<std::string, V> r;

  auto to_session_value = [](auto const& v) -> V {
    using U = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<U, bool>)
      return v;
    else if constexpr (std::is_same_v<U, std::string>)
      return v;
    else if constexpr (std::is_same_v<U, double>)
      return v;
    else if constexpr (std::is_enum_v<U>)
      return gqe::to_string(v);
    else {
      static_assert(std::is_integral_v<U>, "Unhandled optimization_parameters field type");
      return static_cast<int64_t>(v);
    }
  };

  boost::pfr::for_each_field_with_name(p, [&](std::string_view fname, auto const& field) {
    using T = std::decay_t<decltype(field)>;
    std::string name{fname};
    if constexpr (is_optional_v<T>) {
      r[name] = field ? to_session_value(*field) : V{std::monostate{}};
    } else {
      r[name] = to_session_value(field);
    }
  });

  return r;
}

arrow::Status apply_session_option(gqe::optimization_parameters& p,
                                   std::string_view name,
                                   arrow::flight::SessionOptionValue const& value)
{
  bool found           = false;
  arrow::Status status = arrow::Status::OK();

  auto apply_value = [&](auto& v) -> arrow::Status {
    using U = std::decay_t<decltype(v)>;
    if constexpr (std::is_same_v<U, bool>) {
      if (auto* val_ptr = std::get_if<bool>(&value)) {
        v = *val_ptr;
        return arrow::Status::OK();
      }
      return arrow::Status::Invalid(std::format("Expected bool for '{}'", name));
    } else if constexpr (std::is_same_v<U, std::string>) {
      if (auto* val_ptr = std::get_if<std::string>(&value)) {
        v = *val_ptr;
        return arrow::Status::OK();
      }
      return arrow::Status::Invalid(std::format("Expected string for '{}'", name));
    } else if constexpr (std::is_same_v<U, double>) {
      if (auto* val_ptr = std::get_if<double>(&value)) {
        v = *val_ptr;
        return arrow::Status::OK();
      }
      return arrow::Status::Invalid(std::format("Expected double for '{}'", name));
    } else if constexpr (std::is_enum_v<U>) {
      if (auto* val_ptr = std::get_if<std::string>(&value)) {
        try {
          v = gqe::from_string<U>(*val_ptr);
          return arrow::Status::OK();
        } catch (std::invalid_argument const& e) {
          return arrow::Status::Invalid(e.what());
        }
      }
      return arrow::Status::Invalid(std::format("Expected string for '{}'", name));
    } else {
      static_assert(std::is_integral_v<U>, "Unhandled optimization_parameters field type");
      if (auto* val_ptr = std::get_if<int64_t>(&value)) {
        v = static_cast<U>(*val_ptr);
        return arrow::Status::OK();
      }
      return arrow::Status::Invalid(std::format("Expected int64 for '{}'", name));
    }
  };

  boost::pfr::for_each_field_with_name(p, [&](std::string_view fname, auto& field) {
    if (found || std::string_view{name} != fname) return;
    found   = true;
    using T = std::decay_t<decltype(field)>;
    if constexpr (is_optional_v<T>) {
      if (std::get_if<std::monostate>(&value)) {
        field = std::nullopt;
      } else {
        typename T::value_type inner{};
        status = apply_value(inner);
        if (status.ok()) field = inner;
      }
    } else {
      status = apply_value(field);
    }
  });

  if (!found)
    return arrow::Status::KeyError(std::format("Unknown optimization parameter: '{}'", name));
  return status;
}

proto::OptimizationParameters serialize_optimization_parameters(
  gqe::optimization_parameters const& p)
{
  proto::OptimizationParameters out;
  auto& opts = *out.mutable_options();
  for (auto const& [name, value] : optimization_parameters_to_session_options(p)) {
    if (std::holds_alternative<std::monostate>(value)) continue;
    opts[name] = to_proto_value(value);
  }
  return out;
}

arrow::Result<gqe::optimization_parameters> deserialize_optimization_parameters(
  proto::OptimizationParameters const& proto)
{
  auto p = gqe::make_optimization_parameters(/*only_defaults=*/true);
  for (auto const& [name, proto_val] : proto.options()) {
    ARROW_RETURN_NOT_OK(apply_session_option(p, name, from_proto_value(proto_val)));
  }
  return p;
}

}  // namespace gqe::rpc
