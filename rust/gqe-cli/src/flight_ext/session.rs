// Copyright 2026 NVIDIA Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Hand-written prost structs matching the Arrow Flight session-management
//! proto definitions (`arrow.flight.protocol`).  These are not exposed by the
//! `arrow-flight` 57.3.0 crate, so we define them locally.

/// `arrow.flight.protocol.StringListValue`
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StringListValue {
    #[prost(string, repeated, tag = "1")]
    pub values: Vec<String>,
}

/// `arrow.flight.protocol.SessionOptionValue`
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SessionOptionValue {
    #[prost(oneof = "session_option_value::ValueType", tags = "1, 2, 3, 4, 5")]
    pub value_type: Option<session_option_value::ValueType>,
}

pub mod session_option_value {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ValueType {
        #[prost(string, tag = "1")]
        StringValue(String),
        #[prost(bool, tag = "2")]
        BoolValue(bool),
        /// sfixed64 in the proto schema
        #[prost(sfixed64, tag = "3")]
        Int64Value(i64),
        #[prost(double, tag = "4")]
        DoubleValue(f64),
        #[prost(message, tag = "5")]
        StringListValue(super::StringListValue),
    }
}

/// `arrow.flight.protocol.GetSessionOptionsRequest` (empty message)
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetSessionOptionsRequest {}

/// `arrow.flight.protocol.GetSessionOptionsResult`
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct GetSessionOptionsResult {
    #[prost(map = "string, message", tag = "1")]
    pub session_options: ::std::collections::HashMap<String, SessionOptionValue>,
}

/// `arrow.flight.protocol.SetSessionOptionsRequest`
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct SetSessionOptionsRequest {
    #[prost(map = "string, message", tag = "1")]
    pub session_options: ::std::collections::HashMap<String, SessionOptionValue>,
}
