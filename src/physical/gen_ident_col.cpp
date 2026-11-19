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

#include <gqe/physical/gen_ident_col.hpp>

#include <gqe/logical/utility.hpp>

namespace gqe {
namespace physical {
std::vector<cudf::data_type> gen_ident_col_relation::output_data_types() const
{
  // the output data types should be the input data type + new uid (unique row identifer) column
  auto output = this->children_unsafe()[0]->output_data_types();
  // since this relation is to generate an uid, it should always be 64bit integer
  // for this new columns
  output.push_back(cudf::data_type{cudf::type_id::INT64});
  return output;
}

std::string gen_ident_col_relation::to_string() const
{
  std::string gen_ident_col_relation_string = "{\"Gen_Ident_Col\" : {\n";
  // Output data types
  gen_ident_col_relation_string +=
    "\t\"output data types\" : " + logical::utility::list_to_string(output_data_types()) + ",\n";
  // Children
  gen_ident_col_relation_string +=
    "\t\"children\" : " + logical::utility::list_to_string(children_unsafe()) + "\n";
  gen_ident_col_relation_string += "}}";
  return gen_ident_col_relation_string;
}

}  // end of namespace physical
}  // end of namespace gqe
