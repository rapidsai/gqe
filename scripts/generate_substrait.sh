#!/bin/sh

# Substrait C++ code generation
substrait_in_dir=$1
substrait_out_dir=$2
if [ -d ${substrait_out_dir}/substrait ]
then
    echo "Directory ${substrait_out_dir}/substrait exists"
    echo "Substrait C++ code will NOT be re-generated"
else
    echo "Directory ${substrait_out_dir}/substrait does not exist"
    echo "Generating Substrait C++ code ..."
    echo "Assuming substrait submodule is already initialized"
    cd ${substrait_in_dir}/proto
    protoc substrait/*.proto substrait/extensions/*.proto --cpp_out=${substrait_out_dir}
    echo "Done generating Substrait C++ code"
fi
