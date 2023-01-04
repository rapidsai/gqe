#!/bin/sh

# Header files generation
substrait_include_dir="./include/substrait"
if [ -d ${substrait_include_dir} ]
then
    echo "Directory ${substrait_include_dir} exists"
    echo "Header files will NOT be re-generated"
else
    echo "Directory ${substrait_include_dir} does not exist"
    echo "Generating Substrait header files ..."
    echo "Assuming substrait submodule is already initialized"
    cd substrait/proto
    protoc substrait/*.proto substrait/extensions/*.proto --cpp_out=../../include
    echo "Done generating Substrait header files"
fi
