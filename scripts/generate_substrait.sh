#!/bin/sh

# Check whether the correct protobuf version is installed
current_ver="$(protoc --version)"
required_ver="libprotoc 3.20.1"
 if [ "$current_ver" = "$required_ver" ]; then 
        echo "Detected correct protobuf version: ${current_ver}"
 else
        echo "Require protobuf version ${required_ver} but detected ${current_ver}"
        echo "Exiting"
        exit 1
 fi

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
