#!/usr/bin/env bash
torch_dir="/home/ady/Downloads/libtorch"

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=${torch_dir}
make