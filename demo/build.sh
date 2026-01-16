#!/bin/bash

source /workspace/yswang26/python_env/onnx_venv/bin/activate

export DEBUG=1
python -m pip install --quiet --upgrade pip setuptools wheel nanobind
export ONNX_BUILD_TESTS=0
export CMAKE_ARGS="-DONNX_BUILD_CUSTOM_PROTOBUF=ON -DBUILD_SHARED_LIBS=OFF"  
# nanobind 2.10.2
export nanobind_DIR=/workspace/yswang26/python_env/onnx_venv/lib/python3.12/site-packages/nanobind/cmake
# --no-clean to keep build files for debugging
# --no-binary :all: to force building from source
python -m pip install . --no-clean --no-binary :all:

# cmake --build .setuptools-cmake-build/ --target onnx_cpp2py_export
# gdb --args python demo/onnx_infershape/main.py