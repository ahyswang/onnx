#!/bin/bash

source /workspace/yswang26/python_env/onnx_venv/bin/activate

export DEBUG=1
python -m pip install --quiet --upgrade pip setuptools wheel
export ONNX_BUILD_TESTS=0
python -m pip install . --no-clean --no-binary :all:

# gdb --args python demo/onnx_infershape/main.py