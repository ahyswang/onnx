#!/bin/bash

source /workspace/yswang26/python_env/onnx_venv/bin/activate
pushd ${PWD}/../../
pip install -e .
popd
python main.py