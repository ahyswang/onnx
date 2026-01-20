# test_dequantize_linear_per_group. py

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import sys
import os
import test_utils

save_dir = "./data.ignore"
os.makedirs(save_dir, exist_ok=True)

COM_EXAMPLE_DOMAIN = "com.nebula"

def generate_dequantize_linear_per_group_basic(model_path):
    # 输入: 量化的 int8 张量
    # 输出: 反量化的 float32 张量
    dequant_node = helper.make_node(
        "DequantPerGroup",
        name="DequantPerGroup_1",
        inputs=["x", "block_scale", "superblock_scale"],
        outputs=["y_dequant"],
        domain=COM_EXAMPLE_DOMAIN,
        axis=[0,1],
        block_size=[8, 1],
        input_dtype=TensorProto.INT4,
        output_dtype=TensorProto.FLOAT16,
        L1=1,
        L2=0
    )
    
    # Relu 节点
    matmul_node = helper.make_node(
        "MatMul",
        name="MatMul_1",
        inputs=["input", "y_dequant"],
        outputs=["y"],
    )
    
    x = test_utils.generator_onnx_tensor("x", [1024, 2048], TensorProto.INT4)
    block_scale = test_utils.generator_onnx_tensor("block_scale", [128, 2048], TensorProto.INT4)
    superblock_scale = test_utils.generator_onnx_tensor("superblock_scale", [1, 1], TensorProto.FLOAT16)   # TODO: 

    graph = helper.make_graph(
        [dequant_node, matmul_node],
        "dequantize_per_group_basic",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 1024])
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 2048])
        ],
        initializer=[block_scale, superblock_scale, x]
    )
    
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid(COM_EXAMPLE_DOMAIN, 1),
            helper.make_opsetid("", 21)
        ]
    )

    onnx. save(model, model_path)

def generate_dequantize_linear_per_group_superblock(model_path):
    # 输入: 量化的 float8 张量
    # 输出: 反量化的 float16 张量
    dequant_node = helper.make_node(
        "DequantPerGroup",
        name="DequantPerGroup_1",
        inputs=["x", "block_scale", "superblock_scale"],
        outputs=["y_dequant"],
        domain=COM_EXAMPLE_DOMAIN,
        axis=[0, 1],
        superblock_size=[8, 1],
        input_dtype=TensorProto.FLOAT8E4M3FN,
        output_dtype=TensorProto.FLOAT16,
        L1=0,
        L2=1
    )
    
    # Relu 节点
    matmul_node = helper.make_node(
        "MatMul",
        name="MatMul_1",
        inputs=["input", "y_dequant"],
        outputs=["y"],
    )
    
    x = test_utils.generator_onnx_tensor("x", [1024, 2048], TensorProto.FLOAT8E4M3FN)
    block_scale = test_utils.generator_onnx_tensor("block_scale", [1,1], TensorProto.INT4)  # TODO: 
    superblock_scale = test_utils.generator_onnx_tensor("superblock_scale", [128, 2048], TensorProto.FLOAT16)
    
    graph = helper.make_graph(
        [dequant_node, matmul_node],
        "dequantize_per_group_superblock",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 1024]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 2048])
        ],
        initializer=[block_scale, superblock_scale, x]
    )
    
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid(COM_EXAMPLE_DOMAIN, 1),
            helper.make_opsetid("", 21)
        ]
    )
    
    onnx. save(model, model_path)
 
def generate_dequantize_linear_per_group_all(model_path):
    """"""
    dequant_node = helper.make_node(
        "DequantPerGroup",
        name="DequantPerGroup_2",
        inputs=["x", "block_scale", "superblock_scale"],
        outputs=["y_dequant"],
        domain=COM_EXAMPLE_DOMAIN,
        axis=[0, 1],
        block_size=[4, 4],
        superblock_size=[2, 2],
        input_dtype=TensorProto.INT4,
        output_dtype=TensorProto.FLOAT16,
        L1=1,
        L2=1
    )
    
    # 添加 Relu 节点
    matmul_node = helper.make_node(
        "MatMul",
        name="MatMul_1",
        inputs=["input", "y_dequant"],
        outputs=["y"],
    )

    x = test_utils.generator_onnx_tensor("x", [16, 16], TensorProto.INT8)
    block_scale = test_utils.generator_onnx_tensor("block_scale", [4, 4], TensorProto.INT8)
    superblock_scale = test_utils.generator_onnx_tensor("superblock_scale", [2, 2], TensorProto.FLOAT16)

    graph = helper.make_graph(
        [dequant_node, matmul_node],
        "dequantize_per_group_superblock",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [16, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT16, [16, 16])
        ],
        initializer=[block_scale, superblock_scale, x]
    )
    
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid(COM_EXAMPLE_DOMAIN, 1),
            helper.make_opsetid("", 21)
        ]
    )

    onnx.save(model, model_path)

def generate_dequantize_linear_per_group_all_4bit(model_path, input_dtype=TensorProto.INT4, output_dtype=TensorProto.FLOAT16):
    # 输入: 量化的 int8 张量
    # 输出: 反量化的 float32 张量
    dequant_node = helper.make_node(
        "DequantPerGroup",
        name="DequantPerGroup_1",
        inputs=["x", "block_scale", "superblock_scale"],
        outputs=["y_dequant"],
        domain=COM_EXAMPLE_DOMAIN,
        axis=[0,1],
        block_size=[8, 1],
        superblock_size=[8, 1],
        input_dtype=TensorProto.INT4,
        output_dtype=TensorProto.FLOAT16,
        L1=1,
        L2=1
    )
    
    # Relu 节点
    matmul_node = helper.make_node(
        "MatMul",
        name="MatMul_1",
        inputs=["input", "y_dequant"],
        outputs=["y"],
    )
    
    # 初始化常量
    x = test_utils.generator_onnx_tensor("x", [1024, 2048], TensorProto.INT4)
    block_scale = test_utils.generator_onnx_tensor("block_scale", [128, 2048], TensorProto.INT4)
    superblock_scale = test_utils.generator_onnx_tensor("superblock_scale", [16, 2048], TensorProto.FLOAT16)

    graph = helper.make_graph(
        [dequant_node, matmul_node],
        "dequantize_per_group_basic",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 1024]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 2048])
        ],
        initializer=[block_scale, superblock_scale, x]    
    )
    
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid(COM_EXAMPLE_DOMAIN, 1),
            helper.make_opsetid("", 21)  
        ]
    )

    onnx.save(model, model_path)

def test_dequantize_linear_per_group():

    test_cases = [
        {
            "name": "DequantPerGroup_Basic",
            "model_path": "./data.ignore/test_dequantize_per_group_basic.onnx",
            "generate_func": generate_dequantize_linear_per_group_basic,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "y_dequant" : {"type": "FLOAT16", "shape": [1024, 2048]}
            }
        },
        {
            "name": "DequantPerGroup_Superblock",
            "model_path": "./data.ignore/test_dequantize_per_group_superblock.onnx",
            "generate_func": generate_dequantize_linear_per_group_superblock,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "y_dequant" : {"type": "FLOAT16", "shape": [1024, 2048]}
            }
        },
        {
            "name": "DequantPerGroup_All",
            "model_path": "./data.ignore/test_dequantize_per_group_all.onnx",
            "generate_func": generate_dequantize_linear_per_group_all,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "y_dequant" : {"type": "FLOAT16", "shape": [16, 16]}
            }
        },
        {
            "name": "DequantPerGroup_All_4bit",
            "model_path": "./data.ignore/test_dequantize_per_group_all_4bit.onnx",
            "generate_func": generate_dequantize_linear_per_group_all_4bit,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "y_dequant" : {"type": "FLOAT16", "shape": [1024, 2048]}
            }
        },
    ]

    test_utils.run_onnx_test(test_cases)

if __name__ == "__main__":  
    test_dequantize_linear_per_group()
