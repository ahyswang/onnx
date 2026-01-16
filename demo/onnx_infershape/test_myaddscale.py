import onnx
from onnx import helper, TensorProto

import sys 
import os 
import test_utils

os.makedirs("./data.ignore", exist_ok=True)

# 构造包含自定义算子的简单图
def generate_myaddscale(model_path):
    # 前置 Add 节点
    add1_node = helper.make_node(
        "Add",                         
        inputs=["X", "A"],            
        outputs=["X_add"], 
        domain=""           
    )

    # 自定义算子节点
    node = helper.make_node(
        "MyAddScale",    
        name="MyAddScale",              
        inputs=["X_add", "scale"],     
        outputs=["Y_custom"],         
        domain="com.nebula"
    )

    # 后置 Add 节点
    add2_node = helper.make_node(
        "Add",                         
        inputs=["Y_custom", "B"],      
        outputs=["Y"],  
        domain=""               
    )

    graph = helper.make_graph(
        [add1_node, node, add2_node],
        "custom_op_graph",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 5]),
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 5]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 5]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 5])
        ]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.nebula", 1),  # COM_EXAMPLE_DOMAIN = "com.nebula"
                                                    helper.make_opsetid("", 21)]) # ONNX_DOMAIN = ""
    onnx.save(model, model_path)

def test_matmul_rhs_group_quant_with_add():

    test_cases = [
        {
            "name": "matmul_rhs_group_quant",
            "model_path": "./data.ignore/test_myaddscale_inferred.onnx",
            "generate_func": generate_myaddscale,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "Y_custom" : {"type": "FLOAT", "shape": [3, 5]}
            }
        },
    ]

    test_utils.run_onnx_test(test_cases)

if __name__ == "__main__":

    test_matmul_rhs_group_quant_with_add()