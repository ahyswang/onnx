import onnx
from onnx import helper, TensorProto

import sys 
import os 
import test_utils

os.makedirs("./data.ignore", exist_ok=True)

# 构造包含 QLinearAdd 算子的简单图
def generate_qlineadd(model_path="./data.ignore/test_qlinearadd.onnx"):

    # 定义固定的常量值
    a_scale = helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.1])
    a_zero_point = helper.make_tensor("a_zero_point", TensorProto.INT8, [], [0])
    b_scale = helper.make_tensor("b_scale", TensorProto.FLOAT, [], [0.2])
    b_zero_point = helper.make_tensor("b_zero_point", TensorProto.INT8, [], [0])
    y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.3])
    y_zero_point = helper.make_tensor("y_zero_point", TensorProto.INT8, [], [0])

    # QLinearAdd 算子节点
    add1 = helper.make_node(
        "QLinearAdd", 
        name = "QLinearAdd_1",
        inputs=["A", "a_scale", "a_zero_point", "B", "b_scale", "b_zero_point", "y_scale", "y_zero_point"],
        outputs=["Y_custom"],
        domain="com.nebula"
    )

    add2 = helper.make_node(
        "QLinearAdd", 
        name = "QLinearAdd_2",
        inputs=["Y_custom", "a_scale", "a_zero_point", "C", "b_scale", "b_zero_point", "y_scale", "y_zero_point"],
        outputs=["Y_custom_2"],
        domain="com.nebula"  
    )

    graph = helper.make_graph(
        [add1, add2],
        "qlinear_matmul_graph",
        inputs=[
            helper.make_tensor_value_info("A", TensorProto.INT8, [3, 5]),
            helper.make_tensor_value_info("B", TensorProto.INT8, [3, 5]),
            helper.make_tensor_value_info("C", TensorProto.INT8, [3, 5]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y_custom_2", TensorProto.INT8, [3, 5])
        ],
        initializer=[
            a_scale, a_zero_point, b_scale, b_zero_point, y_scale, y_zero_point  # 添加常量到初始化器
        ]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.nebula", 1),  # COM_EXAMPLE_DOMAIN = "com.example"
                                                    helper.make_opsetid("", 21)])  # ONNX_DOMAIN = ""
    onnx.save(model, model_path)

def test_qlinearadd():

    test_cases = [
        {
            "name": "QLinearAdd",
            "model_path": "./data.ignore/test_qlinearadd.onnx",
            "generate_func": generate_qlineadd,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "Y_custom" : {"type": "INT8", "shape": [3, 5]}
            }
        },
    ]

    test_utils.run_onnx_test(test_cases)

if __name__ == "__main__":

    test_qlinearadd()