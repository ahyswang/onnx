import onnx
from onnx import helper, TensorProto

import sys 
import os 
from test_utils import check_node_output_type_and_shape

os.makedirs("./data.ignore", exist_ok=True)

# 构造包含 QLinearAdd 算子的简单图
def test_qlineadd():

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
    onnx.save(model, "./data.ignore/test_qlinearadd.onnx")

    # 尝试加载
    try:
        m = onnx.load("./data.ignore/test_qlinearadd.onnx")
        print("模型加载成功，包含 QLinearAdd 算子！")
        onnx.checker.check_model(m, full_check=True)
        print("模型检查成功，包含 QLinearAdd 算子！")
        inferred_model = onnx.shape_inference.infer_shapes(m)
        onnx.save_model(inferred_model, "./data.ignore/test_qlinearadd_inferred.onnx")
        print("模型推形状成功，包含 QLinearAdd 算子！")

        output_info = check_node_output_type_and_shape(
            "./data.ignore/test_qlinearadd_inferred.onnx",
            "QLinearAdd_1"
        )
        print(output_info)
        assert output_info["Y_custom"]["type"] == "INT8", "输出类型不匹配，预期为 INT8"
        assert output_info["Y_custom"]["shape"] == [3, 5], "输出形状不匹配，预期为 [3, 5]"

    except Exception as e:
        print("加载失败: ", e)