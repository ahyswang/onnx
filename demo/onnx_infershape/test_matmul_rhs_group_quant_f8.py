import onnx
from onnx import helper, TensorProto
import sys 
import os 
import test_utils

os.makedirs("./data.ignore", exist_ok=True)

# 定义 matmul_rhs_group_quant 节点
def generate_matmul_rhs_group_quant_f8(model_path):

    matmul_node = helper.make_node(
        "matmul_rhs_group_quant",
        name="matmul_rhs_group_quant",
        inputs=["A", "B", "scales", "zps"],
        outputs=["Y"],
        domain="com.nebula",  # 自定义域
        bit_width=8,           # Input tensor B int bit width
        group_size=8,          # Input tensor B group size
        a_bit_width=8,        # Input tensor A float bit width
        y_bit_width=8         # Output tensor Y float bit width
    )

    # 定义 cast节点

    cast_node = helper.make_node(
        "Cast",
        inputs=["Y"],
        outputs=["Y_casted"],
        to=TensorProto.FLOAT16  # 将 Y 转换为 FLOAT16
    )

    # 定义 Add 节点
    add_node = helper.make_node(
        "Add",
        inputs=["Y_casted", "C"],  # 使用 cast 的输出 Y_casted 和一个新输入 C
        outputs=["Z"],      # Add 的输出
    )

    # 定义输入和输出张量(动态维度版本)
    inputs = [
        helper.make_tensor_value_info(
            "A", TensorProto.FLOAT8E4M3FN, ["batch_size", 16]  # 第一维度为动态大小
        ),
        helper.make_tensor_value_info("B", TensorProto.INT8, [16, 4]),    # 输入 B
        helper.make_tensor_value_info("scales", TensorProto.FLOAT16, [16 // 8, 4]),  # B 的 scale
        helper.make_tensor_value_info("zps", TensorProto.INT8, [16 // 8, 4]),        # B 的 zero point
        helper.make_tensor_value_info("C", TensorProto.FLOAT16, ["batch_size", 4])  # Add 操作的输入 C
    ]

    outputs = [
        helper.make_tensor_value_info("Z", TensorProto.FLOAT16, ["batch_size", 4])  # Add 操作的输出 Z
    ]

    # 定义初始化器
    scales_initializer = helper.make_tensor("scales", TensorProto.FLOAT16, [16//8, 4], [0.1] * (16//8 * 4))
    zps_initializer = helper.make_tensor("zps", TensorProto.INT8, [16//8, 4], [0] * (16//8 * 4))
    c_initializer = helper.make_tensor("C", TensorProto.FLOAT16, [2, 4], [1.0] * (2 * 4))  # 修改 C 的数据类型为 FLOAT16

    # 构造图
    graph = helper.make_graph(
        nodes=[matmul_node, cast_node, add_node],
        name="matmul_rhs_group_quant_with_add_graph",
        inputs=inputs,
        outputs=outputs,
        initializer=[scales_initializer, zps_initializer, c_initializer]
    )

    # 构造模型
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("com.nebula", 1),  # 自定义域
            helper.make_opsetid("", 21)            # 默认 ONNX 域
        ]
    )

    # 保存模型
    onnx.save(model, model_path)

def test_matmul_rhs_group_quant_with_add():

    test_cases = [
        {
            "name": "matmul_rhs_group_quant",
            "model_path": "./data.ignore/matmul_rhs_group_quant_with_add_f8.onnx",
            "generate_func": generate_matmul_rhs_group_quant_f8,
            "generate_args": {},
            "check_func": test_utils.check_value_info,
            "check_args": { 
                "Y" : {"type": "FLOAT8E4M3FN", "shape": ["dynamic", 4]}
            }
        },
    ]

    test_utils.run_onnx_test(test_cases)

if __name__ == "__main__":

    test_matmul_rhs_group_quant_with_add()