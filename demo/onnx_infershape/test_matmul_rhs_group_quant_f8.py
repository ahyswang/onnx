import onnx
from onnx import helper, TensorProto
import sys 
import os 
from test_utils import check_node_output_type_and_shape

os.makedirs("./data.ignore", exist_ok=True)

# 定义 matmul_rhs_group_quant 节点
def test_matmul_rhs_group_quant_f8():

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
    onnx.save(model, "./data.ignore/matmul_rhs_group_quant_with_add_f8.onnx")
    print("模型已保存为 matmul_rhs_group_quant_with_add_f8.onnx")

    # 尝试加载和推导形状
    try:
        loaded_model = onnx.load("./data.ignore/matmul_rhs_group_quant_with_add_f8.onnx")
        print("模型加载成功，包含 matmul_rhs_group_quant 和 Add 算子！")
        onnx.checker.check_model(loaded_model, full_check=True)
        print("模型检查成功！")
        inferred_model = onnx.shape_inference.infer_shapes(loaded_model)
        onnx.save_model(inferred_model, "./data.ignore/matmul_rhs_group_quant_with_add_inferred_f8.onnx")
        print("形状推导成功，已保存为 matmul_rhs_group_quant_with_add_inferred_f8.onnx")

        output_info = check_node_output_type_and_shape(
            "./data.ignore/matmul_rhs_group_quant_with_add_inferred_f8.onnx",
            "matmul_rhs_group_quant"
        )
        print(output_info)
        assert output_info["Y"]["type"] == "FLOAT8E4M3FN", "输出类型不匹配，预期为 FLOAT16"
        assert output_info["Y"]["shape"] == ["dynamic", 4], "输出形状不匹配，预期为 [dynamic, 4]"

    except Exception as e:
        print("加载或推导形状失败: ", e)