import onnx
from onnx import helper, TensorProto

# 定义 matmul_rhs_group_quant 节点
matmul_node = helper.make_node(
    "matmul_rhs_group_quant",
    inputs=["A", "B", "scales", "zps"],
    outputs=["Y"],
    domain="com.example",  # 自定义域
    bit_width=8,           # Input tensor B int bit width
    group_size=8,          # Input tensor B group size
    a_bit_width=16,        # Input tensor A float bit width
    y_bit_width=16         # Output tensor Y float bit width
)

# 定义 Add 节点
add_node = helper.make_node(
    "Add",
    inputs=["Y", "C"],  # 使用 matmul_rhs_group_quant 的输出 Y 和一个新输入 C
    outputs=["Z"],      # Add 的输出
)

# 定义输入和输出张量
inputs = [
    helper.make_tensor_value_info("A", TensorProto.FLOAT16, [2, 16]),  # 输入 A
    helper.make_tensor_value_info("B", TensorProto.INT8, [16, 4]),    # 输入 B
    helper.make_tensor_value_info("scales", TensorProto.FLOAT16, [16//8,4]),  # B 的 scale
    helper.make_tensor_value_info("zps", TensorProto.INT8, [16//8,4]),     # B 的 zero point
    helper.make_tensor_value_info("C", TensorProto.FLOAT16, [2, 4])  # Add 操作的输入 C
]

outputs = [
    helper.make_tensor_value_info("Z", TensorProto.FLOAT16, [2, 4])  # Add 操作的输出 Z
]

# 定义初始化器
scales_initializer = helper.make_tensor("scales", TensorProto.FLOAT16, [16//8, 4], [0.1] * (16//8 * 4))
zps_initializer = helper.make_tensor("zps", TensorProto.INT8, [16//8, 4], [0] * (16//8 * 4))
c_initializer = helper.make_tensor("C", TensorProto.FLOAT16, [2, 4], [1.0] * (2 * 4))  # 初始化 C 为全 1

# 构造图
graph = helper.make_graph(
    nodes=[matmul_node, add_node],
    name="matmul_rhs_group_quant_with_add_graph",
    inputs=inputs,
    outputs=outputs,
    initializer=[scales_initializer, zps_initializer, c_initializer]
)

# 构造模型
model = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("com.example", 1),  # 自定义域
        helper.make_opsetid("", 13)            # 默认 ONNX 域
    ]
)

# 保存模型
onnx.save(model, "matmul_rhs_group_quant_with_add.onnx")
print("模型已保存为 matmul_rhs_group_quant_with_add.onnx")

# 尝试加载和推导形状
try:
    loaded_model = onnx.load("matmul_rhs_group_quant_with_add.onnx")
    print("模型加载成功，包含 matmul_rhs_group_quant 和 Add 算子！")
    onnx.checker.check_model(loaded_model, full_check=True)
    print("模型检查成功！")
    inferred_model = onnx.shape_inference.infer_shapes(loaded_model)
    onnx.save_model(inferred_model, "matmul_rhs_group_quant_with_add_inferred.onnx")
    print("形状推导成功，已保存为 matmul_rhs_group_quant_with_add_inferred.onnx")
except Exception as e:
    print("加载或推导形状失败: ", e)