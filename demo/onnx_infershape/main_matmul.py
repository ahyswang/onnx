import onnx
from onnx import helper, TensorProto

# 构造包含 QLinearMatMul 算子的简单图

# 前置 QLinearAdd 节点
add1_node = helper.make_node(
    "QLinearAdd",
    inputs=["X", "A"],
    outputs=["X_add"],
    domain="com.example"  # 使用自定义域
)

# 定义固定的常量值
a_scale = helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.1])
a_zero_point = helper.make_tensor("a_zero_point", TensorProto.INT8, [], [0])
b_scale = helper.make_tensor("b_scale", TensorProto.FLOAT, [], [0.2])
b_zero_point = helper.make_tensor("b_zero_point", TensorProto.INT8, [], [0])
y_scale = helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.3])
y_zero_point = helper.make_tensor("y_zero_point", TensorProto.INT8, [], [0])

# QLinearMatMul 算子节点
node = helper.make_node(
    "QLinearMatMul",  # 替换为 QLinearMatMul 算子
    inputs=["X_add", "a_scale", "a_zero_point", "B", "b_scale", "b_zero_point", "y_scale", "y_zero_point"],
    outputs=["Y_custom"],
    domain="com.example"  # 使用自定义域
)

# 后置 QLinearAdd 节点
add2_node = helper.make_node(
    "QLinearAdd",
    inputs=["Y_custom", "C"],
    outputs=["Y"],
    domain="com.example"  # 使用自定义域
)

graph = helper.make_graph(
    [add1_node, node, add2_node],
    "qlinear_matmul_graph",
    inputs=[
        helper.make_tensor_value_info("X", TensorProto.INT8, [3, 5]),
        helper.make_tensor_value_info("A", TensorProto.INT8, [3, 5]),
        helper.make_tensor_value_info("B", TensorProto.INT8, [5, 3]),
        helper.make_tensor_value_info("C", TensorProto.INT8, [3, 3]),
    ],
    outputs=[
        helper.make_tensor_value_info("Y", TensorProto.INT8, [3, 3])
    ],
    initializer=[
        a_scale, a_zero_point, b_scale, b_zero_point, y_scale, y_zero_point  # 添加常量到初始化器
    ]
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.example", 1),  # COM_EXAMPLE_DOMAIN = "com.example"
                                                helper.make_opsetid("", 13)])  # ONNX_DOMAIN = ""
onnx.save(model, "test_qlinearmatmul_with_constants.onnx")

# 尝试加载
try:
    m = onnx.load("test_qlinearmatmul_with_constants.onnx")
    print("模型加载成功，包含 QLinearMatMul 算子！")
    onnx.checker.check_model(m, full_check=True)
    print("模型检查成功，包含 QLinearMatMul 算子！")
    inferred_model = onnx.shape_inference.infer_shapes(m)
    onnx.save_model(inferred_model, "test_qlinearmatmul_with_constants_inferred.onnx")
    print("模型推形状成功，包含 QLinearMatMul 算子！")
except Exception as e:
    print("加载失败: ", e)