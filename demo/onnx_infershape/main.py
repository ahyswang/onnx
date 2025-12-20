import onnx
from onnx import helper, TensorProto

# 构造包含自定义算子的简单图

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
    inputs=["X_add", "scale"],     
    outputs=["Y_custom"],         
    domain="com.example"
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
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.example", 1),  # COM_EXAMPLE_DOMAIN = "com.example"
                                                helper.make_opsetid("", 13)]) # ONNX_DOMAIN = ""
onnx.save(model, "test_myaddscale.onnx")

# 尝试加载
try:
    m = onnx.load("test_myaddscale.onnx")
    print("模型加载成功，包含自定义算子！")
    onnx.checker.check_model(m, full_check=True)
    print("模型检查成功，包含自定义算子！")
    inferred_model = onnx.shape_inference.infer_shapes(m)
    onnx.save_model(inferred_model, "test_myaddscale_inferred.onnx")
    print("模型推形状成功，包含自定义算子！")
except Exception as e:
    print("加载失败: ", e)