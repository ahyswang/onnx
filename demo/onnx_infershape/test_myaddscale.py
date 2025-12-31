import onnx
from onnx import helper, TensorProto

import sys 
import os 
from test_utils import check_node_output_type_and_shape

os.makedirs("./data.ignore", exist_ok=True)

# 构造包含自定义算子的简单图
def test_myaddscale():
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
    onnx.save(model, "./data.ignore/test_myaddscale.onnx")

    # 尝试加载
    try:
        m = onnx.load("./data.ignore/test_myaddscale.onnx")
        print("模型加载成功，包含自定义算子！")
        onnx.checker.check_model(m, full_check=True)
        print("模型检查成功，包含自定义算子！")
        inferred_model = onnx.shape_inference.infer_shapes(m)
        onnx.save_model(inferred_model, "./data.ignore/test_myaddscale_inferred.onnx")
        print("模型推形状成功，包含自定义算子！")

        output_info = check_node_output_type_and_shape(
            "./data.ignore/test_myaddscale_inferred.onnx",
            "MyAddScale"
        )
        print(output_info)
        assert output_info["Y_custom"]["type"] == "FLOAT", "输出类型不匹配，预期为 FLOAT"
        assert output_info["Y_custom"]["shape"] == [3, 5], "输出形状不匹配，预期为 [dynamic, 4]"

    except Exception as e:
        print("加载失败: ", e)