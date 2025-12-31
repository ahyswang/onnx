import onnx
from onnx import helper, TensorProto

def update_node_attr(model_path, save_path, target_node_name="input1_QuantizeLinear", target_attribute_name="output_dtype", new_attribute_value=3): 
    # 加载 ONNX 模型
    model = onnx.load(model_path)

    # 获取模型的图
    graph = model.graph

    # 找到目标节点并修改属性值
    # target_node_name = "TargetNodeName"  # 替换为目标节点的名称
    # target_attribute_name = "TargetAttributeName"  # 替换为目标属性的名称
    # new_attribute_value = 42  # 替换为新的属性值

    for node in graph.node:
        if node.name == target_node_name:
            for attr in node.attribute:
                if attr.name == target_attribute_name:
                    # 修改属性值
                    attr.i = new_attribute_value  # 假设属性是整数类型
                    print(f"Updated attribute '{target_attribute_name}' to {new_attribute_value} in node '{target_node_name}'")
                    break
            else:
                print(f"Attribute '{target_attribute_name}' not found in node '{target_node_name}'")
            break
    else:
        print(f"Node '{target_node_name}' not found in the graph")

    # 保存修改后的模型
    onnx.save(model, save_path)
# 尝试加载
try:
    update_node_attr("conv2d_per_channel_qdq_modified.onnx", "conv2d_per_channel_qdq_modified.update_attr.onnx")
    update_node_attr("conv2d_per_channel_qdq_modified.update_attr.onnx", "conv2d_per_channel_qdq_modified.update_attr2.onnx", target_node_name="output_QuantizeLinear")

    #m = onnx.load("conv2d_per_channel_qdq_modified.onnx")
    m = onnx.load("conv2d_per_channel_qdq_modified.update_attr2.onnx")
    print("模型加载成功，包含 QLinearMatMul 算子！")
    # onnx.checker.check_model(m, full_check=True)
    # print("模型检查成功，包含 QLinearMatMul 算子！")
    inferred_model = onnx.shape_inference.infer_shapes(m)
    onnx.save_model(inferred_model, "conv2d_per_channel_qdq_modified_inferred.onnx")
    print("模型推形状成功，包含 QLinearMatMul 算子！")

except Exception as e:
    print("加载失败: ", e)