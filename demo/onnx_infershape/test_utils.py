import onnx
from onnx import helper, TensorProto
import sys 
import os 


def check_node_output_type_and_shape(model_path, node_name):
    """
    检查 ONNX 模型中指定节点的输出类型和形状。

    Args:
        model_path (str): ONNX 模型文件路径。
        node_name (str): 要检查的节点名称。

    Returns:
        dict: 输出名称及其对应的类型和形状。
    """
    try:
        # 加载模型
        model = onnx.load(model_path)
        print(f"成功加载模型: {model_path}")

        # 遍历图中的节点
        for node in model.graph.node:
            if node.name == node_name:
                print(f"找到节点: {node_name}")
                output_info = {}
                for output in node.output:
                    # 查找输出的类型和形状信息
                    for value_info in model.graph.value_info:
                        if value_info.name == output:
                            elem_type = onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
                            shape = [
                                dim.dim_value if dim.HasField("dim_value") else "dynamic"
                                for dim in value_info.type.tensor_type.shape.dim
                            ]
                            output_info[output] = {"type": elem_type, "shape": shape}
                            break
                return output_info

        print(f"未找到节点: {node_name}")
        return None
    except Exception as e:
        print(f"检查失败: {e}")
        return None
  