import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def pack_int4_values(values):
    """将一个 int4 数值列表打包成一个 numpy uint8 数组。"""
    if len(values) % 2 != 0:
        values.append(0)  # 填充以确保偶数长度
    
    packed_bytes = []
    for i in range(0, len(values), 2):
        val1 = values[i]
        val2 = values[i+1]
        
        # 将 Python int 转换为 4-bit 补码表示 (存储为 uint8)
        byte1 = val1 & 0x0F
        byte2 = val2 & 0x0F

        # 将两个 4-bit 值打包到一个字节中
        packed_byte = (byte1 << 4) | byte2
        packed_bytes.append(packed_byte)
        
    return np.array(packed_bytes, dtype=np.uint8)

def generate_int4_tensor(name, shape, data=None, pack_int4=False):
    """生成一个 INT4 TensorProto"""
    if data is None:
        data = np.ones(shape, np.int8)
    
    if pack_int4:
        packed_numpy_array = pack_int4_values(data.flatten().tolist())
    else:
        packed_numpy_array = data.flatten().astype(np.uint8)

    block_scale = numpy_helper.from_array(
        packed_numpy_array, 
        name=name
    )
    
    block_scale.dims.clear()
    block_scale.dims.extend(shape)
    block_scale.data_type = TensorProto.INT4

    return block_scale

def generator_onnx_tensor(name, shape, dtype=TensorProto.FLOAT, data=None):
    """生成一个指定数据类型的 ONNX TensorProto"""
    if data is None:
        if dtype == TensorProto.INT8 or dtype == TensorProto.INT4:
            data = np.ones(shape, np.int8)
            onnx_tensor = numpy_helper.from_array(
                data.flatten().astype(np.int8), 
                name=name
            )
        elif dtype == TensorProto.FLOAT:
            data = np.ones(shape, np.float32)
            onnx_tensor = numpy_helper.from_array(
                data.flatten().astype(np.float32), 
                name=name
            )
        elif dtype == TensorProto.FLOAT16:
            data = np.ones(shape, np.float16)
            onnx_tensor = numpy_helper.from_array(
                data.flatten().astype(np.float16), 
                name=name
            )
        elif dtype == TensorProto.FLOAT8E4M3FN or dtype == TensorProto.FLOAT8E5M2 or dtype == TensorProto.FLOAT8E4M3FNUZ or dtype == TensorProto.FLOAT8E5M2FNUZ:
            data = np.ones(shape, np.int8)
            onnx_tensor = numpy_helper.from_array(
                data.flatten().astype(np.int8), 
                name=name
            )
        else:
            raise ValueError(f"不支持的数据类型: {dtype}")
    else:
        onnx_tensor = numpy_helper.from_array(
            data.flatten(), 
            name=name
        )
    
    onnx_tensor.dims.clear()
    onnx_tensor.dims.extend(shape)
    onnx_tensor.data_type = dtype

    return onnx_tensor

def infer_shape_and_save(model_path, inferred_path):
    """
    对 ONNX 模型进行形状推断并保存结果。
    """
    m = onnx.load(model_path)

    onnx.checker.check_model(m, full_check=True, check_custom_domain=True)

    inferred_model = onnx.shape_inference.infer_shapes(m)
    onnx.save_model(inferred_model, inferred_path)

def get_value_info_by_tensor(model_path, tensor_name):
    """
    检查 ONNX 模型中指定节点的输出类型和形状。

    Args:
        model_path (str): ONNX 模型文件路径。
        tensor_name (str): 要查找的向量名称。

    Returns:
        dict: 输出名称及其对应的类型和形状。
    """
    try:
        # 加载模型
        model = onnx.load(model_path)
        # 遍历图中的节点
        tensor_info = None
        # 查找输出的类型和形状信息
        for io_value_info in [model.graph.input, model.graph.output]:
            for input_info in model.graph.input:
                if input_info.name == tensor_name:
                    elem_type = onnx.TensorProto.DataType.Name(input_info.type.tensor_type.elem_type)
                    shape = [
                        dim.dim_value if dim.HasField("dim_value") else "dynamic"
                        for dim in input_info.type.tensor_type.shape.dim
                    ]
                    tensor_info = {"type": elem_type, "shape": shape}
                    break

        for value_info in model.graph.value_info:
            if value_info.name == tensor_name:
                elem_type = onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
                shape = [
                    dim.dim_value if dim.HasField("dim_value") else "dynamic"
                    for dim in value_info.type.tensor_type.shape.dim
                ]
                tensor_info = {"type": elem_type, "shape": shape}
                break

        assert tensor_info is not None, f"未找到张量: {tensor_name}"
        print(f"name: {tensor_name},  tensor_info: {tensor_info}")
        return tensor_info
    
    except Exception as e:
        print(f"检查失败: {e}")
        return None
    
def check_value_info(model_path, check_value_infos = {}) -> bool:
    """
    检查 ONNX 模型中指定节点的输出类型和形状。
    Args:
        model_path (str): ONNX 模型文件路径。
        check_value_infos (dict): 要检查的节点名称及其预期类型和形状。
    Returns:
        bool: 检查是否通过。
    """
    for tensor_name, expected_info in check_value_infos.items():
        actual_info = get_value_info_by_tensor(model_path, tensor_name)
        if actual_info != expected_info:
            print(f"检查失败: 张量 {tensor_name} 预期 {expected_info}，实际 {actual_info}")
            return False
        else:
            print(f"检查通过: 张量 {tensor_name} 类型和形状符合预期")
    return True

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
  
def run_onnx_test(test_cases):
    """
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
    """
    for case in test_cases:
        print(f"\n=== Test: {case['name']} ===")
        model_path = case["model_path"]
        inferred_path = model_path.replace(".onnx", "_inferred.onnx")
        # 生成模型
        case["generate_func"](model_path, **case["generate_args"])
        # 形状推断
        infer_shape_and_save(model_path, inferred_path)
        # 检查结果
        case["check_func"](inferred_path, case["check_args"])