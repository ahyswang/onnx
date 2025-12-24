# 说明

## 源码编译

1. 准备环境。

```
# Python 3.12.10
python -m venv /workspace/yswang26/python_env/onnx_venv
source /workspace/yswang26/python_env/onnx_venv/bin/activate
```

2. 下载源码。

```
git clone git@github.com:onnx/onnx.git
git checkout e80f4ce7d47c409dc6e28a7e47fc64856d2a660a
```

3. 编译源码（下载需要翻墙依赖包）。

```
pip install -e .
```

4. 验证测试。

```
python main.py
```

## 添加自定义算子

1. 增加com.example的域名。

```
# onnx/onnx/common/constants.h line:19
constexpr const char* COM_EXAMPLE_DOMAIN = "com.example";
```

```
# /workspace/yswang26/onnx/onnx/defs/schema.h line:921
map_[COM_EXAMPLE_DOMAIN] = std::make_pair(1, 26);
# /workspace/yswang26/onnx/onnx/defs/schema.h line:932
last_release_version_map_[COM_EXAMPLE_DOMAIN] = 26;
```

2. 增加自定义算子的定义和推导形状。

```
#/workspace/yswang26/onnx/onnx/defs/custom_ops.cc
#include "onnx/defs/schema.h"
#include "onnx/defs/function.h"
#include "onnx/defs/shape_inference.h"
#include "shape_inference.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SCHEMA(MyAddScale)
    .SetDomain(COM_EXAMPLE_DOMAIN)
    .SinceVersion(1)
    .Input(0, "A", "Input tensor", "T")
    .Input(1, "scale", "Scalar to add", "T")
    .Output(0, "O", "Output tensor", "T")
    .TypeConstraint("T", {"tensor(float)"}, "Only float tensors supported")
    .SetDoc("Add scale to tensor")
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        // 将输入A的shape复制给输出O
        if (hasInputShape(ctx, 0)) {
            updateOutputShape(ctx, 0, ctx.getInputType(0)->tensor_type().shape());
            updateOutputElemType(ctx, 0, ctx.getInputType(0)->tensor_type().elem_type());
        }
    });
}  // namespace ONNX_NAMESPACE
```

## 问题

1. 自定义domain和内置domain逻辑不清晰，待确定。