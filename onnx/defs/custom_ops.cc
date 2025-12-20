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