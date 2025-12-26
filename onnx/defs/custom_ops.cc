#include "onnx/defs/schema.h"
#include "onnx/defs/function.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/math/utils.h"
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

ONNX_OPERATOR_SCHEMA(QLinearAdd)
    .SetDomain("com.example")
    .SinceVersion(1)
    .Input(0, "A", "Input tensor A", "T")
    .Input(1, "B", "Input tensor B", "T")
    .Output(0, "C", "Output tensor", "T")
    .TypeConstraint("T", {"tensor(int8)"}, "Supports int8 tensors")
    .SetDoc("Custom Add operator for int8 tensors")
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        if (hasInputShape(ctx, 0)) {
            propagateShapeAndTypeFromFirstInput(ctx);
        }
    });

ONNX_OPERATOR_SCHEMA(QLinearMatMul)
    .SetDomain(COM_EXAMPLE_DOMAIN)
    .SinceVersion(1)
    .SetDoc("Quantized matrix multiplication of two N-dimensional matrices a and b with "
            "scales and zero points for inputs and output.")
    .Input(0, "a", "N-dimensional quantized matrix a", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
    .Input(1, "a_scale", "scale of quantized input a", "TS", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
    .Input(
        2,
        "a_zero_point",
        "zero point of quantized input a",
        "T1",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable)
    .Input(3, "b", "N-dimensional quantized matrix b", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
    .Input(4, "b_scale", "scale of quantized input b", "TS", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
    .Input(
        5,
        "b_zero_point",
        "zero point of quantized input b",
        "T2",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable)
    .Input(
        6,
        "y_scale",
        "scale of quantized output y",
        "TS",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable)
    .Input(
        7,
        "y_zero_point",
        "zero point of quantized output y",
        "T3",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable)
    .Output(
        0,
        "y",
        "Quantized matrix multiply results from a * b",
        "T3",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable)
    .TypeConstraint("TS", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain scales.")
    .TypeConstraint(
        "T1",
        {"tensor(int8)",
            "tensor(uint8)",
            "tensor(float8e4m3fn)",
            "tensor(float8e4m3fnuz)",
            "tensor(float8e5m2)",
            "tensor(float8e5m2fnuz)"},
        "The type of input a and its zeropoint.")
    .TypeConstraint(
        "T2",
        {"tensor(int8)",
            "tensor(uint8)",
            "tensor(float8e4m3fn)",
            "tensor(float8e4m3fnuz)",
            "tensor(float8e5m2)",
            "tensor(float8e5m2fnuz)"},
        "The type of input b and its zeropoint.")
    .TypeConstraint(
        "T3",
        {"tensor(int8)",
            "tensor(uint8)",
            "tensor(float8e4m3fn)",
            "tensor(float8e4m3fnuz)",
            "tensor(float8e5m2)",
            "tensor(float8e5m2fnuz)"},
        "The type of the output and its zeropoint.")
    .TypeAndShapeInferenceFunction(defs::math::utils::QLinearMatMulShapeInference);

}  // namespace ONNX_NAMESPACE

