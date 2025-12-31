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
    .SetDomain(COM_EXAMPLE_DOMAIN)
    .SinceVersion(1)
    .SetDoc("Quantized add of two N-dimensional tensor a and b with "
            "scales and zero points for inputs and output.")
    .Input(0, "a", "N-dimensional quantized tensor a", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
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
    .Input(3, "b", "N-dimensional quantized tensor b", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
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
        "Quantized tensor add results from a * b",
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
    .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
        if (hasInputShape(ctx, 0)) {
            propagateShapeAndTypeFromFirstInput(ctx);
        }
    });

void matmul_rhs_group_quant_ShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    const auto* const a_type = ctx.getInputType(0);
    const auto* const b_type = ctx.getInputType(1);
    if (nullptr == a_type || nullptr == b_type || a_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType ||
        b_type->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
        fail_type_inference("inputs are expected to have tensor type.");
    }

    auto groupSizeAttr = ctx.getAttribute("group_size");
    // if axis is not defined
    if (!groupSizeAttr) {
        fail_shape_inference("Required attribute group_size is missing");
    }
    int group_size = static_cast<int>(groupSizeAttr->i());
    if (group_size <= 0) {
        fail_shape_inference("group_size must be a positive integer.");
    }

    const auto* const b_scales_type = ctx.getInputType(2);
    if (nullptr == b_scales_type || 
        b_scales_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto::FLOAT16) {
        fail_type_inference("scales input is expected to have float16 type.");
    }
    if (b_scales_type->tensor_type().shape().dim_size() != 2) {
        fail_type_inference("scales input is expected to be 2D tensor.");
    }
    if (!(b_scales_type->tensor_type().shape().dim(0).has_dim_value() && b_type->tensor_type().shape().dim(0).has_dim_value() && 
        b_scales_type->tensor_type().shape().dim(0).dim_value() == b_type->tensor_type().shape().dim(0).dim_value() / group_size)) {
        fail_type_inference("scales input's first dimension is expected to match the group size.");
    }
    if (!(b_scales_type->tensor_type().shape().dim(1).has_dim_value() && b_type->tensor_type().shape().dim(1).has_dim_value() && 
        b_scales_type->tensor_type().shape().dim(1).dim_value() == b_type->tensor_type().shape().dim(1).dim_value())) {
        fail_type_inference("scales input's second dimension is expected to match the input B's second dimension.");
    }
    
    const auto* const b_zero_point_type = ctx.getInputType(3);
    if (nullptr == b_zero_point_type ||
        b_zero_point_type->tensor_type().elem_type() != b_type->tensor_type().elem_type()) {
        fail_type_inference("input and zero_point pair is expected to have same type.");
    }
    if (b_zero_point_type->tensor_type().shape().dim_size() != 2) {
        fail_type_inference("zero point input is expected to be 2D tensor.");
    }
    if (!(b_zero_point_type->tensor_type().shape().dim(0).has_dim_value() && b_type->tensor_type().shape().dim(0).has_dim_value() && 
        b_zero_point_type->tensor_type().shape().dim(0).dim_value() == b_type->tensor_type().shape().dim(0).dim_value() / group_size)) {
        fail_type_inference("scales input's first dimension is expected to match the group size.");
    }
    if (!(b_zero_point_type->tensor_type().shape().dim(1).has_dim_value() && b_type->tensor_type().shape().dim(1).has_dim_value() && 
        b_zero_point_type->tensor_type().shape().dim(1).dim_value() == b_type->tensor_type().shape().dim(1).dim_value())) {
        fail_type_inference("zero point input's second dimension is expected to match the input B's second dimension.");
    }

    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    
    onnx::defs::math::utils::MatMulShapeInference(ctx, 0, 1);
}
    
ONNX_OPERATOR_SCHEMA(matmul_rhs_group_quant)
    .SetDomain(COM_EXAMPLE_DOMAIN)
    .SinceVersion(1)
    .Attr("bit_width", "Input tensor B int bit width", AttributeProto::INT, static_cast<int64_t>(8))
    .Attr("group_size", "Input tensor B group size", AttributeProto::INT, static_cast<int64_t>(8))
    .Attr("a_bit_width", "Input tensor a float bit width", AttributeProto::INT, static_cast<int64_t>(16))
    .Attr("y_bit_width", "Output tensor y float bit width", AttributeProto::INT, static_cast<int64_t>(16))
    .Input(0, "a", "Input tensor A", "TA")
    .Input(1, "b", "Input tensor B", "TB")
    .Input(2, "scales", "Scale tensor for input B", "TS")
    .Input(3, "zps", "Zero point tensor for input B", "TZ")
    .Output(0, "y", "Output tensor", "TY")
    .TypeConstraint("TA", {"tensor(float16)", "tensor(float8e4m3fn)"}, "The type of input a.")
    .TypeConstraint("TB", {"tensor(int8)"}, "The type of input b.")
    .TypeConstraint("TY", {"tensor(float16)", "tensor(float8e4m3fn)"}, "The type of output y.")
    .TypeConstraint("TS", {"tensor(float16)"}, "The type of input b scales.")
    .TypeConstraint("TZ", {"tensor(int8)"}, "The type of input b zeropoint.")
    .SetDoc("Custom matmul operator for quantized tensors with per-channel quantization on the right-hand side")
    .TypeAndShapeInferenceFunction(matmul_rhs_group_quant_ShapeInference);


}  // namespace ONNX_NAMESPACE

