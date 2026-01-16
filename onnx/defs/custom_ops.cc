#include <iterator>
#include <vector>
#include "onnx/defs/schema.h"
#include "onnx/defs/function.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/math/utils.h"
#include "shape_inference.h"

namespace ONNX_NAMESPACE {


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

void Dequant_PerGroup_ShapeInference(ONNX_NAMESPACE::InferenceContext& ctx) {
    
    // Attribute validation
    if (!hasInputShape(ctx, 0)) {
        fail_shape_inference("Input shape is required for shape inference.");
    }
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();  
    if (input_shape.dim_size() < 1) {
        fail_shape_inference("Input tensor must have at least 1 dimension.");
    }
    auto n_input_dims = input_shape.dim_size();
       
    // axis validation
    std::vector<int64_t> axis;
    if (getRepeatedAttribute(ctx, "axis", axis)) {
        if (axis.size() != size_t(n_input_dims)) {
            fail_shape_inference("Attribute 'axis' size must match input tensor rank.");
        }   
    } else {
        fail_shape_inference("Attribute 'axis' is required.");
    }
    
    // input_dtype validation 
    auto const input_dtype =
        static_cast<TensorProto_DataType>(getAttribute(ctx, "input_dtype", TensorProto::UNDEFINED));
    if (input_dtype == TensorProto::UNDEFINED) {
        fail_shape_inference("Attribute 'input_dtype' is required.");
    } else {
        const std::vector<TensorProto_DataType> allowed_types = {
            TensorProto_DataType_INT8,
            TensorProto_DataType_UINT8,
            TensorProto_DataType_INT4,
            TensorProto_DataType_UINT4,
            TensorProto_DataType_FLOAT8E4M3FN,
            TensorProto_DataType_FLOAT8E4M3FNUZ,
            TensorProto_DataType_FLOAT8E5M2,
            TensorProto_DataType_FLOAT8E5M2FNUZ
        };
        if (std::find(allowed_types.begin(), allowed_types.end(), input_dtype) == allowed_types.end()) {
            fail_shape_inference("Attribute 'input_dtype' must be one of INT8, UINT8, INT4, UINT4, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ.");
        }
        if (TensorProto_DataType_UINT4 == input_dtype) {
            if (ctx.getInputType(0)->tensor_type().elem_type() != TensorProto_DataType_UINT4 &&
                ctx.getInputType(0)->tensor_type().elem_type() != TensorProto_DataType_UINT8) {
                fail_shape_inference("Input tensor 'x' type must be UINT4 or UINT8 as specified by attribute 'input_dtype'.");
            }
        }
        if (TensorProto_DataType_INT4 == input_dtype) {
            if (ctx.getInputType(0)->tensor_type().elem_type() != TensorProto_DataType_INT4 &&
                ctx.getInputType(0)->tensor_type().elem_type() != TensorProto_DataType_INT8) {
                fail_shape_inference("Input tensor 'x' type must be INT4 or INT8 as specified by attribute 'input_dtype'.");
            }
        }
    }

    // output_dtype validation 
    auto const output_dtype =
        static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
    if (output_dtype == TensorProto::UNDEFINED) {
        fail_shape_inference("Attribute 'output_dtype' is required.");
    } else {
        const std::vector<TensorProto_DataType> allowed_types = {
            TensorProto_DataType_FLOAT,
            TensorProto_DataType_FLOAT16,
            TensorProto_DataType_FLOAT8E4M3FN,
            TensorProto_DataType_FLOAT8E4M3FNUZ,
            TensorProto_DataType_FLOAT8E5M2,
            TensorProto_DataType_FLOAT8E5M2FNUZ
        };
        if (std::find(allowed_types.begin(), allowed_types.end(), output_dtype) == allowed_types.end()) {
            fail_shape_inference("Attribute 'output_dtype' must be one of FLOAT, FLOAT16, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ.");
        }
    }

    // L1/L2/L1+L2 validation
    auto const L1 = getAttribute(ctx, "L1", static_cast<int64_t>(1));
    if (L1 != 0 && L1 != 1) {
        fail_shape_inference("Attribute L1 must be 0 or 1.");
    }
    auto const L2 = getAttribute(ctx, "L2", static_cast<int64_t>(1));
    if (L2 != 0 && L2 != 1) {
        fail_shape_inference("Attribute L2 must be 0 or 1.");
    }
    
    if (L1 && !L2) {
        std::vector<int64_t> block_size;
        if (getRepeatedAttribute(ctx, "block_size", block_size)) {
            if (block_size.size() != size_t(n_input_dims)) {
                fail_shape_inference("Attribute 'block_size' size must match input tensor rank.");
            }   
        } else {
            fail_shape_inference("Attribute 'block_size' is required.");
        }
        if (!hasInputShape(ctx, 1)) {
            fail_shape_inference("Input tensor(block_scale) shape is required for shape inference.");
        }
        auto block_scale_shape = ctx.getInputType(1)->tensor_type().shape();  
        if (block_scale_shape.dim_size() < 1) {
            fail_shape_inference("Input tensor(block_scale) must have at least 1 dimension.");
        }
        if (block_scale_shape.dim_size() != n_input_dims) {
            fail_shape_inference("Input tensor(block_scale) rank must match input tensor rank.");
        }
        for (size_t i = 0; i < axis.size(); i++) {
            if (block_scale_shape.dim(i).has_dim_value() && input_shape.dim(i).has_dim_value()) {
                int64_t expected_block_scale_dim = input_shape.dim(i).dim_value() / block_size[i];
                if (block_scale_shape.dim(i).dim_value() != expected_block_scale_dim) {
                    fail_shape_inference("Input tensor(block_scale) dimension at axis " + std::to_string(i) + " does not match expected size.");
                }
            }
        }

        // input_dtype validation for L1 only (i4//i8)
        if (input_dtype != TensorProto_DataType_INT8 && input_dtype != TensorProto_DataType_UINT8 &&
            input_dtype != TensorProto_DataType_INT4 && input_dtype != TensorProto_DataType_UINT4) {
            fail_shape_inference("For L1 dequantization only, attribute 'input_dtype' must be one of INT8, UINT8, INT4, UINT4.");
        }
    }
    
    if (!L1 && L2) {
        std::vector<int64_t> superblock_size;
        if (getRepeatedAttribute(ctx, "superblock_size", superblock_size)) {
            if (superblock_size.size() != size_t(n_input_dims)) {
                fail_shape_inference("Attribute 'superblock_size' size must match input tensor rank.");
            }   
        } else {
            fail_shape_inference("Attribute 'superblock_size' is required.");
        }
        if (!hasInputShape(ctx, 2)) {
            fail_shape_inference("Input tensor(superblock_scale)  shape is required for shape inference.");
        }
        auto superblock_scale_shape = ctx.getInputType(2)->tensor_type().shape();  
        if (superblock_scale_shape.dim_size() < 1) {
            fail_shape_inference("Input tensor(superblock_scale) must have at least 1 dimension.");
        }
        if (superblock_scale_shape.dim_size() != n_input_dims) {
            fail_shape_inference("Input tensor(superblock_scale) rank must match input tensor rank.");
        }
        for (size_t i = 0; i < axis.size(); i++) {
            if (superblock_scale_shape.dim(i).has_dim_value() && input_shape.dim(i).has_dim_value()) {
                int64_t expected_block_scale_dim = input_shape.dim(i).dim_value() / superblock_size[i];
                if (superblock_scale_shape.dim(i).dim_value() != expected_block_scale_dim) {
                    fail_shape_inference("Input tensor(superblock_scale) dimension at axis " + std::to_string(i) + " does not match expected size.");
                }
            }
        }

        // input_dtype validation for L2 only (f8)
        if (input_dtype != TensorProto_DataType_FLOAT8E4M3FN && input_dtype != TensorProto_DataType_FLOAT8E4M3FNUZ &&
            input_dtype != TensorProto_DataType_FLOAT8E5M2 && input_dtype != TensorProto_DataType_FLOAT8E5M2FNUZ) {
            fail_shape_inference("For L2 dequantization only, attribute 'input_dtype' must be one of FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ.");
        }
    }

    if (L1 && L2) {
        std::vector<int64_t> block_size;
        if (getRepeatedAttribute(ctx, "block_size", block_size)) {
            if (block_size.size() != size_t(n_input_dims)) {
                fail_shape_inference("Attribute 'block_size' size must match input tensor rank.");
            }   
        } else {
            fail_shape_inference("Attribute 'block_size' is required.");
        }
        if (!hasInputShape(ctx, 1)) {
            fail_shape_inference("Input tensor(block_scale) shape is required for shape inference.");
        }
        auto block_scale_shape = ctx.getInputType(1)->tensor_type().shape();  
        if (block_scale_shape.dim_size() < 1) {
            fail_shape_inference("Input tensor(block_scale) must have at least 1 dimension.");
        }
        if (block_scale_shape.dim_size() != n_input_dims) {
            fail_shape_inference("Input tensor(block_scale) rank must match input tensor rank.");
        }
        for (size_t i = 0; i < axis.size(); i++) {
            if (block_scale_shape.dim(i).has_dim_value() && input_shape.dim(i).has_dim_value()) {
                int64_t expected_block_scale_dim = input_shape.dim(i).dim_value() / block_size[i];
                if (block_scale_shape.dim(i).dim_value() != expected_block_scale_dim) {
                    fail_shape_inference("Input tensor(block_scale) dimension at axis " + std::to_string(i) + " does not match expected size.");
                }
            }
        }

        std::vector<int64_t> superblock_size;
        if (getRepeatedAttribute(ctx, "superblock_size", superblock_size)) {
            if (superblock_size.size() != size_t(n_input_dims)) {
                fail_shape_inference("Attribute 'superblock_size' size must match input tensor rank.");
            }   
        } else {
            fail_shape_inference("Attribute 'superblock_size' is required.");
        }
        if (!hasInputShape(ctx, 2)) {
            fail_shape_inference("Input tensor(superblock_scale)  shape is required for shape inference.");
        }
        auto superblock_scale_shape = ctx.getInputType(2)->tensor_type().shape();  
        if (superblock_scale_shape.dim_size() < 1) {
            fail_shape_inference("Input tensor(superblock_scale) must have at least 1 dimension.");
        }
        if (superblock_scale_shape.dim_size() != n_input_dims) {
            fail_shape_inference("Input tensor(superblock_scale) rank must match input tensor rank.");
        }

        for (size_t i = 0; i < axis.size(); i++) {
            if (superblock_scale_shape.dim(i).has_dim_value() && input_shape.dim(i).has_dim_value()) {
                int64_t expected_block_scale_dim = input_shape.dim(i).dim_value() / (block_size[i]*superblock_size[i]);
                if (superblock_scale_shape.dim(i).dim_value() != expected_block_scale_dim) {
                    fail_shape_inference("Input tensor(superblock_scale) dimension at axis " + std::to_string(i) + " does not match expected size.");
                }
            }
        }

        // input_dtype validation for L1 only (i4//i8)
        if (input_dtype != TensorProto_DataType_INT8 && input_dtype != TensorProto_DataType_UINT8 &&
            input_dtype != TensorProto_DataType_INT4 && input_dtype != TensorProto_DataType_UINT4) {
            fail_shape_inference("For L1 dequantization only, attribute 'input_dtype' must be one of INT8, UINT8, INT4, UINT4.");
        }

        if (ctx.getInputType(1)->tensor_type().elem_type() != TensorProto_DataType_INT8 &&
            ctx.getInputType(1)->tensor_type().elem_type() != TensorProto_DataType_UINT8 &&
            ctx.getInputType(1)->tensor_type().elem_type() != TensorProto_DataType_INT4 &&
            ctx.getInputType(1)->tensor_type().elem_type() != TensorProto_DataType_UINT4) {
            fail_shape_inference("Input tensor 'block_scale' type must be one of INT8, UINT8, INT4, UINT4 as specified by attribute 'input_dtype'.");
        }
    }

    {
        auto const output_dtype =
        static_cast<TensorProto_DataType>(getAttribute(ctx, "output_dtype", TensorProto::UNDEFINED));
        if (output_dtype != TensorProto::UNDEFINED) {
            propagateElemTypeFromAttributeToOutput(ctx, "output_dtype", 0);
        } else {
            assert(false); // should never reach here due to previous checks
            propagateElemTypeFromInputToOutput(ctx, 1, 0);
        }
        if (!hasInputShape(ctx, 0)) {
            return;
        }
        auto& input_shape = getInputShape(ctx, 0);
        updateOutputShape(ctx, 0, input_shape); 
    }
    
}

    
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

    static constexpr const char* Dequant_PerGroup_ver1_doc = R"DOC(
        The linear dequantization operator.
        )DOC";

    ONNX_OPERATOR_SCHEMA(Dequant_PerGroup)
            .SetDomain(COM_EXAMPLE_DOMAIN)
            .SinceVersion(1)
            .Input(0, "x", "N-D quantized input tensor to be de-quantized.", "T1")
            .Input(1, "block_scale", "Block scale for input `x`.", "T2")
            .Input(2, "superblock_scale", "Superblock scale for input `x`.", "T3")
            .Output(0, "y", "N-D full precision output tensor. It has the same shape as input `x`.", "T4")
            .Attr("axis","The axis of the dequantizing dimension of the input tensor.", AttributeProto::INTS, OPTIONAL_VALUE)
            .Attr("block_size", "The size of the quantization block.", AttributeProto::INTS, OPTIONAL_VALUE)
            .Attr("superblock_size", "The size of the superblock for quantization.", AttributeProto::INTS, OPTIONAL_VALUE)
            .Attr("L1", "(Optional) Enable flag for L1/first-level quantization.", AttributeProto::INT, static_cast<int64_t>(1))
            .Attr("L2", "(Optional) Enable flag for L2/first-level quantization.", AttributeProto::INT, static_cast<int64_t>(1))
            .Attr("input_dtype", "(Optional) The input data type.", AttributeProto::INT, static_cast<int64_t>(0))
            .Attr("output_dtype", "(Optional) The output data type.", AttributeProto::INT, static_cast<int64_t>(0))
            .Attr("round_type", "(Optional) The round type.", AttributeProto::INT, static_cast<int64_t>(0))
            .TypeConstraint("T1",{"tensor(int8)", "tensor(uint8)", "tensor(float8e4m3fn)", "tensor(float8e4m3fnuz)", "tensor(float8e5m2)", "tensor(float8e5m2fnuz)", "tensor(uint4)", "tensor(int4)"}, "The type of the inputs 'x'.")
            .TypeConstraint("T2", {"tensor(int8)", "tensor(uint8)", "tensor(uint4)", "tensor(int4)"}, "The type of the input 'block_scale'.")
            .TypeConstraint("T3", {"tensor(float16)"}, "The type of the output 'superblock_size'.")
            .TypeConstraint("T4", {"tensor(float)", "tensor(float16)"}, "The type of the output 'y'.")
            .SetDoc(Dequant_PerGroup_ver1_doc)
            .TypeAndShapeInferenceFunction(Dequant_PerGroup_ShapeInference);

    

}  // namespace ONNX_NAMESPACE

