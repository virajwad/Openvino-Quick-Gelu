#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "exceptions.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/multiply.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector quickgelu(const ov::frontend::onnx::Node& node) {
    // Original Documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QuickGelu
    // Goal: Compute x * Sigmoid(alpha * x)

    const auto inputs = node.get_ov_inputs();

    // Only one input (x) so give a check
    auto num_inputs = inputs.size();
    FRONT_END_GENERAL_CHECK(num_inputs == 1, "QuickGelu takes only 1 input but was provided " + std::to_string(num_inputs));
    const auto& x = inputs[0];

    // Constrain input type to float16, float, double (f64), bfloat16
    CHECK_VALID_NODE(node,
        x.get_element_type() == ov::element::f16 || x.get_element_type() == ov::element::f32 || x.get_element_type() == ov::element::f64 || x.get_element_type() == ov::element::bf16,
        "Unsupported input x type, accepted FP16, FP32, FP64, BFP16 but got: ", x.get_element_type());

    // Get attribute from node
    const float alpha = node.get_attribute_value<float>("alpha");

    // TODO: Check accuracy / validity of below
    auto alpha_x = std::make_shared<v1::Multiply>(alpha, x);
    auto sig_alpha_x = std::make_shared<v0::Sigmoid>(alpha_x);
    auto result = std::make_shared<v1::Multiply>(x, sig_alpha_x);

    // TODO: Check if output tensor type needs to be constrained

    return {result};
} // func end

ONNX_OP("QuickGelu", OPSET_SINCE(1), com_microsoft::opset_1::quickgelu, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov