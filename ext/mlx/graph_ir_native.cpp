#include "graph_ir_native.hpp"

#include <ruby.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

using OrderedJson = nlohmann::ordered_json;

namespace {

using Shape = std::vector<int64_t>;
using ShapeMap = std::unordered_map<std::string, Shape>;
using DtypeMap = std::unordered_map<std::string, std::string>;
using NameSet = std::unordered_set<std::string>;

static VALUE mGraphIR;
static VALUE mGraphIRNative;

constexpr int64_t kGraphIrVersion = 1;

static const std::unordered_map<std::string, std::string> kOnnxOpMap = {
    {"Add", "Add"},
    {"AddMM", "Gemm"},
    {"Subtract", "Sub"},
    {"Multiply", "Mul"},
    {"Square", "Mul"},
    {"Divide", "Div"},
    {"AsType", "Cast"},
    {"Exp", "Exp"},
    {"Log", "Log"},
    {"Sin", "Sin"},
    {"Cos", "Cos"},
    {"Erf", "Erf"},
    {"Sqrt", "Sqrt"},
    {"Abs", "Abs"},
    {"Floor", "Floor"},
    {"Negative", "Neg"},
    {"Relu", "Relu"},
    {"Sigmoid", "Sigmoid"},
    {"Tanh", "Tanh"},
    {"Softmax", "Softmax"},
    {"Greater", "Greater"},
    {"Less", "Less"},
    {"Equal", "Equal"},
    {"Select", "Where"},
    {"Full", "Identity"},
    {"Matmul", "MatMul"},
    {"Reshape", "Reshape"},
    {"Flatten", "Reshape"},
    {"Unflatten", "Reshape"},
    {"Transpose", "Transpose"},
    {"Squeeze", "Squeeze"},
    {"ExpandDims", "Unsqueeze"},
    {"Broadcast", "Expand"},
    {"Arange", "Constant"},
    {"AsStrided", "Gather"},
    {"Concatenate", "Concat"},
    {"Convolution", "Conv"},
    {"ConvolutionTranspose", "ConvTranspose"},
    {"Gather", "Gather"},
    {"GatherAxis", "GatherElements"},
    {"Slice", "Slice"},
    {"Split", "Split"},
    {"LogSumExp", "ReduceLogSumExp"},
    {"Pad", "Pad"},
    {"Scan", "CumSum"},
    {"ScatterAxis", "ScatterElements"},
    {"Maximum", "Max"},
    {"Minimum", "Min"},
    {"Power", "Pow"},
};

static const std::unordered_map<int64_t, std::string> kReduceCodeToOnnxOp = {
    {0, "ReduceMin"},
    {1, "ReduceMax"},
    {2, "ReduceSum"},
    {3, "ReduceProd"},
    {4, "ReduceMin"},
    {5, "ReduceMax"},
};

static const std::unordered_map<int64_t, std::string> kArgReduceCodeToOnnxOp = {
    {0, "ArgMin"},
    {1, "ArgMax"},
};

static const std::unordered_map<std::string, std::string> kOnnxDtypeMap = {
    {"bool", "BOOL"},
    {"bool_", "BOOL"},
    {"uint8", "UINT8"},
    {"uint16", "UINT16"},
    {"uint32", "UINT32"},
    {"uint64", "UINT64"},
    {"int8", "INT8"},
    {"int16", "INT16"},
    {"int32", "INT32"},
    {"int64", "INT64"},
    {"float16", "FLOAT16"},
    {"float32", "FLOAT"},
    {"float64", "DOUBLE"},
    {"bfloat16", "BFLOAT16"},
    {"complex64", "COMPLEX64"},
};

static const std::unordered_map<std::string, int> kDtypePromotionRank = {
    {"bool", 0},
    {"uint8", 1},
    {"int8", 2},
    {"uint16", 3},
    {"int16", 4},
    {"uint32", 5},
    {"int32", 6},
    {"uint64", 7},
    {"int64", 8},
    {"bfloat16", 9},
    {"float16", 10},
    {"float32", 11},
    {"float64", 12},
};

static std::string std_string_from_ruby(VALUE value) {
  VALUE str = rb_obj_as_string(value);
  return std::string(RSTRING_PTR(str), static_cast<size_t>(RSTRING_LEN(str)));
}

static VALUE ruby_string_from_std(const std::string& value) {
  return rb_str_new(value.data(), static_cast<long>(value.size()));
}

static bool json_is_numeric(const OrderedJson& value) {
  return value.is_number_integer() || value.is_number_unsigned() || value.is_number_float();
}

static int64_t normalized_integer_scalar(const OrderedJson& value, const std::string& label) {
  constexpr int64_t int64_min = std::numeric_limits<int64_t>::min();
  constexpr int64_t int64_max = std::numeric_limits<int64_t>::max();

  if (value.is_number_integer()) {
    return value.get<int64_t>();
  }
  if (value.is_number_unsigned()) {
    const auto raw = value.get<uint64_t>();
    if (raw <= static_cast<uint64_t>(int64_max)) {
      return static_cast<int64_t>(raw);
    }

    constexpr uint64_t uint64_max = std::numeric_limits<uint64_t>::max();
    constexpr unsigned __int128 uint64_modulus = static_cast<unsigned __int128>(uint64_max) + 1;
    const unsigned __int128 raw128 = static_cast<unsigned __int128>(raw);
    const __int128 wrapped = static_cast<__int128>(raw128) - static_cast<__int128>(uint64_modulus);
    if (wrapped >= static_cast<__int128>(int64_min) && wrapped <= static_cast<__int128>(int64_max)) {
      return static_cast<int64_t>(wrapped);
    }

    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] " << label
        << " is outside supported signed 64-bit range";
    throw std::range_error(out.str());
  }
  if (value.is_number_float()) {
    const double raw = value.get<double>();
    if (!std::isfinite(raw) || std::trunc(raw) != raw) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] " << label << " must be an Integer";
      throw std::invalid_argument(out.str());
    }
    if (raw < static_cast<double>(int64_min) || raw > static_cast<double>(int64_max)) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] " << label
          << " is outside supported signed 64-bit range";
      throw std::range_error(out.str());
    }
    return static_cast<int64_t>(raw);
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] " << label << " must be an Integer";
  throw std::invalid_argument(out.str());
}

static std::vector<int64_t> normalize_integer_vector(const OrderedJson& value, const std::string& label) {
  if (value.is_array()) {
    std::vector<int64_t> out;
    out.reserve(value.size());
    for (const auto& item : value) {
      out.push_back(normalized_integer_scalar(item, label));
    }
    return out;
  }

  if (json_is_numeric(value)) {
    return {normalized_integer_scalar(value, label)};
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] " << label
      << " must be an Integer or Array of Integer";
  throw std::invalid_argument(out.str());
}

static std::vector<std::string> json_string_vector(const OrderedJson& value, const std::string& label) {
  if (!value.is_array()) {
    std::ostringstream out;
    out << label << " must be an Array";
    throw std::invalid_argument(out.str());
  }

  std::vector<std::string> out;
  out.reserve(value.size());
  for (const auto& item : value) {
    if (!item.is_string()) {
      std::ostringstream msg;
      msg << label << " must contain String values";
      throw std::invalid_argument(msg.str());
    }
    out.push_back(item.get<std::string>());
  }
  return out;
}

static OrderedJson json_from_shape(const Shape& shape) {
  OrderedJson out = OrderedJson::array();
  for (const auto dim : shape) {
    out.push_back(dim);
  }
  return out;
}

static OrderedJson json_from_string_vector(const std::vector<std::string>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto& value : values) {
    out.push_back(value);
  }
  return out;
}

static OrderedJson json_from_int_vector(const std::vector<int64_t>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto value : values) {
    out.push_back(value);
  }
  return out;
}

static std::optional<Shape> known_shape_for(const ShapeMap& known_shapes, const std::string& name) {
  const auto it = known_shapes.find(name);
  if (it == known_shapes.end()) {
    return std::nullopt;
  }
  return it->second;
}

static std::optional<std::string> known_dtype_for(const DtypeMap& known_dtypes, const std::string& name) {
  const auto it = known_dtypes.find(name);
  if (it == known_dtypes.end()) {
    return std::nullopt;
  }
  return it->second;
}

static std::string canonical_dtype(const std::string& dtype) {
  return dtype == "bool_" ? "bool" : dtype;
}

static std::optional<std::string> canonical_dtype(const std::optional<std::string>& dtype) {
  if (!dtype.has_value()) {
    return std::nullopt;
  }
  return canonical_dtype(dtype.value());
}

static std::optional<std::string> onnx_effective_dtype(const std::optional<std::string>& dtype) {
  if (!dtype.has_value()) {
    return std::nullopt;
  }
  const auto canonical = canonical_dtype(dtype.value());
  return canonical == "bfloat16" ? std::optional<std::string>("float32") : std::optional<std::string>(canonical);
}

static std::string onnx_effective_dtype(const std::string& dtype) {
  const auto canonical = canonical_dtype(dtype);
  return canonical == "bfloat16" ? "float32" : canonical;
}

static const std::string& onnx_dtype_symbol(const std::string& dtype) {
  const auto it = kOnnxDtypeMap.find(dtype);
  if (it != kOnnxDtypeMap.end()) {
    return it->second;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported dtype " << dtype;
  throw std::runtime_error(out.str());
}

static const std::string& onnx_op_name(const std::string& op) {
  const auto it = kOnnxOpMap.find(op);
  if (it != kOnnxOpMap.end()) {
    return it->second;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported op " << op;
  throw std::runtime_error(out.str());
}

static int64_t normalize_positive_integer(VALUE value, const char* label) {
  VALUE integer = rb_Integer(value);
  const int64_t out = NUM2LL(integer);
  if (out <= 0) {
    std::ostringstream msg;
    msg << label << " must be a positive Integer";
    throw std::invalid_argument(msg.str());
  }
  return out;
}

static std::string non_empty_model_name(VALUE value) {
  std::string out = std_string_from_ruby(value);
  if (out.empty()) {
    throw std::invalid_argument("model_name must not be empty");
  }
  return out;
}

static int64_t normalize_axis(int64_t axis, size_t rank, const std::string& label) {
  int64_t index = axis;
  if (index < 0) {
    index += static_cast<int64_t>(rank);
  }
  if (index < 0 || index >= static_cast<int64_t>(rank)) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] " << label << " " << axis
        << " is out of bounds for rank " << rank;
    throw std::invalid_argument(out.str());
  }
  return index;
}

static std::string unique_aux_tensor_name(NameSet& used_tensor_names, size_t node_index, const std::string& label) {
  const std::string base = "__mlxir_aux_node" + std::to_string(node_index) + "_" + label;
  std::string candidate = base;
  size_t suffix = 0;
  while (used_tensor_names.find(candidate) != used_tensor_names.end()) {
    ++suffix;
    candidate = base + "_" + std::to_string(suffix);
  }
  used_tensor_names.insert(candidate);
  return candidate;
}

static OrderedJson build_onnx_node_spec(
    const std::string& name,
    const std::string& op_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    OrderedJson attributes) {
  OrderedJson out = OrderedJson::object();
  out["name"] = name;
  out["op_type"] = op_type;
  out["inputs"] = json_from_string_vector(inputs);
  out["outputs"] = json_from_string_vector(outputs);
  out["attributes"] = std::move(attributes);
  return out;
}

static OrderedJson normalize_initializer_int64_values(const OrderedJson& value, const std::string& label) {
  if (value.is_array()) {
    OrderedJson out = OrderedJson::array();
    for (const auto& item : value) {
      out.push_back(normalize_initializer_int64_values(item, label));
    }
    return out;
  }
  return OrderedJson(normalized_integer_scalar(value, label));
}

static OrderedJson onnx_value_info(const OrderedJson& tensor) {
  OrderedJson out = OrderedJson::object();
  const auto dtype = onnx_effective_dtype(tensor.at("dtype").get<std::string>());
  out["name"] = tensor.at("name").get<std::string>();
  out["shape"] = tensor.at("shape");
  out["dtype"] = dtype;
  out["onnx_elem_type"] = onnx_dtype_symbol(dtype);
  return out;
}

static OrderedJson onnx_initializer_info(const OrderedJson& tensor) {
  OrderedJson info = onnx_value_info(tensor);
  OrderedJson values = tensor.at("values");
  if (info.at("dtype").get<std::string>() == "int64") {
    const std::string label = "initializer " + info.at("name").get<std::string>();
    values = normalize_initializer_int64_values(values, label);
  }
  info["values"] = std::move(values);
  return info;
}

static NameSet collect_payload_tensor_names(const OrderedJson& payload) {
  NameSet names;

  for (const auto& tensor : payload.at("inputs")) {
    names.insert(tensor.at("name").get<std::string>());
  }
  for (const auto& tensor : payload.at("constants")) {
    names.insert(tensor.at("name").get<std::string>());
  }
  for (const auto& tensor : payload.at("outputs")) {
    names.insert(tensor.at("name").get<std::string>());
  }
  for (const auto& node : payload.at("nodes")) {
    for (const auto& name : node.at("inputs")) {
      names.insert(name.get<std::string>());
    }
    for (const auto& name : node.at("outputs")) {
      names.insert(name.get<std::string>());
    }
  }

  return names;
}

static ShapeMap collect_known_tensor_shapes(const OrderedJson& payload) {
  ShapeMap out;

  for (const auto& tensor : payload.at("inputs")) {
    out[tensor.at("name").get<std::string>()] = normalize_integer_vector(tensor.at("shape"), "tensor shape");
  }
  for (const auto& tensor : payload.at("constants")) {
    out[tensor.at("name").get<std::string>()] = normalize_integer_vector(tensor.at("shape"), "tensor shape");
  }
  for (const auto& tensor : payload.at("outputs")) {
    out[tensor.at("name").get<std::string>()] = normalize_integer_vector(tensor.at("shape"), "tensor shape");
  }

  return out;
}

static DtypeMap collect_known_tensor_dtypes(const OrderedJson& payload) {
  DtypeMap out;

  for (const auto& tensor : payload.at("inputs")) {
    out[tensor.at("name").get<std::string>()] = onnx_effective_dtype(tensor.at("dtype").get<std::string>());
  }
  for (const auto& tensor : payload.at("constants")) {
    out[tensor.at("name").get<std::string>()] = onnx_effective_dtype(tensor.at("dtype").get<std::string>());
  }
  for (const auto& tensor : payload.at("outputs")) {
    out[tensor.at("name").get<std::string>()] = onnx_effective_dtype(tensor.at("dtype").get<std::string>());
  }

  return out;
}

struct ConvolutionAttributes {
  std::vector<int64_t> strides;
  std::vector<int64_t> padding_low;
  std::vector<int64_t> padding_high;
  std::vector<int64_t> pads;
  std::vector<int64_t> kernel_dilation;
  std::vector<int64_t> input_dilation;
  int64_t groups;
  bool flip;
  size_t spatial_rank;
};

struct ConvTransposeAttributes {
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;
  std::vector<int64_t> pads_begin;
  std::vector<int64_t> pads_end;
  std::vector<int64_t> pads;
  std::vector<int64_t> output_padding;
};

struct ScanArguments {
  int64_t reduce_type;
  int64_t axis;
  bool reverse;
  bool inclusive;
};

struct AsStridedArguments {
  Shape output_shape;
  std::vector<int64_t> strides;
  int64_t offset;
};

struct ArangeArguments {
  bool integral;
  int64_t start_i;
  int64_t stop_i;
  int64_t step_i;
  double start_f;
  double stop_f;
  double step_f;
  std::string dtype;
};

static std::optional<std::vector<int64_t>> transpose_perm_from_arguments(const OrderedJson& arguments) {
  if (!arguments.is_array()) {
    return std::nullopt;
  }

  for (const auto& value : arguments) {
    if (!value.is_array()) {
      continue;
    }

    try {
      return normalize_integer_vector(value, "Transpose permutation");
    } catch (const std::exception&) {
      // Try next argument.
    }
  }

  return std::nullopt;
}

static std::optional<int64_t> concatenate_axis_from_arguments(const OrderedJson& arguments, bool strict) {
  if (arguments.is_array() && arguments.size() == 1) {
    try {
      return normalized_integer_scalar(arguments.at(0), "Concatenate axis");
    } catch (const std::exception&) {
      // Handled below.
    }
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported Concatenate arguments " << arguments.dump()
      << "; expected [axis]";
  throw std::runtime_error(out.str());
}

static std::optional<int64_t> gather_axis_from_arguments(const OrderedJson& arguments, bool strict) {
  if (arguments.is_array() && !arguments.empty()) {
    const auto& first = arguments.at(0);
    try {
      return normalized_integer_scalar(first, "Gather axis");
    } catch (const std::exception&) {
      // Try vector-encoded axis.
    }

    if (first.is_array() && first.size() == 1) {
      try {
        return normalized_integer_scalar(first.at(0), "Gather axis");
      } catch (const std::exception&) {
        // Handled below.
      }
    }
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported Gather arguments " << arguments.dump()
      << "; expected first argument to encode axis";
  throw std::runtime_error(out.str());
}

static std::optional<OrderedJson> scatter_axis_attributes_from_arguments(const OrderedJson& arguments, bool strict) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported ScatterAxis arguments " << arguments.dump()
        << "; expected [mode, axis]";
    throw std::runtime_error(out.str());
  }

  int64_t mode = 0;
  int64_t axis = 0;
  try {
    mode = normalized_integer_scalar(arguments.at(0), "ScatterAxis mode");
    axis = normalized_integer_scalar(arguments.at(1), "ScatterAxis axis");
  } catch (const std::exception&) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported ScatterAxis arguments " << arguments.dump()
        << "; mode/axis must be Integer";
    throw std::runtime_error(out.str());
  }

  if (mode != 1) {
    if (!strict) {
      return std::nullopt;
    }

    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported ScatterAxis mode " << mode
        << "; only update mode (1) is supported";
    throw std::runtime_error(out.str());
  }

  OrderedJson attributes = OrderedJson::object();
  attributes["axis"] = axis;
  return attributes;
}

static OrderedJson onnx_node_attributes(const OrderedJson& node) {
  const auto op = node.at("op").get<std::string>();
  const OrderedJson arguments = node.contains("arguments") ? node.at("arguments") : OrderedJson::array();

  if (op == "Transpose") {
    const auto perm = transpose_perm_from_arguments(arguments);
    if (!perm.has_value() || perm->empty()) {
      return OrderedJson::object();
    }
    OrderedJson out = OrderedJson::object();
    out["perm"] = json_from_int_vector(*perm);
    return out;
  }

  if (op == "Concatenate") {
    OrderedJson out = OrderedJson::object();
    out["axis"] = concatenate_axis_from_arguments(arguments, true).value();
    return out;
  }

  if (op == "Gather" || op == "GatherAxis") {
    OrderedJson out = OrderedJson::object();
    out["axis"] = gather_axis_from_arguments(arguments, true).value();
    return out;
  }

  if (op == "ScatterAxis") {
    return scatter_axis_attributes_from_arguments(arguments, true).value();
  }

  return OrderedJson::object();
}

static std::optional<ConvolutionAttributes> convolution_attributes_from_arguments(
    const OrderedJson& arguments,
    bool strict) {
  if (!(arguments.is_array() && arguments.size() >= 7)) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Convolution arguments " << arguments.dump()
        << "; expected [strides, padding_low, padding_high, kernel_dilation, input_dilation, groups, flip]";
    throw std::runtime_error(out.str());
  }

  ConvolutionAttributes out;
  out.strides = normalize_integer_vector(arguments.at(0), "Convolution strides");
  out.padding_low = normalize_integer_vector(arguments.at(1), "Convolution padding_low");
  out.padding_high = normalize_integer_vector(arguments.at(2), "Convolution padding_high");
  out.kernel_dilation = normalize_integer_vector(arguments.at(3), "Convolution kernel_dilation");
  out.input_dilation = normalize_integer_vector(arguments.at(4), "Convolution input_dilation");
  out.groups = normalized_integer_scalar(arguments.at(5), "Convolution groups");
  if (!arguments.at(6).is_boolean()) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Convolution flip must be boolean");
  }
  out.flip = arguments.at(6).get<bool>();
  out.spatial_rank = out.strides.size();

  const std::vector<size_t> lengths = {
      out.padding_low.size(),
      out.padding_high.size(),
      out.kernel_dilation.size(),
      out.input_dilation.size()};
  for (size_t length : lengths) {
    if (length != out.spatial_rank) {
      std::ostringstream msg;
      msg << "[graph_ir_to_onnx_stub] Convolution argument lengths must match spatial rank "
          << out.spatial_rank;
      throw std::invalid_argument(msg.str());
    }
  }

  if (std::any_of(out.strides.begin(), out.strides.end(), [](int64_t value) { return value <= 0; })) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Convolution strides must be positive");
  }
  if (std::any_of(out.padding_low.begin(), out.padding_low.end(), [](int64_t value) { return value < 0; }) ||
      std::any_of(out.padding_high.begin(), out.padding_high.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error("[graph_ir_to_onnx_stub] unsupported Convolution with negative padding");
  }
  if (std::any_of(out.kernel_dilation.begin(), out.kernel_dilation.end(), [](int64_t value) { return value <= 0; })) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Convolution kernel_dilation must be positive");
  }
  if (std::any_of(out.input_dilation.begin(), out.input_dilation.end(), [](int64_t value) { return value <= 0; })) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Convolution input_dilation must be positive");
  }
  if (out.groups <= 0) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Convolution groups must be a positive Integer");
  }

  out.pads = out.padding_low;
  out.pads.insert(out.pads.end(), out.padding_high.begin(), out.padding_high.end());
  return out;
}

static ConvTransposeAttributes convtranspose_attributes_from_convolution(
    const ConvolutionAttributes& convolution,
    const Shape& weight_shape) {
  const auto weight = normalize_integer_vector(json_from_int_vector(weight_shape), "ConvolutionTranspose weight shape");
  const size_t spatial_rank = convolution.spatial_rank;
  const size_t expected_rank = spatial_rank + 2;
  if (weight.size() != expected_rank) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] ConvolutionTranspose weight rank mismatch: expected " << expected_rank
        << ", got " << weight.size();
    throw std::invalid_argument(out.str());
  }

  std::vector<int64_t> kernel_shape(weight.begin() + 1, weight.end() - 1);
  const auto& kernel_dilation = convolution.kernel_dilation;
  const auto& padding_low = convolution.padding_low;
  const auto& padding_high = convolution.padding_high;
  const auto& strides = convolution.input_dilation;

  std::vector<int64_t> base_padding;
  base_padding.reserve(spatial_rank);
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    base_padding.push_back(kernel_dilation[axis] * (kernel_shape[axis] - 1));
  }

  ConvTransposeAttributes out;
  out.strides = strides;
  out.dilations = kernel_dilation;
  out.pads_begin.reserve(spatial_rank);
  out.pads_end.reserve(spatial_rank);
  out.output_padding.reserve(spatial_rank);

  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    out.pads_begin.push_back(base_padding[axis] - padding_low[axis]);
    out.output_padding.push_back(padding_high[axis] - padding_low[axis]);
    out.pads_end.push_back(base_padding[axis] - padding_high[axis] + out.output_padding[axis]);
  }

  if (std::any_of(out.pads_begin.begin(), out.pads_begin.end(), [](int64_t value) { return value < 0; }) ||
      std::any_of(out.pads_end.begin(), out.pads_end.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error(
        "[graph_ir_to_onnx_stub] unsupported ConvolutionTranspose derived negative padding from arguments");
  }
  if (std::any_of(out.output_padding.begin(), out.output_padding.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error(
        "[graph_ir_to_onnx_stub] unsupported ConvolutionTranspose with negative output_padding");
  }

  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    if (out.output_padding[axis] >= out.strides[axis]) {
      std::ostringstream msg;
      msg << "[graph_ir_to_onnx_stub] unsupported ConvolutionTranspose output_padding "
          << json_from_int_vector(out.output_padding).dump() << "; each value must be < corresponding stride "
          << json_from_int_vector(out.strides).dump();
      throw std::runtime_error(msg.str());
    }
  }

  out.pads = out.pads_begin;
  out.pads.insert(out.pads.end(), out.pads_end.begin(), out.pads_end.end());
  return out;
}

static std::optional<std::string> reduce_onnx_op_type(const OrderedJson& arguments, bool strict) {
  int64_t reduce_code = 0;
  if (arguments.is_array() && !arguments.empty()) {
    reduce_code = normalized_integer_scalar(arguments.at(0), "Reduce code");
  } else if (!strict) {
    return std::nullopt;
  }

  const auto it = kReduceCodeToOnnxOp.find(reduce_code);
  if (it != kReduceCodeToOnnxOp.end()) {
    return it->second;
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported Reduce code " << reduce_code;
  throw std::runtime_error(out.str());
}

static std::optional<std::string> argreduce_onnx_op_type(const OrderedJson& arguments, bool strict) {
  int64_t reduce_code = 0;
  if (arguments.is_array() && !arguments.empty()) {
    reduce_code = normalized_integer_scalar(arguments.at(0), "ArgReduce code");
  } else if (!strict) {
    return std::nullopt;
  }

  const auto it = kArgReduceCodeToOnnxOp.find(reduce_code);
  if (it != kArgReduceCodeToOnnxOp.end()) {
    return it->second;
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported ArgReduce code " << reduce_code;
  throw std::runtime_error(out.str());
}

static std::optional<Shape> infer_elementwise_output_shape(const std::optional<Shape>& lhs_shape, const std::optional<Shape>& rhs_shape) {
  if (!lhs_shape.has_value() || !rhs_shape.has_value()) {
    return std::nullopt;
  }

  const auto& lhs = lhs_shape.value();
  const auto& rhs = rhs_shape.value();
  const size_t max_rank = std::max(lhs.size(), rhs.size());

  Shape lhs_aligned(max_rank - lhs.size(), 1);
  lhs_aligned.insert(lhs_aligned.end(), lhs.begin(), lhs.end());

  Shape rhs_aligned(max_rank - rhs.size(), 1);
  rhs_aligned.insert(rhs_aligned.end(), rhs.begin(), rhs.end());

  Shape out;
  out.reserve(max_rank);

  for (size_t i = 0; i < max_rank; ++i) {
    const auto left = lhs_aligned[i];
    const auto right = rhs_aligned[i];
    if (left == right) {
      out.push_back(left);
    } else if (left == 1) {
      out.push_back(right);
    } else if (right == 1) {
      out.push_back(left);
    } else {
      return std::nullopt;
    }
  }

  return out;
}

static std::optional<Shape> infer_matmul_output_shape(const std::optional<Shape>& lhs_shape, const std::optional<Shape>& rhs_shape) {
  if (!lhs_shape.has_value() || !rhs_shape.has_value()) {
    return std::nullopt;
  }

  const Shape& lhs = lhs_shape.value();
  const Shape& rhs = rhs_shape.value();
  if (lhs.empty() || rhs.empty()) {
    return std::nullopt;
  }

  const bool lhs_was_1d = lhs.size() == 1;
  const bool rhs_was_1d = rhs.size() == 1;

  Shape lhs_matrix = lhs_was_1d ? Shape{1, lhs[0]} : Shape{lhs[lhs.size() - 2], lhs[lhs.size() - 1]};
  Shape rhs_matrix = rhs_was_1d ? Shape{rhs[0], 1} : Shape{rhs[rhs.size() - 2], rhs[rhs.size() - 1]};

  if (lhs_matrix[1] != rhs_matrix[0]) {
    return std::nullopt;
  }

  Shape lhs_batch = lhs_was_1d ? Shape{} : Shape(lhs.begin(), lhs.end() - 2);
  Shape rhs_batch = rhs_was_1d ? Shape{} : Shape(rhs.begin(), rhs.end() - 2);
  const auto batch = infer_elementwise_output_shape(lhs_batch, rhs_batch);
  if (!batch.has_value()) {
    return std::nullopt;
  }

  Shape out = batch.value();
  out.push_back(lhs_matrix[0]);
  out.push_back(rhs_matrix[1]);

  if (lhs_was_1d) {
    out.erase(out.begin() + static_cast<long>(batch->size()));
  }
  if (rhs_was_1d) {
    out.pop_back();
  }

  return out;
}

static std::optional<std::string> promote_binary_dtype(
    const std::optional<std::string>& lhs_dtype,
    const std::optional<std::string>& rhs_dtype) {
  const auto lhs = canonical_dtype(lhs_dtype);
  const auto rhs = canonical_dtype(rhs_dtype);

  if (!lhs.has_value()) {
    return rhs;
  }
  if (!rhs.has_value()) {
    return lhs;
  }
  if (lhs.value() == rhs.value()) {
    return lhs;
  }

  const auto lhs_rank_it = kDtypePromotionRank.find(lhs.value());
  const auto rhs_rank_it = kDtypePromotionRank.find(rhs.value());
  if (lhs_rank_it == kDtypePromotionRank.end() || rhs_rank_it == kDtypePromotionRank.end()) {
    return lhs;
  }

  return lhs_rank_it->second >= rhs_rank_it->second ? lhs : rhs;
}

static int64_t normalize_slice_index(int64_t value, int64_t dim) {
  int64_t index = value;
  if (index < 0) {
    index += dim;
  }
  if (index < 0) {
    index = 0;
  }
  if (index > dim) {
    index = dim;
  }
  return index;
}

static std::vector<int64_t> reduce_axes_from_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Reduce arguments must include reduction code and axes");
  }
  return normalize_integer_vector(arguments.at(1), "Reduce axes");
}

static std::optional<Shape> infer_reduce_keepdims_shape(const std::optional<Shape>& input_shape, const std::vector<int64_t>& axes) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape shape = input_shape.value();
  const size_t rank = shape.size();

  std::set<int64_t> normalized_axes;
  for (const auto axis : axes) {
    normalized_axes.insert(normalize_axis(axis, rank, "Reduce axis"));
  }

  for (const auto axis : normalized_axes) {
    shape[static_cast<size_t>(axis)] = 1;
  }

  return shape;
}

static std::string as_type_target_dtype(
    const OrderedJson& arguments,
    const std::vector<std::string>& outputs,
    const DtypeMap& known_dtypes) {
  if (arguments.is_array() && !arguments.empty()) {
    const auto& target = arguments.at(0);
    if (!(target.is_string() && kOnnxDtypeMap.find(target.get<std::string>()) != kOnnxDtypeMap.end())) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported AsType arguments " << arguments.dump()
          << "; expected first argument to be dtype String";
      throw std::runtime_error(out.str());
    }
    return target.get<std::string>();
  }

  std::set<std::string> candidates;
  for (const auto& name : outputs) {
    const auto it = known_dtypes.find(name);
    if (it != known_dtypes.end()) {
      candidates.insert(it->second);
    }
  }

  if (candidates.size() == 1) {
    return *candidates.begin();
  }
  if (candidates.size() > 1) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported AsType with inconsistent output dtypes";
    throw std::runtime_error(out.str());
  }

  throw std::runtime_error("[graph_ir_to_onnx_stub] unsupported AsType without target dtype argument");
}

static bool equal_nan_from_arguments(const OrderedJson& arguments) {
  if (!arguments.is_array() || arguments.empty()) {
    return false;
  }
  if (!(arguments.size() == 1 && arguments.at(0).is_boolean())) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Equal arguments " << arguments.dump()
        << "; expected [equal_nan]";
    throw std::runtime_error(out.str());
  }
  return arguments.at(0).get<bool>();
}

static std::vector<int64_t> infer_logsumexp_axes(const Shape& input_shape, const std::optional<Shape>& output_shape) {
  const Shape input = input_shape;
  if (output_shape.has_value()) {
    const Shape output = output_shape.value();

    if (output.size() == input.size()) {
      std::vector<int64_t> axes;
      for (size_t i = 0; i < input.size(); ++i) {
        const auto dim = input[i];
        const auto out_dim = output[i];
        if (out_dim == 1 && dim != 1) {
          axes.push_back(static_cast<int64_t>(i));
        } else if (out_dim != dim) {
          std::ostringstream out;
          out << "[graph_ir_to_onnx_stub] unsupported LogSumExp output shape "
              << json_from_shape(output).dump() << " for input " << json_from_shape(input).dump();
          throw std::runtime_error(out.str());
        }
      }
      if (axes.empty()) {
        return {static_cast<int64_t>(input.size() - 1)};
      }
      return axes;
    }

    if (output.size() == input.size() - 1) {
      return {static_cast<int64_t>(input.size() - 1)};
    }
  }

  return {static_cast<int64_t>(input.size() - 1)};
}

static std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> pad_axes_and_sizes_from_arguments(
    const OrderedJson& arguments,
    const Shape& input_shape) {
  if (!(arguments.is_array() && arguments.size() >= 3)) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Pad arguments " << arguments.dump()
        << "; expected [axes, low, high]";
    throw std::runtime_error(out.str());
  }

  auto axes = normalize_integer_vector(arguments.at(0), "Pad axes");
  auto low = normalize_integer_vector(arguments.at(1), "Pad low");
  auto high = normalize_integer_vector(arguments.at(2), "Pad high");

  if (!(axes.size() == low.size() && low.size() == high.size())) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] Pad axes/low/high lengths must match: "
        << axes.size() << "/" << low.size() << "/" << high.size();
    throw std::invalid_argument(out.str());
  }

  if (std::any_of(low.begin(), low.end(), [](int64_t value) { return value < 0; }) ||
      std::any_of(high.begin(), high.end(), [](int64_t value) { return value < 0; })) {
    throw std::runtime_error("[graph_ir_to_onnx_stub] unsupported Pad with negative padding");
  }

  const size_t rank = input_shape.size();
  std::vector<int64_t> normalized_axes;
  normalized_axes.reserve(axes.size());
  for (const auto axis : axes) {
    normalized_axes.push_back(normalize_axis(axis, rank, "Pad axis"));
  }

  std::set<int64_t> uniq(normalized_axes.begin(), normalized_axes.end());
  if (uniq.size() != normalized_axes.size()) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Pad axes must not contain duplicates");
  }

  return {normalized_axes, low, high};
}

static std::optional<Shape> infer_pad_output_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& pads_begin,
    const std::vector<int64_t>& pads_end) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  const Shape shape = input_shape.value();
  if (!(pads_begin.size() == shape.size() && pads_end.size() == shape.size())) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] Pad low/high ranks must match input rank " << shape.size();
    throw std::invalid_argument(out.str());
  }

  Shape out;
  out.reserve(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    out.push_back(shape[i] + pads_begin[i] + pads_end[i]);
  }
  return out;
}

static ScanArguments scan_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 4)) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Scan arguments " << arguments.dump()
        << "; expected [reduce_type, axis, reverse, inclusive]";
    throw std::runtime_error(out.str());
  }

  const auto reduce_type = normalized_integer_scalar(arguments.at(0), "Scan reduce_type");
  const auto axis = normalized_integer_scalar(arguments.at(1), "Scan axis");
  if (!arguments.at(2).is_boolean()) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Scan reverse must be boolean");
  }
  if (!arguments.at(3).is_boolean()) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Scan inclusive must be boolean");
  }

  return {reduce_type, axis, arguments.at(2).get<bool>(), arguments.at(3).get<bool>()};
}

static std::pair<int64_t, int64_t> argreduce_mode_axis(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] ArgReduce arguments must include [mode, axis]");
  }

  const int64_t mode = normalized_integer_scalar(arguments.at(0), "ArgReduce mode");
  const int64_t axis = normalized_integer_scalar(arguments.at(1), "ArgReduce axis");
  if (kArgReduceCodeToOnnxOp.find(mode) == kArgReduceCodeToOnnxOp.end()) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported ArgReduce code " << mode;
    throw std::runtime_error(out.str());
  }

  return {mode, axis};
}

static std::optional<Shape> infer_argreduce_keepdims_shape(const std::optional<Shape>& input_shape, int64_t axis) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape shape = input_shape.value();
  const auto axis_index = normalize_axis(axis, shape.size(), "ArgReduce axis");
  shape[static_cast<size_t>(axis_index)] = 1;
  return shape;
}

static std::optional<Shape> infer_convolution_output_shape(
    const std::optional<Shape>& input_shape,
    const std::optional<Shape>& weight_shape,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& padding_low,
    const std::vector<int64_t>& padding_high,
    const std::vector<int64_t>& kernel_dilation,
    int64_t groups) {
  if (!input_shape.has_value() || !weight_shape.has_value()) {
    return std::nullopt;
  }

  const Shape input = input_shape.value();
  const Shape weight = weight_shape.value();
  const size_t spatial_rank = strides.size();
  const size_t expected_rank = spatial_rank + 2;

  if (!(input.size() == expected_rank && weight.size() == expected_rank)) {
    return std::nullopt;
  }
  if (!(padding_low.size() == spatial_rank && padding_high.size() == spatial_rank && kernel_dilation.size() == spatial_rank)) {
    return std::nullopt;
  }

  const int64_t batch = input[0];
  const int64_t input_channels = input.back();
  const int64_t output_channels = weight[0];
  const int64_t weight_input_channels = weight.back();
  if (input_channels != weight_input_channels * groups) {
    return std::nullopt;
  }

  Shape output_spatial;
  output_spatial.reserve(spatial_rank);
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    const int64_t input_dim = input[axis + 1];
    const int64_t kernel_dim = weight[axis + 1];
    const int64_t dilation = kernel_dilation[axis];
    const int64_t stride = strides[axis];
    const int64_t low = padding_low[axis];
    const int64_t high = padding_high[axis];

    if (kernel_dim <= 0) {
      return std::nullopt;
    }

    const int64_t effective_kernel = dilation * (kernel_dim - 1) + 1;
    const int64_t numerator = input_dim + low + high - effective_kernel;
    if (numerator < 0) {
      return std::nullopt;
    }

    output_spatial.push_back((numerator / stride) + 1);
  }

  Shape out;
  out.push_back(batch);
  out.insert(out.end(), output_spatial.begin(), output_spatial.end());
  out.push_back(output_channels);
  return out;
}

static std::optional<Shape> infer_convolution_transpose_output_shape(
    const std::optional<Shape>& input_shape,
    const std::optional<Shape>& weight_shape,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& pads_begin,
    const std::vector<int64_t>& pads_end,
    const std::vector<int64_t>& kernel_dilation,
    const std::vector<int64_t>& output_padding,
    int64_t groups) {
  if (!input_shape.has_value() || !weight_shape.has_value()) {
    return std::nullopt;
  }

  const Shape input = input_shape.value();
  const Shape weight = weight_shape.value();
  const size_t spatial_rank = strides.size();
  const size_t expected_rank = spatial_rank + 2;

  if (!(input.size() == expected_rank && weight.size() == expected_rank)) {
    return std::nullopt;
  }
  if (!(pads_begin.size() == spatial_rank && pads_end.size() == spatial_rank && kernel_dilation.size() == spatial_rank &&
        output_padding.size() == spatial_rank)) {
    return std::nullopt;
  }

  const int64_t batch = input[0];
  const int64_t input_channels = input.back();
  const int64_t output_channels = weight[0];
  const int64_t weight_input_channels = weight.back();
  if (input_channels != weight_input_channels * groups) {
    return std::nullopt;
  }

  Shape output_spatial;
  output_spatial.reserve(spatial_rank);
  for (size_t axis = 0; axis < spatial_rank; ++axis) {
    const int64_t input_dim = input[axis + 1];
    const int64_t kernel_dim = weight[axis + 1];
    const int64_t dilation = kernel_dilation[axis];
    const int64_t stride = strides[axis];
    const int64_t low = pads_begin[axis];
    const int64_t high = pads_end[axis];
    const int64_t out_padding = output_padding[axis];

    if (kernel_dim <= 0) {
      return std::nullopt;
    }

    const int64_t effective_kernel = dilation * (kernel_dim - 1) + 1;
    const int64_t dim = stride * (input_dim - 1) + out_padding + effective_kernel - low - high;
    if (dim < 0) {
      return std::nullopt;
    }

    output_spatial.push_back(dim);
  }

  Shape out;
  out.push_back(batch);
  out.insert(out.end(), output_spatial.begin(), output_spatial.end());
  out.push_back(output_channels);
  return out;
}

static std::optional<Shape> infer_gather_output_shape(
    const std::optional<Shape>& data_shape,
    const std::optional<Shape>& indices_shape,
    int64_t axis) {
  if (!data_shape.has_value() || !indices_shape.has_value()) {
    return std::nullopt;
  }

  Shape out;
  const auto& data = data_shape.value();
  const auto& indices = indices_shape.value();

  out.insert(out.end(), data.begin(), data.begin() + axis);
  out.insert(out.end(), indices.begin(), indices.end());
  out.insert(out.end(), data.begin() + axis + 1, data.end());
  return out;
}

static std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
slice_vectors_from_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Slice arguments must include starts and ends");
  }

  auto starts = normalize_integer_vector(arguments.at(0), "Slice starts");
  auto ends = normalize_integer_vector(arguments.at(1), "Slice ends");
  std::vector<int64_t> steps;
  if (arguments.size() >= 3) {
    steps = normalize_integer_vector(arguments.at(2), "Slice steps");
  } else {
    steps = std::vector<int64_t>(starts.size(), 1);
  }

  if (!(starts.size() == ends.size() && starts.size() == steps.size())) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] Slice starts/ends/steps lengths must match: "
        << starts.size() << "/" << ends.size() << "/" << steps.size();
    throw std::invalid_argument(out.str());
  }
  if (std::any_of(steps.begin(), steps.end(), [](int64_t value) { return value == 0; })) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Slice steps must not contain zero");
  }

  std::vector<int64_t> axes;
  axes.reserve(starts.size());
  for (size_t i = 0; i < starts.size(); ++i) {
    axes.push_back(static_cast<int64_t>(i));
  }

  return {starts, ends, axes, steps};
}

static std::optional<Shape> infer_slice_output_shape(
    const std::optional<Shape>& input_shape,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& axes,
    const std::vector<int64_t>& steps) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape out_shape = input_shape.value();

  for (size_t i = 0; i < axes.size(); ++i) {
    const auto axis_index = normalize_axis(axes[i], out_shape.size(), "Slice axis");
    const auto dim = out_shape[static_cast<size_t>(axis_index)];
    const auto start_v = normalize_slice_index(starts[i], dim);
    const auto end_v = normalize_slice_index(ends[i], dim);
    const auto step_v = steps[i];

    if (step_v <= 0) {
      return std::nullopt;
    }

    out_shape[static_cast<size_t>(axis_index)] =
        end_v <= start_v ? 0 : ((end_v - start_v - 1) / step_v) + 1;
  }

  return out_shape;
}

static std::vector<int64_t> split_lengths_from_indices(const std::vector<int64_t>& indices, int64_t dim) {
  int64_t prev = 0;
  std::vector<int64_t> lengths;
  lengths.reserve(indices.size() + 1);

  for (const auto index : indices) {
    int64_t value = index;
    if (value < 0) {
      value += dim;
    }
    if (value < prev || value > dim) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] Split boundary " << index
          << " is out of range or not non-decreasing for dim " << dim;
      throw std::invalid_argument(out.str());
    }
    lengths.push_back(value - prev);
    prev = value;
  }

  lengths.push_back(dim - prev);
  return lengths;
}

static std::pair<int64_t, std::vector<int64_t>> split_axis_and_lengths(
    const OrderedJson& arguments,
    const std::optional<Shape>& input_shape,
    int64_t output_count) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Split arguments must include split spec and axis");
  }
  if (output_count <= 0) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Split must have at least one output");
  }

  const auto spec = normalize_integer_vector(arguments.at(0), "Split spec");
  const auto axis = normalized_integer_scalar(arguments.at(1), "Split axis");

  if (!input_shape.has_value()) {
    throw std::runtime_error("[graph_ir_to_onnx_stub] unsupported Split without known input shape");
  }

  const Shape data_shape = input_shape.value();
  const auto axis_index = normalize_axis(axis, data_shape.size(), "Split axis");
  const int64_t dim = data_shape[static_cast<size_t>(axis_index)];

  std::vector<int64_t> lengths;
  if (spec.size() == 1 && spec[0] == output_count) {
    const auto parts = spec[0];
    if (parts <= 0) {
      throw std::invalid_argument("[graph_ir_to_onnx_stub] Split parts must be positive");
    }

    const auto quotient = dim / parts;
    const auto remainder = dim % parts;
    if (remainder != 0) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported uneven equal Split: dim " << dim
          << " not divisible by " << parts;
      throw std::runtime_error(out.str());
    }
    lengths.assign(static_cast<size_t>(parts), quotient);
  } else if (spec.size() == static_cast<size_t>(output_count - 1)) {
    lengths = split_lengths_from_indices(spec, dim);
  } else if (spec.size() == static_cast<size_t>(output_count)) {
    lengths = spec;
  } else {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Split spec " << json_from_int_vector(spec).dump()
        << " for " << output_count << " outputs";
    throw std::runtime_error(out.str());
  }

  if (static_cast<int64_t>(lengths.size()) != output_count) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] Split lengths count " << lengths.size()
        << " does not match outputs " << output_count;
    throw std::invalid_argument(out.str());
  }

  if (std::any_of(lengths.begin(), lengths.end(), [](int64_t value) { return value < 0; })) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Split lengths must be non-negative");
  }

  const int64_t sum = std::accumulate(lengths.begin(), lengths.end(), static_cast<int64_t>(0));
  if (sum != dim) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Split lengths " << json_from_int_vector(lengths).dump()
        << "; expected sum " << dim;
    throw std::runtime_error(out.str());
  }

  return {axis_index, lengths};
}

static std::optional<std::vector<Shape>> infer_split_output_shapes(
    const std::optional<Shape>& input_shape,
    int64_t axis,
    const std::vector<int64_t>& lengths) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  const auto shape = input_shape.value();
  std::vector<Shape> out;
  out.reserve(lengths.size());

  for (const auto length : lengths) {
    Shape current = shape;
    current[static_cast<size_t>(axis)] = length;
    out.push_back(std::move(current));
  }

  return out;
}

static std::optional<Shape> infer_concatenate_output_shape(
    const std::vector<std::optional<Shape>>& input_shapes,
    int64_t axis) {
  if (input_shapes.empty()) {
    return std::nullopt;
  }
  if (std::any_of(input_shapes.begin(), input_shapes.end(), [](const auto& item) { return !item.has_value(); })) {
    return std::nullopt;
  }

  const Shape first = input_shapes.front().value();
  const size_t rank = first.size();
  for (const auto& shape : input_shapes) {
    if (shape->size() != rank) {
      return std::nullopt;
    }
  }

  const auto axis_index = normalize_axis(axis, rank, "Concatenate axis");
  Shape out = first;
  out[static_cast<size_t>(axis_index)] = 0;

  for (const auto& shape : input_shapes) {
    for (size_t i = 0; i < rank; ++i) {
      if (static_cast<int64_t>(i) == axis_index) {
        continue;
      }
      if ((*shape)[i] != out[i]) {
        return std::nullopt;
      }
    }
    out[static_cast<size_t>(axis_index)] += (*shape)[static_cast<size_t>(axis_index)];
  }

  return out;
}

static std::optional<Shape> infer_squeeze_output_shape(const std::optional<Shape>& input_shape, const std::vector<int64_t>& axes) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape shape = input_shape.value();
  const size_t rank = shape.size();

  std::set<int64_t> normalized_axes;
  for (const auto axis : axes) {
    normalized_axes.insert(normalize_axis(axis, rank, "Squeeze axis"));
  }

  for (auto it = normalized_axes.rbegin(); it != normalized_axes.rend(); ++it) {
    const auto axis_index = static_cast<size_t>(*it);
    const auto dim = shape[axis_index];
    if (dim != 1) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Squeeze axis " << axis_index
          << " for dim " << dim << "; expected dimension 1";
      throw std::runtime_error(out.str());
    }
    shape.erase(shape.begin() + static_cast<long>(axis_index));
  }

  return shape;
}

static std::optional<Shape> infer_unsqueeze_output_shape(const std::optional<Shape>& input_shape, const std::vector<int64_t>& axes) {
  if (!input_shape.has_value()) {
    return std::nullopt;
  }

  Shape out = input_shape.value();
  const size_t output_rank = out.size() + axes.size();

  std::vector<int64_t> normalized_axes;
  normalized_axes.reserve(axes.size());
  for (const auto axis : axes) {
    int64_t value = axis;
    if (value < 0) {
      value += static_cast<int64_t>(output_rank);
    }
    if (value < 0 || value >= static_cast<int64_t>(output_rank)) {
      std::ostringstream msg;
      msg << "[graph_ir_to_onnx_stub] ExpandDims axis " << axis
          << " is out of bounds for output rank " << output_rank;
      throw std::invalid_argument(msg.str());
    }
    normalized_axes.push_back(value);
  }

  std::set<int64_t> uniq(normalized_axes.begin(), normalized_axes.end());
  if (uniq.size() != normalized_axes.size()) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] ExpandDims axes must not contain duplicates");
  }

  std::sort(normalized_axes.begin(), normalized_axes.end());
  for (const auto axis : normalized_axes) {
    out.insert(out.begin() + static_cast<long>(axis), 1);
  }

  return out;
}

static std::vector<int64_t> gather_reorder_permutation(int64_t data_rank, int64_t indices_rank, int64_t axis) {
  std::vector<int64_t> indices;
  for (int64_t i = axis; i < axis + indices_rank; ++i) {
    indices.push_back(i);
  }

  std::vector<int64_t> prefix;
  for (int64_t i = 0; i < axis; ++i) {
    prefix.push_back(i);
  }

  std::vector<int64_t> suffix;
  const int64_t suffix_start = axis + indices_rank;
  const int64_t suffix_end = data_rank + indices_rank - 2;
  if (suffix_start <= suffix_end) {
    for (int64_t i = suffix_start; i <= suffix_end; ++i) {
      suffix.push_back(i);
    }
  }

  std::vector<int64_t> out;
  out.reserve(indices.size() + prefix.size() + suffix.size());
  out.insert(out.end(), indices.begin(), indices.end());
  out.insert(out.end(), prefix.begin(), prefix.end());
  out.insert(out.end(), suffix.begin(), suffix.end());
  return out;
}

static bool identity_permutation(const std::vector<int64_t>& perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      return false;
    }
  }
  return true;
}

static std::pair<std::vector<int64_t>, std::vector<int64_t>> convolution_data_permutations(size_t spatial_rank) {
  const auto rank = static_cast<int64_t>(spatial_rank + 2);

  std::vector<int64_t> to_onnx = {0, rank - 1};
  for (int64_t i = 1; i < rank - 1; ++i) {
    to_onnx.push_back(i);
  }

  std::vector<int64_t> from_onnx = {0};
  for (int64_t i = 2; i < rank; ++i) {
    from_onnx.push_back(i);
  }
  from_onnx.push_back(1);

  return {to_onnx, from_onnx};
}

static std::vector<int64_t> convolution_weight_permutation(size_t spatial_rank) {
  const auto rank = static_cast<int64_t>(spatial_rank + 2);
  std::vector<int64_t> out = {0, rank - 1};
  for (int64_t i = 1; i < rank - 1; ++i) {
    out.push_back(i);
  }
  return out;
}

static std::vector<int64_t> convolution_transpose_weight_permutation(size_t spatial_rank) {
  const auto rank = static_cast<int64_t>(spatial_rank + 2);
  std::vector<int64_t> out = {rank - 1, 0};
  for (int64_t i = 1; i < rank - 1; ++i) {
    out.push_back(i);
  }
  return out;
}

static Shape permute_shape(const Shape& shape, const std::vector<int64_t>& perm, const std::string& label) {
  if (perm.size() != shape.size()) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] invalid permutation " << json_from_int_vector(perm).dump()
        << " for " << label << " rank " << shape.size();
    throw std::invalid_argument(out.str());
  }

  std::vector<int64_t> sorted = perm;
  std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (sorted[i] != static_cast<int64_t>(i)) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] invalid permutation " << json_from_int_vector(perm).dump()
          << " for " << label << " rank " << shape.size();
      throw std::invalid_argument(out.str());
    }
  }

  Shape out;
  out.reserve(perm.size());
  for (const auto axis : perm) {
    out.push_back(shape[static_cast<size_t>(axis)]);
  }
  return out;
}

static std::vector<int64_t> integer_vector_argument(const OrderedJson& arguments, const std::string& op_name) {
  if (arguments.is_array()) {
    for (const auto& value : arguments) {
      if (!value.is_array()) {
        continue;
      }

      try {
        return normalize_integer_vector(value, op_name + " argument");
      } catch (const std::exception&) {
        // Try next.
      }
    }
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] " << op_name << " requires an integer-vector argument";
  throw std::invalid_argument(out.str());
}

static Shape flatten_shape_from_arguments(const OrderedJson& arguments, const Shape& input_shape) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Flatten arguments " << arguments.dump()
        << "; expected [start_axis, end_axis]";
    throw std::runtime_error(out.str());
  }

  const auto rank = input_shape.size();
  if (rank <= 0) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Flatten input shape must have rank >= 1");
  }

  const auto start_axis = normalized_integer_scalar(arguments.at(0), "Flatten start_axis");
  const auto end_axis = normalized_integer_scalar(arguments.at(1), "Flatten end_axis");
  const auto start_index = normalize_axis(start_axis, rank, "Flatten start_axis");
  const auto end_index = normalize_axis(end_axis, rank, "Flatten end_axis");
  if (end_index < start_index) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Flatten axis range " << arguments.dump()
        << " for rank " << rank;
    throw std::runtime_error(out.str());
  }

  Shape out;
  out.insert(out.end(), input_shape.begin(), input_shape.begin() + start_index);

  int64_t middle = 1;
  for (int64_t axis = start_index; axis <= end_index; ++axis) {
    middle *= input_shape[static_cast<size_t>(axis)];
  }
  out.push_back(middle);

  out.insert(out.end(), input_shape.begin() + end_index + 1, input_shape.end());
  return out;
}

static Shape unflatten_shape_from_arguments(const OrderedJson& arguments, const Shape& input_shape) {
  if (!(arguments.is_array() && arguments.size() >= 2)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Unflatten arguments must include [axis, shape]");
  }

  const auto axis = normalized_integer_scalar(arguments.at(0), "Unflatten axis");
  auto target_shape = normalize_integer_vector(arguments.at(1), "Unflatten shape");
  const auto source_rank = input_shape.size();
  const auto axis_index = normalize_axis(axis, source_rank, "Unflatten axis");
  const auto source_dim = input_shape[static_cast<size_t>(axis_index)];

  std::vector<size_t> negative_indices;
  int64_t known_product = 1;

  for (size_t i = 0; i < target_shape.size(); ++i) {
    const auto dim = target_shape[i];
    if (dim == -1) {
      negative_indices.push_back(i);
      continue;
    }
    if (dim <= 0) {
      throw std::invalid_argument("[graph_ir_to_onnx_stub] Unflatten shape values must be positive or -1");
    }
    known_product *= dim;
  }

  if (negative_indices.size() > 1) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Unflatten shape " << json_from_int_vector(target_shape).dump()
        << "; at most one -1 is allowed";
    throw std::runtime_error(out.str());
  }

  if (negative_indices.size() == 1) {
    const auto unknown_index = negative_indices.front();
    if (known_product <= 0 || source_dim % known_product != 0) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Unflatten shape " << json_from_int_vector(target_shape).dump()
          << " for source dim " << source_dim;
      throw std::runtime_error(out.str());
    }
    target_shape[unknown_index] = source_dim / known_product;
  } else if (known_product != source_dim) {
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Unflatten shape " << json_from_int_vector(target_shape).dump()
        << "; product " << known_product << " must match source dim " << source_dim;
    throw std::runtime_error(out.str());
  }

  Shape out;
  out.insert(out.end(), input_shape.begin(), input_shape.begin() + axis_index);
  out.insert(out.end(), target_shape.begin(), target_shape.end());
  out.insert(out.end(), input_shape.begin() + axis_index + 1, input_shape.end());
  return out;
}

static bool integer_like_numeric(const OrderedJson& value) {
  if (value.is_number_integer() || value.is_number_unsigned()) {
    return true;
  }
  if (!value.is_number_float()) {
    return false;
  }
  const double raw = value.get<double>();
  return std::isfinite(raw) && std::trunc(raw) == raw;
}

static ArangeArguments arange_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 3)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Arange arguments must include [start, stop, step]");
  }

  const auto& start = arguments.at(0);
  const auto& stop = arguments.at(1);
  const auto& step = arguments.at(2);
  if (!(json_is_numeric(start) && json_is_numeric(stop) && json_is_numeric(step))) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Arange start/stop/step must be Numeric");
  }

  ArangeArguments out;
  if (integer_like_numeric(start) && integer_like_numeric(stop) && integer_like_numeric(step)) {
    out.integral = true;
    out.start_i = normalized_integer_scalar(start, "Arange start");
    out.stop_i = normalized_integer_scalar(stop, "Arange stop");
    out.step_i = normalized_integer_scalar(step, "Arange step");
    if (out.step_i == 0) {
      throw std::invalid_argument("[graph_ir_to_onnx_stub] Arange step must not be zero");
    }
    out.dtype = "int64";
    return out;
  }

  out.integral = false;
  out.start_f = start.get<double>();
  out.stop_f = stop.get<double>();
  out.step_f = step.get<double>();
  if (!(std::isfinite(out.start_f) && std::isfinite(out.stop_f) && std::isfinite(out.step_f))) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Arange start/stop/step must be finite Numeric values");
  }
  if (out.step_f == 0.0) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Arange step must not be zero");
  }
  out.dtype = "float32";
  return out;
}

static OrderedJson arange_values(const ArangeArguments& args) {
  OrderedJson values = OrderedJson::array();
  if (args.integral) {
    int64_t current = args.start_i;
    if (args.step_i > 0) {
      while (current < args.stop_i) {
        values.push_back(current);
        current += args.step_i;
      }
    } else {
      while (current > args.stop_i) {
        values.push_back(current);
        current += args.step_i;
      }
    }
    return values;
  }

  double current = args.start_f;
  if (args.step_f > 0.0) {
    while (current < args.stop_f) {
      values.push_back(current);
      current += args.step_f;
    }
  } else {
    while (current > args.stop_f) {
      values.push_back(current);
      current += args.step_f;
    }
  }
  return values;
}

static std::pair<double, double> addmm_alpha_beta(const OrderedJson& arguments) {
  if (!arguments.is_array() || arguments.empty()) {
    return {1.0, 1.0};
  }
  if (arguments.size() < 2 || !(json_is_numeric(arguments.at(0)) && json_is_numeric(arguments.at(1)))) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] AddMM arguments must include [alpha, beta]");
  }
  return {arguments.at(0).get<double>(), arguments.at(1).get<double>()};
}

static bool sqrt_is_reciprocal(const OrderedJson& arguments) {
  if (!arguments.is_array() || arguments.empty()) {
    return false;
  }
  if (!arguments.at(0).is_boolean()) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] Sqrt reciprocal flag must be boolean when present");
  }
  return arguments.at(0).get<bool>();
}

static AsStridedArguments asstrided_arguments(const OrderedJson& arguments) {
  if (!(arguments.is_array() && arguments.size() >= 3)) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] AsStrided arguments must include [shape, strides, offset]");
  }

  AsStridedArguments out;
  out.output_shape = normalize_integer_vector(arguments.at(0), "AsStrided shape");
  out.strides = normalize_integer_vector(arguments.at(1), "AsStrided strides");
  out.offset = normalized_integer_scalar(arguments.at(2), "AsStrided offset");

  if (out.output_shape.size() != out.strides.size()) {
    std::ostringstream msg;
    msg << "[graph_ir_to_onnx_stub] AsStrided shape/strides length mismatch: "
        << out.output_shape.size() << "/" << out.strides.size();
    throw std::invalid_argument(msg.str());
  }

  if (std::any_of(out.output_shape.begin(), out.output_shape.end(), [](int64_t value) { return value < 0; })) {
    throw std::invalid_argument("[graph_ir_to_onnx_stub] AsStrided shape values must be non-negative");
  }

  return out;
}

static int64_t tensor_size_from_shape(const Shape& shape) {
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
}

static std::vector<int64_t> asstrided_linear_indices(
    const Shape& output_shape,
    const std::vector<int64_t>& strides,
    int64_t offset,
    int64_t input_size) {
  if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t value) { return value == 0; })) {
    return {};
  }

  int64_t total = output_shape.empty()
      ? 1
      : std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

  std::vector<int64_t> indices;
  indices.reserve(static_cast<size_t>(total));

  for (int64_t linear_index = 0; linear_index < total; ++linear_index) {
    int64_t remainder = linear_index;
    int64_t source_index = offset;

    for (int64_t axis = static_cast<int64_t>(output_shape.size()) - 1; axis >= 0; --axis) {
      const auto dim = output_shape[static_cast<size_t>(axis)];
      const auto coord = remainder % dim;
      remainder /= dim;
      source_index += coord * strides[static_cast<size_t>(axis)];
    }

    if (source_index < 0 || source_index >= input_size) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported AsStrided index " << source_index
          << " out of bounds for input size " << input_size;
      throw std::runtime_error(out.str());
    }
    indices.push_back(source_index);
  }

  return indices;
}

static std::pair<std::vector<std::string>, std::vector<OrderedJson>> cast_inputs_to_dtype(
    size_t node_index,
    const std::string& op_name,
    const std::vector<std::string>& inputs,
    const std::string& target_dtype,
    ShapeMap& known_shapes,
    DtypeMap& known_dtypes,
    NameSet& used_tensor_names,
    const std::optional<std::vector<size_t>>& indices = std::nullopt) {
  std::vector<OrderedJson> cast_nodes;
  std::vector<std::string> casted_inputs = inputs;
  const auto& cast_to = onnx_dtype_symbol(target_dtype);

  std::unordered_set<size_t> index_filter;
  if (indices.has_value()) {
    for (const auto index : indices.value()) {
      index_filter.insert(index);
    }
  }

  for (size_t index = 0; index < inputs.size(); ++index) {
    if (indices.has_value() && index_filter.find(index) == index_filter.end()) {
      continue;
    }

    const auto& input_name = inputs[index];
    const auto input_dtype = canonical_dtype(known_dtype_for(known_dtypes, input_name));
    if (!input_dtype.has_value() || input_dtype.value() == target_dtype) {
      continue;
    }

    const auto cast_output = unique_aux_tensor_name(
        used_tensor_names,
        node_index,
        op_name + "_input" + std::to_string(index) + "_cast");

    cast_nodes.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_" + op_name + "CastInput" + std::to_string(index),
        "Cast",
        {input_name},
        {cast_output},
        OrderedJson::object({{"to", cast_to}})));

    const auto input_shape = known_shape_for(known_shapes, input_name);
    if (input_shape.has_value()) {
      known_shapes[cast_output] = input_shape.value();
    }
    known_dtypes[cast_output] = target_dtype;
    casted_inputs[index] = cast_output;
  }

  return {casted_inputs, cast_nodes};
}

static std::string append_aux_int64_initializer(
    OrderedJson& initializers,
    NameSet& used_tensor_names,
    size_t node_index,
    const std::string& label,
    const std::vector<int64_t>& values) {
  const auto name = unique_aux_tensor_name(used_tensor_names, node_index, label);

  OrderedJson tensor = OrderedJson::object();
  tensor["name"] = name;
  tensor["shape"] = json_from_int_vector({static_cast<int64_t>(values.size())});
  tensor["dtype"] = "int64";
  tensor["values"] = json_from_int_vector(values);

  initializers.push_back(onnx_initializer_info(tensor));
  return name;
}

static std::optional<std::string> flatten_onnx_op_type(
    const OrderedJson& arguments,
    bool strict,
    const OrderedJson* node,
    const ShapeMap* known_shapes) {
  if (!(arguments.is_array() && arguments.size() == 2 && json_is_numeric(arguments.at(0)) && json_is_numeric(arguments.at(1)))) {
    if (!strict) {
      return std::nullopt;
    }
    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Flatten arguments " << arguments.dump()
        << "; expected [start_axis, end_axis]";
    throw std::runtime_error(out.str());
  }

  if (known_shapes != nullptr && node != nullptr) {
    const auto inputs = json_string_vector(node->at("inputs"), "Flatten inputs");
    if (inputs.empty()) {
      throw std::runtime_error("[graph_ir_to_onnx_stub] Flatten requires one input");
    }
    const auto input_name = inputs.front();
    const auto input_shape = known_shape_for(*known_shapes, input_name);
    if (!input_shape.has_value()) {
      if (!strict) {
        return std::nullopt;
      }
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Flatten for tensor " << input_name
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    flatten_shape_from_arguments(arguments, input_shape.value());
  }

  return onnx_op_name("Flatten");
}

static std::optional<std::string> convolution_onnx_op_type(const OrderedJson& arguments, bool strict) {
  const auto parsed = convolution_attributes_from_arguments(arguments, strict);
  if (!parsed.has_value()) {
    return std::nullopt;
  }

  if (parsed->flip) {
    return onnx_op_name("ConvolutionTranspose");
  }

  if (std::any_of(parsed->input_dilation.begin(), parsed->input_dilation.end(), [](int64_t value) { return value != 1; })) {
    if (!strict) {
      return std::nullopt;
    }

    std::ostringstream out;
    out << "[graph_ir_to_onnx_stub] unsupported Convolution input_dilation "
        << json_from_int_vector(parsed->input_dilation).dump()
        << "; only all-ones input_dilation is supported for flip=false";
    throw std::runtime_error(out.str());
  }

  return onnx_op_name("Convolution");
}

static std::optional<std::string> onnx_op_type_for_node(
    const OrderedJson& node,
    bool strict,
    const ShapeMap* known_shapes) {
  const auto op = node.at("op").get<std::string>();
  const OrderedJson arguments = node.contains("arguments") ? node.at("arguments") : OrderedJson::array();

  if (op == "Convolution") {
    return convolution_onnx_op_type(arguments, strict);
  }
  if (op == "Reduce") {
    return reduce_onnx_op_type(arguments, strict);
  }
  if (op == "ArgReduce") {
    return argreduce_onnx_op_type(arguments, strict);
  }
  if (op == "Flatten") {
    return flatten_onnx_op_type(arguments, strict, &node, known_shapes);
  }
  if (op == "Concatenate") {
    if (!concatenate_axis_from_arguments(arguments, strict).has_value()) {
      return std::nullopt;
    }
    return onnx_op_name(op);
  }

  const auto it = kOnnxOpMap.find(op);
  if (it != kOnnxOpMap.end()) {
    return it->second;
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported op " << op;
  throw std::runtime_error(out.str());
}

static std::vector<OrderedJson> lower_onnx_node_default(
    const OrderedJson& node,
    size_t node_index,
    OrderedJson& initializers,
    NameSet& used_tensor_names,
    ShapeMap& known_shapes,
    DtypeMap& known_dtypes) {
  const auto op = node.at("op").get<std::string>();
  const auto op_type = onnx_op_type_for_node(node, true, &known_shapes).value();

  auto inputs = json_string_vector(node.at("inputs"), "node inputs");
  auto outputs = json_string_vector(node.at("outputs"), "node outputs");
  OrderedJson attributes = onnx_node_attributes(node);
  const OrderedJson arguments = node.contains("arguments") ? node.at("arguments") : OrderedJson::array();

  std::optional<Shape> inferred_output_shape;
  std::optional<std::string> inferred_output_dtype;

  if (op == "Arange") {
    const auto parsed = arange_arguments(arguments);
    const auto values = arange_values(parsed);

    OrderedJson tensor = OrderedJson::object();
    tensor["name"] = outputs.at(0);
    tensor["shape"] = json_from_int_vector({static_cast<int64_t>(values.size())});
    tensor["dtype"] = parsed.dtype;
    tensor["values"] = values;

    initializers.push_back(onnx_initializer_info(tensor));
    known_shapes[outputs.at(0)] = {static_cast<int64_t>(values.size())};
    known_dtypes[outputs.at(0)] = parsed.dtype;
    return {};
  }

  if (op == "Transpose") {
    const auto input_name = inputs.at(0);
    const auto input_shape = known_shape_for(known_shapes, input_name);
    if (input_shape.has_value()) {
      auto perm_opt = transpose_perm_from_arguments(arguments);
      std::vector<int64_t> perm;
      if (perm_opt.has_value() && !perm_opt->empty()) {
        perm = perm_opt.value();
      } else {
        perm.reserve(input_shape->size());
        for (int64_t i = static_cast<int64_t>(input_shape->size()) - 1; i >= 0; --i) {
          perm.push_back(i);
        }
      }
      inferred_output_shape = permute_shape(input_shape.value(), perm, "Transpose permutation");
    }
    inferred_output_dtype = known_dtype_for(known_dtypes, input_name);
  }

  if (op == "Convolution") {
    const auto convolution = convolution_attributes_from_arguments(arguments, true).value();

    if (convolution.flip) {
      const auto spatial_rank = convolution.spatial_rank;
      const auto [input_perm, output_perm] = convolution_data_permutations(spatial_rank);
      const auto weight_perm = convolution_transpose_weight_permutation(spatial_rank);

      const auto input_shape = known_shape_for(known_shapes, inputs[0]);
      const auto weight_shape = known_shape_for(known_shapes, inputs[1]);
      if (!weight_shape.has_value()) {
        throw std::runtime_error(
            "[graph_ir_to_onnx_stub] unsupported Convolution flip=true without known static weight shape");
      }

      const auto transposed_input = unique_aux_tensor_name(used_tensor_names, node_index, "conv_transpose_input_ncx");
      const auto transposed_weight = unique_aux_tensor_name(used_tensor_names, node_index, "conv_transpose_weight_icx");
      const auto conv_output = unique_aux_tensor_name(used_tensor_names, node_index, "conv_transpose_output_ncx");

      if (input_shape.has_value()) {
        known_shapes[transposed_input] = permute_shape(
            input_shape.value(),
            input_perm,
            "ConvolutionTranspose input permutation");
      }
      known_shapes[transposed_weight] = permute_shape(
          weight_shape.value(),
          weight_perm,
          "ConvolutionTranspose weight permutation");

      const auto conv_transpose = convtranspose_attributes_from_convolution(convolution, weight_shape.value());
      inferred_output_shape = infer_convolution_transpose_output_shape(
          input_shape,
          weight_shape,
          conv_transpose.strides,
          conv_transpose.pads_begin,
          conv_transpose.pads_end,
          conv_transpose.dilations,
          conv_transpose.output_padding,
          convolution.groups);
      if (inferred_output_shape.has_value()) {
        known_shapes[conv_output] = permute_shape(
            inferred_output_shape.value(),
            input_perm,
            "ConvolutionTranspose output permutation");
        for (const auto& name : outputs) {
          known_shapes[name] = inferred_output_shape.value();
        }
      }

      OrderedJson conv_transpose_attributes = OrderedJson::object();
      conv_transpose_attributes["strides"] = json_from_int_vector(conv_transpose.strides);
      conv_transpose_attributes["pads"] = json_from_int_vector(conv_transpose.pads);
      conv_transpose_attributes["dilations"] = json_from_int_vector(conv_transpose.dilations);
      conv_transpose_attributes["group"] = convolution.groups;
      conv_transpose_attributes["output_padding"] = json_from_int_vector(conv_transpose.output_padding);

      const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
      const auto weight_dtype = known_dtype_for(known_dtypes, inputs[1]);
      auto conv_output_dtype = promote_binary_dtype(input_dtype, weight_dtype);
      if (!conv_output_dtype.has_value()) {
        conv_output_dtype = input_dtype.has_value() ? input_dtype : weight_dtype;
      }

      if (input_dtype.has_value()) {
        known_dtypes[transposed_input] = input_dtype.value();
      }
      if (weight_dtype.has_value()) {
        known_dtypes[transposed_weight] = weight_dtype.value();
      }
      if (conv_output_dtype.has_value()) {
        known_dtypes[conv_output] = conv_output_dtype.value();
        for (const auto& name : outputs) {
          known_dtypes[name] = conv_output_dtype.value();
        }
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_InputTranspose",
              "Transpose",
              {inputs[0]},
              {transposed_input},
              OrderedJson::object({{"perm", json_from_int_vector(input_perm)}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_WeightTranspose",
              "Transpose",
              {inputs[1]},
              {transposed_weight},
              OrderedJson::object({{"perm", json_from_int_vector(weight_perm)}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_ConvTranspose",
              "ConvTranspose",
              {transposed_input, transposed_weight},
              {conv_output},
              conv_transpose_attributes),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_OutputTranspose",
              "Transpose",
              {conv_output},
              outputs,
              OrderedJson::object({{"perm", json_from_int_vector(output_perm)}}))};
    }

    if (std::any_of(convolution.input_dilation.begin(), convolution.input_dilation.end(), [](int64_t value) { return value != 1; })) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Convolution input_dilation "
          << json_from_int_vector(convolution.input_dilation).dump()
          << "; only all-ones input_dilation is supported for flip=false";
      throw std::runtime_error(out.str());
    }

    const auto spatial_rank = convolution.spatial_rank;
    const auto [input_perm, output_perm] = convolution_data_permutations(spatial_rank);
    const auto weight_perm = convolution_weight_permutation(spatial_rank);

    const auto input_shape = known_shape_for(known_shapes, inputs[0]);
    const auto weight_shape = known_shape_for(known_shapes, inputs[1]);
    const size_t input_rank = spatial_rank + 2;

    if (input_shape.has_value() && input_shape->size() != input_rank) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] Convolution input rank mismatch: expected "
          << input_rank << ", got " << input_shape->size();
      throw std::invalid_argument(out.str());
    }
    if (weight_shape.has_value() && weight_shape->size() != input_rank) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] Convolution weight rank mismatch: expected "
          << input_rank << ", got " << weight_shape->size();
      throw std::invalid_argument(out.str());
    }

    const auto transposed_input = unique_aux_tensor_name(used_tensor_names, node_index, "conv_input_ncx");
    const auto transposed_weight = unique_aux_tensor_name(used_tensor_names, node_index, "conv_weight_ocx");
    const auto conv_output = unique_aux_tensor_name(used_tensor_names, node_index, "conv_output_ncx");

    if (input_shape.has_value()) {
      known_shapes[transposed_input] = permute_shape(
          input_shape.value(),
          input_perm,
          "Convolution input permutation");
    }
    if (weight_shape.has_value()) {
      known_shapes[transposed_weight] = permute_shape(
          weight_shape.value(),
          weight_perm,
          "Convolution weight permutation");
    }

    inferred_output_shape = infer_convolution_output_shape(
        input_shape,
        weight_shape,
        convolution.strides,
        convolution.padding_low,
        convolution.padding_high,
        convolution.kernel_dilation,
        convolution.groups);
    if (inferred_output_shape.has_value()) {
      known_shapes[conv_output] = permute_shape(
          inferred_output_shape.value(),
          input_perm,
          "Convolution output permutation");
      for (const auto& name : outputs) {
        known_shapes[name] = inferred_output_shape.value();
      }
    }

    OrderedJson conv_attributes = OrderedJson::object();
    conv_attributes["strides"] = json_from_int_vector(convolution.strides);
    conv_attributes["pads"] = json_from_int_vector(convolution.pads);
    conv_attributes["dilations"] = json_from_int_vector(convolution.kernel_dilation);
    conv_attributes["group"] = convolution.groups;

    return {
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_InputTranspose",
            "Transpose",
            {inputs[0]},
            {transposed_input},
            OrderedJson::object({{"perm", json_from_int_vector(input_perm)}})),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_WeightTranspose",
            "Transpose",
            {inputs[1]},
            {transposed_weight},
            OrderedJson::object({{"perm", json_from_int_vector(weight_perm)}})),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_Conv",
            "Conv",
            {transposed_input, transposed_weight},
            {conv_output},
            conv_attributes),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_OutputTranspose",
            "Transpose",
            {conv_output},
            outputs,
            OrderedJson::object({{"perm", json_from_int_vector(output_perm)}}))};
  }

  if (op == "Reduce") {
    const auto reduce_code = arguments.is_array() && !arguments.empty() ? normalized_integer_scalar(arguments.at(0), "Reduce code") : 0;
    const auto axes = reduce_axes_from_arguments(arguments);
    const auto axes_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axes", axes);

    inferred_output_shape = infer_reduce_keepdims_shape(known_shape_for(known_shapes, inputs.front()), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs.front());

    if (reduce_code == 0 || reduce_code == 1) {
      const auto cast_bool_out = unique_aux_tensor_name(used_tensor_names, node_index, "cast_bool");
      const auto cast_int_out = unique_aux_tensor_name(used_tensor_names, node_index, "cast_int64");
      const auto reduce_out = unique_aux_tensor_name(used_tensor_names, node_index, "reduce");
      const auto reduce_type = kReduceCodeToOnnxOp.at(reduce_code);

      for (const auto& name : outputs) {
        if (inferred_output_shape.has_value()) {
          known_shapes[name] = inferred_output_shape.value();
        }
        known_dtypes[name] = "bool";
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_CastToBool",
              "Cast",
              {inputs.front()},
              {cast_bool_out},
              OrderedJson::object({{"to", "BOOL"}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_CastToInt64",
              "Cast",
              {cast_bool_out},
              {cast_int_out},
              OrderedJson::object({{"to", "INT64"}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_" + reduce_type,
              reduce_type,
              {cast_int_out, axes_name},
              {reduce_out},
              OrderedJson::object({{"keepdims", 1}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_CastOutBool",
              "Cast",
              {reduce_out},
              outputs,
              OrderedJson::object({{"to", "BOOL"}}))};
    }

    inputs.push_back(axes_name);
    attributes["keepdims"] = 1;
  }

  if (op == "AsType") {
    const auto target_dtype = onnx_effective_dtype(as_type_target_dtype(arguments, outputs, known_dtypes));
    attributes["to"] = onnx_dtype_symbol(target_dtype);
    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    inferred_output_dtype = target_dtype;
  }

  if (op == "Reshape") {
    const auto shape = integer_vector_argument(arguments, "Reshape");
    const auto shape_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "shape", shape);
    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Add" || op == "Subtract" || op == "Multiply" || op == "Divide" ||
      op == "Maximum" || op == "Minimum" || op == "Power") {
    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);

    if (promoted_dtype.has_value()) {
      auto [casted_inputs, cast_nodes] = cast_inputs_to_dtype(
          node_index,
          op,
          inputs,
          promoted_dtype.value(),
          known_shapes,
          known_dtypes,
          used_tensor_names);
      if (!cast_nodes.empty()) {
        inputs = casted_inputs;
        inferred_output_shape = infer_elementwise_output_shape(
            known_shape_for(known_shapes, inputs[0]),
            known_shape_for(known_shapes, inputs[1]));
        inferred_output_dtype = promoted_dtype;
        if (inferred_output_shape.has_value()) {
          for (const auto& name : outputs) {
            known_shapes[name] = inferred_output_shape.value();
          }
        }
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
        cast_nodes.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_" + op_type,
            op_type,
            inputs,
            outputs,
            attributes));
        return cast_nodes;
      }
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = promoted_dtype.has_value()
        ? promoted_dtype
        : (lhs_dtype.has_value() ? lhs_dtype : rhs_dtype);
  }

  if (op == "Exp" || op == "Log" || op == "Abs" || op == "Negative" || op == "Relu" || op == "Sigmoid" ||
      op == "Tanh" || op == "Softmax" || op == "Sin" || op == "Cos" || op == "Erf" || op == "Floor") {
    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Sqrt") {
    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
    if (sqrt_is_reciprocal(arguments)) {
      const auto sqrt_output = unique_aux_tensor_name(used_tensor_names, node_index, "sqrt");
      if (inferred_output_shape.has_value()) {
        known_shapes[sqrt_output] = inferred_output_shape.value();
      }
      if (inferred_output_dtype.has_value()) {
        known_dtypes[sqrt_output] = inferred_output_dtype.value();
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_Sqrt",
              "Sqrt",
              inputs,
              {sqrt_output},
              OrderedJson::object()),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_Reciprocal",
              "Reciprocal",
              {sqrt_output},
              outputs,
              OrderedJson::object())};
    }
  }

  if (op == "Matmul") {
    inferred_output_shape = infer_matmul_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "AddMM") {
    const auto [alpha, beta] = addmm_alpha_beta(arguments);
    if (alpha != 1.0) {
      attributes["alpha"] = alpha;
    }
    if (beta != 1.0) {
      attributes["beta"] = beta;
    }
    attributes["transA"] = 0;
    attributes["transB"] = 0;
    inferred_output_shape = infer_matmul_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Square") {
    const auto square_input = inputs.at(0);
    inputs = {square_input, square_input};
    inferred_output_shape = known_shape_for(known_shapes, square_input);
    inferred_output_dtype = known_dtype_for(known_dtypes, square_input);
  }

  if (op == "Gather") {
    const auto axis = gather_axis_from_arguments(arguments, true).value();
    std::vector<OrderedJson> pre_nodes;

    auto indices_input = inputs[1];
    const auto indices_dtype = canonical_dtype(known_dtype_for(known_dtypes, indices_input));
    if (indices_dtype.has_value() && indices_dtype.value() != "int32" && indices_dtype.value() != "int64") {
      const auto cast_indices = unique_aux_tensor_name(used_tensor_names, node_index, "gather_indices_cast");
      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherCastIndices",
          "Cast",
          {indices_input},
          {cast_indices},
          OrderedJson::object({{"to", onnx_dtype_symbol("int64")}})));
      const auto index_shape = known_shape_for(known_shapes, indices_input);
      if (index_shape.has_value()) {
        known_shapes[cast_indices] = index_shape.value();
      }
      known_dtypes[cast_indices] = "int64";
      inputs[1] = cast_indices;
      indices_input = cast_indices;
    }

    const auto data_shape = known_shape_for(known_shapes, inputs[0]);
    const auto indices_shape = known_shape_for(known_shapes, inputs[1]);
    if (!data_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Gather for tensor " << inputs[0]
          << " without known static shape";
      throw std::runtime_error(out.str());
    }
    if (!indices_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Gather for indices " << inputs[1]
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    const auto data_rank = static_cast<int64_t>(data_shape->size());
    const auto axis_index = normalize_axis(axis, static_cast<size_t>(data_rank), "Gather axis");
    const auto indices_rank = static_cast<int64_t>(indices_shape->size());

    const auto gather_output = unique_aux_tensor_name(used_tensor_names, node_index, "gather");
    const auto gather_reordered = unique_aux_tensor_name(used_tensor_names, node_index, "gather_reordered");
    const auto unsqueeze_axis = axis_index + indices_rank;
    const auto axes_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axes", {unsqueeze_axis});

    const auto gather_shape = infer_gather_output_shape(data_shape, indices_shape, axis_index);
    const auto perm = gather_reorder_permutation(data_rank, indices_rank, axis_index);
    const bool needs_reorder = !identity_permutation(perm);

    std::optional<Shape> reordered_shape;
    if (gather_shape.has_value()) {
      Shape permuted;
      permuted.reserve(perm.size());
      for (const auto dim_index : perm) {
        permuted.push_back(gather_shape->at(static_cast<size_t>(dim_index)));
      }
      reordered_shape = permuted;

      known_shapes[gather_output] = gather_shape.value();
      if (needs_reorder) {
        known_shapes[gather_reordered] = reordered_shape.value();
      }

      const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
      if (input_dtype.has_value()) {
        known_dtypes[gather_output] = input_dtype.value();
        if (needs_reorder) {
          known_dtypes[gather_reordered] = input_dtype.value();
        }
      }

      Shape final_shape = reordered_shape.value();
      final_shape.insert(final_shape.begin() + static_cast<long>(unsqueeze_axis), 1);
      for (const auto& name : outputs) {
        known_shapes[name] = final_shape;
        if (input_dtype.has_value()) {
          known_dtypes[name] = input_dtype.value();
        }
      }
    }

    std::vector<OrderedJson> lowered;
    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_Gather",
        "Gather",
        inputs,
        {gather_output},
        attributes));

    std::string unsqueeze_input = gather_output;
    if (needs_reorder) {
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherTranspose",
          "Transpose",
          {gather_output},
          {gather_reordered},
          OrderedJson::object({{"perm", json_from_int_vector(perm)}})));
      unsqueeze_input = gather_reordered;
    }

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_Unsqueeze",
        "Unsqueeze",
        {unsqueeze_input, axes_name},
        outputs,
        OrderedJson::object()));

    pre_nodes.insert(pre_nodes.end(), lowered.begin(), lowered.end());
    return pre_nodes;
  }

  if (op == "GatherAxis") {
    const auto axis = gather_axis_from_arguments(arguments, true).value();
    std::vector<OrderedJson> pre_nodes;

    auto indices_input = inputs[1];
    const auto indices_dtype = canonical_dtype(known_dtype_for(known_dtypes, indices_input));
    if (indices_dtype.has_value() && indices_dtype.value() != "int32" && indices_dtype.value() != "int64") {
      const auto cast_indices = unique_aux_tensor_name(used_tensor_names, node_index, "gatheraxis_indices_cast");
      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherAxisCastIndices",
          "Cast",
          {indices_input},
          {cast_indices},
          OrderedJson::object({{"to", onnx_dtype_symbol("int64")}})));
      const auto index_shape = known_shape_for(known_shapes, indices_input);
      if (index_shape.has_value()) {
        known_shapes[cast_indices] = index_shape.value();
      }
      known_dtypes[cast_indices] = "int64";
      inputs[1] = cast_indices;
      indices_input = cast_indices;
    }

    const auto data_shape = known_shape_for(known_shapes, inputs[0]);
    const auto indices_shape = known_shape_for(known_shapes, inputs[1]);
    if (!data_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported GatherAxis for tensor " << inputs[0]
          << " without known static shape";
      throw std::runtime_error(out.str());
    }
    if (!indices_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported GatherAxis for indices " << inputs[1]
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    const auto data_rank = static_cast<int64_t>(data_shape->size());
    const auto indices_rank = static_cast<int64_t>(indices_shape->size());
    const auto axis_index = normalize_axis(axis, static_cast<size_t>(data_rank), "GatherAxis axis");

    if (data_rank != indices_rank) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported GatherAxis rank mismatch: data rank "
          << data_rank << ", indices rank " << indices_rank;
      throw std::runtime_error(out.str());
    }

    const auto data_dims = data_shape.value();
    const auto indices_dims = indices_shape.value();
    Shape expanded_data_shape = data_dims;
    bool needs_data_expand = false;

    for (size_t dim_index = 0; dim_index < data_dims.size(); ++dim_index) {
      if (static_cast<int64_t>(dim_index) == axis_index) {
        continue;
      }
      const auto dim = data_dims[dim_index];
      const auto index_dim = indices_dims[dim_index];
      if (dim == index_dim) {
        continue;
      }
      if (dim == 1) {
        expanded_data_shape[dim_index] = index_dim;
        needs_data_expand = true;
        continue;
      }
      if (index_dim <= dim) {
        continue;
      }

      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported GatherAxis shape mismatch at dim " << dim_index
          << ": data=" << dim << ", indices=" << index_dim;
      throw std::runtime_error(out.str());
    }

    if (needs_data_expand) {
      const auto expand_shape_name = append_aux_int64_initializer(
          initializers,
          used_tensor_names,
          node_index,
          "gatheraxis_expand_shape",
          expanded_data_shape);
      const auto expanded_data = unique_aux_tensor_name(used_tensor_names, node_index, "gatheraxis_expanded_data");

      known_shapes[expanded_data] = expanded_data_shape;
      const auto input_dtype = known_dtype_for(known_dtypes, inputs[0]);
      if (input_dtype.has_value()) {
        known_dtypes[expanded_data] = input_dtype.value();
      }

      inferred_output_shape = indices_dims;
      inferred_output_dtype = input_dtype;

      if (inferred_output_shape.has_value()) {
        for (const auto& name : outputs) {
          known_shapes[name] = inferred_output_shape.value();
        }
      }
      if (inferred_output_dtype.has_value()) {
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
      }

      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_GatherAxisExpand",
          "Expand",
          {inputs[0], expand_shape_name},
          {expanded_data},
          OrderedJson::object()));
      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_" + op_type,
          op_type,
          {expanded_data, inputs[1]},
          outputs,
          attributes));
      return pre_nodes;
    }

    inferred_output_shape = indices_dims;
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);

    if (!pre_nodes.empty()) {
      if (inferred_output_shape.has_value()) {
        for (const auto& name : outputs) {
          known_shapes[name] = inferred_output_shape.value();
        }
      }
      if (inferred_output_dtype.has_value()) {
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
      }

      pre_nodes.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_" + op_type,
          op_type,
          inputs,
          outputs,
          attributes));
      return pre_nodes;
    }
  }

  if (op == "LogSumExp") {
    const auto input_shape = known_shape_for(known_shapes, inputs[0]);
    if (!input_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported LogSumExp for tensor " << inputs[0]
          << " without known static shape";
      throw std::runtime_error(out.str());
    }
    const auto output_shape = known_shape_for(known_shapes, outputs.front());
    const auto axes = infer_logsumexp_axes(input_shape.value(), output_shape);
    const auto axes_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axes", axes);
    inputs.push_back(axes_name);
    attributes["keepdims"] = 1;
    inferred_output_shape = infer_reduce_keepdims_shape(input_shape, axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Pad") {
    const auto input_shape = known_shape_for(known_shapes, inputs[0]);
    if (!input_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Pad for tensor " << inputs[0]
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    const auto [axes, pad_low, pad_high] = pad_axes_and_sizes_from_arguments(arguments, input_shape.value());
    const auto rank = input_shape->size();

    std::vector<int64_t> pads_begin(rank, 0);
    std::vector<int64_t> pads_end(rank, 0);
    for (size_t i = 0; i < axes.size(); ++i) {
      const auto axis_index = normalize_axis(axes[i], rank, "Pad axis");
      pads_begin[static_cast<size_t>(axis_index)] = pad_low[i];
      pads_end[static_cast<size_t>(axis_index)] = pad_high[i];
    }

    std::vector<int64_t> pads = pads_begin;
    pads.insert(pads.end(), pads_end.begin(), pads_end.end());
    const auto pads_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "pads", pads);

    if (!(inputs.size() >= 1 && inputs.size() <= 2)) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Pad input arity " << inputs.size()
          << "; expected 1 or 2 inputs";
      throw std::runtime_error(out.str());
    }

    std::vector<std::string> padded_inputs = {inputs.front(), pads_name};
    if (inputs.size() == 2) {
      padded_inputs.push_back(inputs[1]);
    }
    inputs = std::move(padded_inputs);

    attributes["mode"] = "constant";
    inferred_output_shape = infer_pad_output_shape(input_shape, pads_begin, pads_end);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs.front());
  }

  if (op == "Scan") {
    const auto parsed = scan_arguments(arguments);
    if (parsed.reduce_type != 2) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Scan reduce_type " << parsed.reduce_type
          << "; only CumSum (2) is supported";
      throw std::runtime_error(out.str());
    }

    int64_t axis_value = parsed.axis;
    const auto input_shape = known_shape_for(known_shapes, inputs[0]);
    if (input_shape.has_value()) {
      axis_value = normalize_axis(parsed.axis, input_shape->size(), "Scan axis");
    }

    const auto axis_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axis", {axis_value});
    inputs.push_back(axis_name);
    if (!parsed.inclusive) {
      attributes["exclusive"] = 1;
    }
    if (parsed.reverse) {
      attributes["reverse"] = 1;
    }

    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    auto output_dtype = known_dtype_for(known_dtypes, outputs.front());
    inferred_output_dtype = output_dtype.has_value() ? output_dtype : known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Slice") {
    const auto [starts, ends, axes, steps] = slice_vectors_from_arguments(arguments);
    const auto starts_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "starts", starts);
    const auto ends_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "ends", ends);
    const auto axes_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axes", axes);
    const auto steps_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "steps", steps);

    inputs.push_back(starts_name);
    inputs.push_back(ends_name);
    inputs.push_back(axes_name);
    inputs.push_back(steps_name);

    inferred_output_shape = infer_slice_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        starts,
        ends,
        axes,
        steps);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Split") {
    const auto [axis, lengths] = split_axis_and_lengths(arguments, known_shape_for(known_shapes, inputs[0]), static_cast<int64_t>(outputs.size()));
    const auto split_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "split", lengths);
    inputs.push_back(split_name);
    attributes["axis"] = axis;

    const auto split_shapes = infer_split_output_shapes(known_shape_for(known_shapes, inputs[0]), axis, lengths);
    if (split_shapes.has_value()) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        known_shapes[outputs[i]] = split_shapes->at(i);
      }
    }
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "ArgReduce") {
    const auto [arg_mode, arg_axis] = argreduce_mode_axis(arguments);
    const auto arg_op = kArgReduceCodeToOnnxOp.at(arg_mode);
    const auto arg_output = unique_aux_tensor_name(used_tensor_names, node_index, "argreduce");

    const auto arg_shape = infer_argreduce_keepdims_shape(known_shape_for(known_shapes, inputs[0]), arg_axis);
    if (arg_shape.has_value()) {
      known_shapes[arg_output] = arg_shape.value();
      for (const auto& name : outputs) {
        known_shapes[name] = arg_shape.value();
      }
    }

    return {
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_" + arg_op,
            arg_op,
            inputs,
            {arg_output},
            OrderedJson::object({{"axis", arg_axis}, {"keepdims", 1}})),
        build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_CastUint32",
            "Cast",
            {arg_output},
            outputs,
            OrderedJson::object({{"to", "UINT32"}}))};
  }

  if (op == "AsStrided") {
    const auto input_name = inputs.at(0);
    const auto input_shape = known_shape_for(known_shapes, input_name);
    if (!input_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported AsStrided for tensor " << input_name
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    const auto parsed = asstrided_arguments(arguments);
    const auto input_size = tensor_size_from_shape(input_shape.value());
    const auto indices = asstrided_linear_indices(parsed.output_shape, parsed.strides, parsed.offset, input_size);

    const auto indices_name = unique_aux_tensor_name(used_tensor_names, node_index, "asstrided_indices");
    OrderedJson indices_tensor = OrderedJson::object();
    indices_tensor["name"] = indices_name;
    indices_tensor["shape"] = json_from_shape(parsed.output_shape);
    indices_tensor["dtype"] = "int64";
    indices_tensor["values"] = json_from_int_vector(indices);
    initializers.push_back(onnx_initializer_info(indices_tensor));

    const auto input_rank = static_cast<int64_t>(input_shape->size());
    auto gather_input = input_name;
    std::vector<OrderedJson> lowered;
    if (input_rank != 1) {
      const auto flatten_shape_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "asstrided_flatten_shape", {-1});
      gather_input = unique_aux_tensor_name(used_tensor_names, node_index, "asstrided_input_flat");
      lowered.push_back(build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_AsStridedInputFlatten",
          "Reshape",
          {input_name, flatten_shape_name},
          {gather_input},
          OrderedJson::object()));
    }

    lowered.push_back(build_onnx_node_spec(
        "node_" + std::to_string(node_index) + "_AsStridedGather",
        "Gather",
        {gather_input, indices_name},
        outputs,
        OrderedJson::object({{"axis", 0}})));

    const auto input_dtype = known_dtype_for(known_dtypes, input_name);
    for (const auto& name : outputs) {
      known_shapes[name] = parsed.output_shape;
      if (input_dtype.has_value()) {
        known_dtypes[name] = input_dtype.value();
      }
    }

    return lowered;
  }

  if (op == "ScatterAxis") {
    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Greater" || op == "Less") {
    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);
    if (promoted_dtype.has_value()) {
      auto [casted_inputs, cast_nodes] = cast_inputs_to_dtype(
          node_index,
          op,
          inputs,
          promoted_dtype.value(),
          known_shapes,
          known_dtypes,
          used_tensor_names);
      if (!cast_nodes.empty()) {
        inputs = casted_inputs;
        inferred_output_shape = infer_elementwise_output_shape(
            known_shape_for(known_shapes, inputs[0]),
            known_shape_for(known_shapes, inputs[1]));
        inferred_output_dtype = "bool";
        if (inferred_output_shape.has_value()) {
          for (const auto& name : outputs) {
            known_shapes[name] = inferred_output_shape.value();
          }
        }
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
        cast_nodes.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_" + op_type,
            op_type,
            inputs,
            outputs,
            attributes));
        return cast_nodes;
      }
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = "bool";
  }

  if (op == "Equal") {
    if (equal_nan_from_arguments(arguments)) {
      throw std::runtime_error(
          "[graph_ir_to_onnx_stub] unsupported Equal equal_nan=true; only equal_nan=false is supported");
    }

    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[0]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);

    if (promoted_dtype.has_value()) {
      auto [casted_inputs, cast_nodes] = cast_inputs_to_dtype(
          node_index,
          op,
          inputs,
          promoted_dtype.value(),
          known_shapes,
          known_dtypes,
          used_tensor_names);
      if (!cast_nodes.empty()) {
        inputs = casted_inputs;
        inferred_output_shape = infer_elementwise_output_shape(
            known_shape_for(known_shapes, inputs[0]),
            known_shape_for(known_shapes, inputs[1]));
        inferred_output_dtype = "bool";
        if (inferred_output_shape.has_value()) {
          for (const auto& name : outputs) {
            known_shapes[name] = inferred_output_shape.value();
          }
        }
        for (const auto& name : outputs) {
          known_dtypes[name] = inferred_output_dtype.value();
        }
        cast_nodes.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_" + op_type,
            op_type,
            inputs,
            outputs,
            attributes));
        return cast_nodes;
      }
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[0]),
        known_shape_for(known_shapes, inputs[1]));
    inferred_output_dtype = "bool";
  }

  if (op == "Select") {
    const auto lhs_dtype = known_dtype_for(known_dtypes, inputs[1]);
    const auto rhs_dtype = known_dtype_for(known_dtypes, inputs[2]);
    const auto promoted_dtype = promote_binary_dtype(lhs_dtype, rhs_dtype);

    if (promoted_dtype.has_value()) {
      auto [casted_inputs, cast_nodes] = cast_inputs_to_dtype(
          node_index,
          op,
          inputs,
          promoted_dtype.value(),
          known_shapes,
          known_dtypes,
          used_tensor_names,
          std::optional<std::vector<size_t>>{{1, 2}});
      if (!cast_nodes.empty()) {
        inputs = casted_inputs;
        inferred_output_shape = infer_elementwise_output_shape(
            known_shape_for(known_shapes, inputs[1]),
            known_shape_for(known_shapes, inputs[2]));
        inferred_output_dtype = promoted_dtype;

        if (inferred_output_shape.has_value()) {
          for (const auto& name : outputs) {
            known_shapes[name] = inferred_output_shape.value();
          }
        }
        if (inferred_output_dtype.has_value()) {
          for (const auto& name : outputs) {
            known_dtypes[name] = inferred_output_dtype.value();
          }
        }

        cast_nodes.push_back(build_onnx_node_spec(
            "node_" + std::to_string(node_index) + "_" + op_type,
            op_type,
            inputs,
            outputs,
            attributes));
        return cast_nodes;
      }
    }

    inferred_output_shape = infer_elementwise_output_shape(
        known_shape_for(known_shapes, inputs[1]),
        known_shape_for(known_shapes, inputs[2]));
    inferred_output_dtype = promoted_dtype.has_value()
        ? promoted_dtype
        : (lhs_dtype.has_value() ? lhs_dtype : rhs_dtype);
  }

  if (op == "Full") {
    inferred_output_shape = known_shape_for(known_shapes, inputs[0]);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Concatenate") {
    const auto axis = concatenate_axis_from_arguments(arguments, true).value();
    std::vector<std::optional<Shape>> input_shapes;
    input_shapes.reserve(inputs.size());
    for (const auto& name : inputs) {
      input_shapes.push_back(known_shape_for(known_shapes, name));
    }
    inferred_output_shape = infer_concatenate_output_shape(input_shapes, axis);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs.front());
  }

  if (op == "Flatten") {
    const auto flatten_input = inputs.front();
    const auto input_shape = known_shape_for(known_shapes, flatten_input);
    if (!input_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Flatten for tensor " << flatten_input
          << " without known static shape";
      throw std::runtime_error(out.str());
    }
    const auto shape = flatten_shape_from_arguments(arguments, input_shape.value());
    const auto shape_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "shape", shape);
    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, flatten_input);
  }

  if (op == "Unflatten") {
    const auto unflatten_input = inputs.front();
    const auto input_shape = known_shape_for(known_shapes, unflatten_input);
    if (!input_shape.has_value()) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported Unflatten for tensor " << unflatten_input
          << " without known static shape";
      throw std::runtime_error(out.str());
    }

    const auto shape = unflatten_shape_from_arguments(arguments, input_shape.value());
    const auto shape_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "shape", shape);
    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, unflatten_input);
  }

  if (op == "Squeeze") {
    const auto axes = integer_vector_argument(arguments, "Squeeze");
    const auto axes_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axes", axes);
    inputs.push_back(axes_name);
    inferred_output_shape = infer_squeeze_output_shape(known_shape_for(known_shapes, inputs[0]), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "ExpandDims") {
    const auto axes = integer_vector_argument(arguments, "ExpandDims");
    const auto axes_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "axes", axes);
    inputs.push_back(axes_name);
    inferred_output_shape = infer_unsqueeze_output_shape(known_shape_for(known_shapes, inputs[0]), axes);
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (op == "Broadcast") {
    const auto shape = integer_vector_argument(arguments, "Broadcast");
    const auto shape_name = append_aux_int64_initializer(initializers, used_tensor_names, node_index, "shape", shape);
    const auto input_name = inputs.at(0);
    const auto input_dtype = known_dtype_for(known_dtypes, input_name);

    if (input_dtype.has_value() && input_dtype.value() == "bfloat16") {
      const auto cast_input = unique_aux_tensor_name(used_tensor_names, node_index, "broadcast_cast_input");
      const auto expand_output = unique_aux_tensor_name(used_tensor_names, node_index, "broadcast_expand_output");

      const auto input_shape = known_shape_for(known_shapes, input_name);
      if (input_shape.has_value()) {
        known_shapes[cast_input] = input_shape.value();
      }
      known_shapes[expand_output] = shape;
      known_dtypes[cast_input] = "float32";
      known_dtypes[expand_output] = "float32";

      for (const auto& name : outputs) {
        known_shapes[name] = shape;
        known_dtypes[name] = "bfloat16";
      }

      return {
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_BroadcastCastInput",
              "Cast",
              {input_name},
              {cast_input},
              OrderedJson::object({{"to", onnx_dtype_symbol("float32")}})),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_Expand",
              "Expand",
              {cast_input, shape_name},
              {expand_output},
              OrderedJson::object()),
          build_onnx_node_spec(
              "node_" + std::to_string(node_index) + "_BroadcastCastOutput",
              "Cast",
              {expand_output},
              outputs,
              OrderedJson::object({{"to", onnx_dtype_symbol("bfloat16")}}))};
    }

    inputs.push_back(shape_name);
    inferred_output_shape = shape;
    inferred_output_dtype = known_dtype_for(known_dtypes, inputs[0]);
  }

  if (inferred_output_shape.has_value()) {
    for (const auto& name : outputs) {
      known_shapes[name] = inferred_output_shape.value();
    }
  }
  if (inferred_output_dtype.has_value()) {
    for (const auto& name : outputs) {
      known_dtypes[name] = inferred_output_dtype.value();
    }
  }

  return {
      build_onnx_node_spec(
          "node_" + std::to_string(node_index) + "_" + op_type,
          op_type,
          inputs,
          outputs,
          attributes)};
}

static OrderedJson graph_ir_to_onnx_json_payload(
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name) {
  OrderedJson initializers = OrderedJson::array();
  for (const auto& tensor : payload.at("constants")) {
    initializers.push_back(onnx_initializer_info(tensor));
  }

  auto used_tensor_names = collect_payload_tensor_names(payload);
  auto known_shapes = collect_known_tensor_shapes(payload);
  auto known_dtypes = collect_known_tensor_dtypes(payload);

  OrderedJson nodes = OrderedJson::array();
  const auto& source_nodes = payload.at("nodes");
  for (size_t index = 0; index < source_nodes.size(); ++index) {
    auto lowered = lower_onnx_node_default(
        source_nodes.at(index),
        index,
        initializers,
        used_tensor_names,
        known_shapes,
        known_dtypes);
    for (auto& lowered_node : lowered) {
      nodes.push_back(std::move(lowered_node));
    }
  }

  OrderedJson graph = OrderedJson::object();
  graph["name"] = model_name;

  OrderedJson input_infos = OrderedJson::array();
  for (const auto& tensor : payload.at("inputs")) {
    input_infos.push_back(onnx_value_info(tensor));
  }
  graph["inputs"] = std::move(input_infos);

  OrderedJson output_infos = OrderedJson::array();
  for (const auto& tensor : payload.at("outputs")) {
    output_infos.push_back(onnx_value_info(tensor));
  }
  graph["outputs"] = std::move(output_infos);

  graph["initializers"] = std::move(initializers);
  graph["nodes"] = std::move(nodes);

  OrderedJson out = OrderedJson::object();
  out["format"] = "onnx_stub_v1";
  out["ir_version"] = kGraphIrVersion;
  out["opset"] = opset;
  out["producer_name"] = "mlx-ruby";
  out["graph"] = std::move(graph);
  return out;
}

static VALUE graph_ir_to_onnx_json_from_graph_ir_json(VALUE graph_ir_json, VALUE opset, VALUE model_name) {
  VALUE graph_ir_json_str = StringValue(graph_ir_json);

  const auto opset_int = normalize_positive_integer(opset, "opset");
  const auto model_name_str = non_empty_model_name(model_name);

  const std::string payload_raw(
      RSTRING_PTR(graph_ir_json_str),
      static_cast<size_t>(RSTRING_LEN(graph_ir_json_str)));

  OrderedJson payload;
  try {
    payload = OrderedJson::parse(payload_raw);
  } catch (const std::exception& error) {
    std::ostringstream out;
    out << "failed to parse graph ir json: " << error.what();
    throw std::invalid_argument(out.str());
  }

  const auto onnx_payload = graph_ir_to_onnx_json_payload(payload, opset_int, model_name_str);
  const auto content = onnx_payload.dump();
  return ruby_string_from_std(content);
}

static VALUE graph_ir_native_graph_ir_to_onnx_json(
    VALUE,
    VALUE graph_ir_json,
    VALUE opset,
    VALUE model_name) {
  try {
    return graph_ir_to_onnx_json_from_graph_ir_json(graph_ir_json, opset, model_name);
  } catch (const std::exception& error) {
    rb_raise(rb_eRuntimeError, "%s", error.what());
    return Qnil;
  }
}

static VALUE graph_ir_native_export_onnx_json(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless,
    VALUE opset,
    VALUE model_name) {
  try {
    if (!RB_TYPE_P(args_array, T_ARRAY)) {
      rb_raise(rb_eTypeError, "args_array must be an Array");
    }
    if (!(NIL_P(kwargs_hash) || RB_TYPE_P(kwargs_hash, T_HASH))) {
      rb_raise(rb_eTypeError, "kwargs_hash must be a Hash or nil");
    }
    if (!(shapeless == Qtrue || shapeless == Qfalse)) {
      rb_raise(rb_eTypeError, "shapeless must be true or false");
    }

    std::vector<VALUE> capture_argv;
    capture_argv.reserve(
        static_cast<size_t>(1 + RARRAY_LEN(args_array) + (NIL_P(kwargs_hash) ? 0 : 1) + 1));
    capture_argv.push_back(fun);

    const long args_len = RARRAY_LEN(args_array);
    for (long i = 0; i < args_len; ++i) {
      capture_argv.push_back(rb_ary_entry(args_array, i));
    }

    if (!NIL_P(kwargs_hash)) {
      capture_argv.push_back(kwargs_hash);
    }
    capture_argv.push_back(shapeless);

    VALUE graph_ir_json =
        core_native_export_graph_ir_json(static_cast<int>(capture_argv.size()), capture_argv.data(), Qnil);
    return graph_ir_to_onnx_json_from_graph_ir_json(graph_ir_json, opset, model_name);
  } catch (const std::exception& error) {
    rb_raise(rb_eRuntimeError, "%s", error.what());
    return Qnil;
  }
}

} // namespace

extern "C" void init_graph_ir_native_bindings(VALUE mMLX) {
  mGraphIR = rb_define_module_under(mMLX, "GraphIR");
  mGraphIRNative = rb_define_module_under(mGraphIR, "Native");

  rb_define_singleton_method(
      mGraphIRNative, "export_graph_ir_capture", RUBY_METHOD_FUNC(core_native_export_graph_ir), -1);
  rb_define_singleton_method(
      mGraphIRNative, "export_graph_ir_json", RUBY_METHOD_FUNC(core_native_export_graph_ir_json), -1);
  rb_define_singleton_method(
      mGraphIRNative,
      "graph_ir_to_onnx_json",
      RUBY_METHOD_FUNC(graph_ir_native_graph_ir_to_onnx_json),
      3);
  rb_define_singleton_method(
      mGraphIRNative,
      "export_onnx_json",
      RUBY_METHOD_FUNC(graph_ir_native_export_onnx_json),
      6);
}
