#include "graph_ir_native.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#ifdef snprintf
#undef snprintf
#endif

#include <nlohmann/json.hpp>

#include "mlx/export.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

using OrderedJson = nlohmann::ordered_json;

namespace {

// ============================================================================
// Section: Shared Types, Constants, and Lookup Tables
// ============================================================================

using Shape = std::vector<int64_t>;
using ShapeMap = std::map<std::string, Shape>;
using DtypeMap = std::map<std::string, std::string>;
using NameSet = std::set<std::string>;

static VALUE mGraphIR;
static VALUE mGraphIRNative;
static VALUE eGraphIRNativeUnsupportedError = Qnil;

constexpr int64_t kGraphIrVersion = 1;

static constexpr std::array<std::pair<const char*, const char*>, 49> kOnnxOpPairs = {{
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
}};

static constexpr std::array<std::pair<int64_t, const char*>, 6> kReduceCodeToOnnxOpPairs = {{
    {0, "ReduceMin"},
    {1, "ReduceMax"},
    {2, "ReduceSum"},
    {3, "ReduceProd"},
    {4, "ReduceMin"},
    {5, "ReduceMax"},
}};

static constexpr std::array<std::pair<int64_t, const char*>, 2> kArgReduceCodeToOnnxOpPairs = {{
    {0, "ArgMin"},
    {1, "ArgMax"},
}};

static constexpr std::array<std::pair<const char*, const char*>, 15> kOnnxDtypePairs = {{
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
}};

static constexpr std::array<std::pair<const char*, int>, 13> kDtypePromotionRankPairs = {{
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
}};

template <size_t N>
static const char* lookup_string_pair(
    const std::array<std::pair<const char*, const char*>, N>& pairs,
    const std::string& key) {
  for (const auto& [candidate_key, candidate_value] : pairs) {
    if (key == candidate_key) {
      return candidate_value;
    }
  }
  return nullptr;
}

template <size_t N>
static const char* lookup_int_string_pair(
    const std::array<std::pair<int64_t, const char*>, N>& pairs,
    int64_t key) {
  for (const auto& [candidate_key, candidate_value] : pairs) {
    if (candidate_key == key) {
      return candidate_value;
    }
  }
  return nullptr;
}

template <size_t N>
static std::optional<int> lookup_string_int_pair(
    const std::array<std::pair<const char*, int>, N>& pairs,
    const std::string& key) {
  for (const auto& [candidate_key, candidate_value] : pairs) {
    if (key == candidate_key) {
      return candidate_value;
    }
  }
  return std::nullopt;
}

using GraphTensorInfo = std::tuple<std::string, mx::Shape, mx::Dtype>;

struct GraphIrExportInvocation {
  VALUE fun;
  mx::Args args;
  mx::Kwargs kwargs;
  bool shapeless;
};

struct GraphIrExportTimingStats {
  double export_function_ms = 0.0;
  double constants_capture_ms = 0.0;
  size_t constants_count = 0;
  size_t constant_elements = 0;
};

struct OnnxBinaryWriteOptions {
  bool external_data = false;
  std::string external_data_file = "weights.bin";
  int64_t external_data_size_threshold = 1024;
};

struct OnnxBinaryArtifact {
  std::string model_bytes;
  std::string external_data_bytes;
  bool has_external_data = false;
};

// ============================================================================
// Section: Timing and Diagnostics Helpers
// ============================================================================

static bool graph_ir_native_timing_enabled() {
  const char* raw = std::getenv("MLX_GRAPH_IR_NATIVE_TIMING");
  if (raw == nullptr) {
    return false;
  }

  std::string value(raw);
  if (value.empty()) {
    return false;
  }

  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return !(value == "0" || value == "false" || value == "off" || value == "no");
}

static double elapsed_millis(std::chrono::steady_clock::time_point started_at) {
  const auto finished_at = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(finished_at - started_at).count();
}

static void emit_graph_ir_native_timing_line(const std::string& line) {
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

static void emit_export_onnx_json_timing_line(
    const GraphIrExportInvocation& invocation,
    int64_t opset,
    const std::string& model_name,
    const GraphIrExportTimingStats& export_stats,
    double args_decode_ms,
    double export_graph_ir_ms,
    double lower_onnx_ms,
    double dump_json_ms,
    double total_ms,
    size_t onnx_json_bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << "[mlx.graph_ir.native.timing] export_onnx_json";
  out << " total_ms=" << total_ms;
  out << " args_decode_ms=" << args_decode_ms;
  out << " export_graph_ir_ms=" << export_graph_ir_ms;
  out << " trace_export_ms=" << export_stats.export_function_ms;
  out << " constants_capture_ms=" << export_stats.constants_capture_ms;
  out << " constants_count=" << export_stats.constants_count;
  out << " constant_elements=" << export_stats.constant_elements;
  out << " lower_onnx_ms=" << lower_onnx_ms;
  out << " json_dump_ms=" << dump_json_ms;
  out << " onnx_json_bytes=" << onnx_json_bytes;
  out << " shapeless=" << (invocation.shapeless ? "true" : "false");
  out << " opset=" << opset;
  out << " model_name=" << model_name;
  emit_graph_ir_native_timing_line(out.str());
}

static void emit_graph_ir_to_onnx_json_timing_line(
    int64_t opset,
    const std::string& model_name,
    double parse_json_ms,
    double lower_onnx_ms,
    double dump_json_ms,
    double total_ms,
    size_t onnx_json_bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << "[mlx.graph_ir.native.timing] graph_ir_to_onnx_json";
  out << " total_ms=" << total_ms;
  out << " parse_json_ms=" << parse_json_ms;
  out << " lower_onnx_ms=" << lower_onnx_ms;
  out << " json_dump_ms=" << dump_json_ms;
  out << " onnx_json_bytes=" << onnx_json_bytes;
  out << " opset=" << opset;
  out << " model_name=" << model_name;
  emit_graph_ir_native_timing_line(out.str());
}

static std::string dtype_to_string(mx::Dtype dtype) {
  std::ostringstream out;
  out << dtype;
  return out.str();
}

static void ruby_hash_set_cstr(VALUE hash, const char* key, VALUE value) {
  rb_hash_aset(hash, rb_str_new_cstr(key), value);
}

// ============================================================================
// Section: GraphIR Export Capture and Trace Conversion
// ============================================================================

static GraphIrExportInvocation parse_graph_ir_export_invocation(
    int argc,
    VALUE* argv,
    const char* method_name) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "%s expects at least callable", method_name);
  }
  VALUE fun = argv[0];

  bool shapeless = false;
  int end = argc;
  if (argc > 1 && (argv[argc - 1] == Qtrue || argv[argc - 1] == Qfalse)) {
    shapeless = RTEST(argv[argc - 1]);
    end -= 1;
  }

  std::vector<VALUE> extras;
  extras.reserve(static_cast<size_t>(std::max(0, end - 1)));
  for (int i = 1; i < end; ++i) {
    extras.push_back(argv[i]);
  }

  VALUE kwargs_hash = Qnil;
  if (!extras.empty() && RB_TYPE_P(extras.back(), T_HASH)) {
    kwargs_hash = extras.back();
    extras.pop_back();
  }

  mx::Args args;
  if (extras.size() == 1) {
    VALUE item = extras[0];
    if (RB_TYPE_P(item, T_ARRAY)) {
      args = graph_ir_array_vector_from_ruby(item);
    } else {
      args.push_back(graph_ir_array_from_ruby(item));
    }
  } else {
    args.reserve(extras.size());
    for (VALUE item : extras) {
      args.push_back(graph_ir_array_from_ruby(item));
    }
  }

  mx::Kwargs kwargs = NIL_P(kwargs_hash) ? mx::Kwargs{} : graph_ir_array_map_from_ruby_hash(kwargs_hash);
  if (args.empty() && kwargs.empty()) {
    rb_raise(
        rb_eArgError,
        "[%s] Inputs must include at least one positional or keyword array",
        method_name);
  }

  GraphIrExportInvocation invocation;
  invocation.fun = fun;
  invocation.args = std::move(args);
  invocation.kwargs = std::move(kwargs);
  invocation.shapeless = shapeless;
  return invocation;
}

[[noreturn]] static void raise_graph_ir_native_exception(const std::exception& error);

template <typename ValueAt>
static OrderedJson capture_build_nested_json_array(
    const mx::Shape& shape,
    size_t dim,
    size_t& flat_index,
    ValueAt value_at) {
  if (dim == shape.size()) {
    return value_at(flat_index++);
  }

  OrderedJson out = OrderedJson::array();
  for (size_t i = 0; i < shape[dim]; ++i) {
    out.push_back(capture_build_nested_json_array(shape, dim + 1, flat_index, value_at));
  }
  return out;
}

template <typename ValueAt>
static OrderedJson capture_build_flat_json_array(size_t size, ValueAt value_at) {
  OrderedJson out = OrderedJson::array();
  for (size_t i = 0; i < size; ++i) {
    out.push_back(value_at(i));
  }
  return out;
}

static OrderedJson capture_json_shape_from_mx_shape(const mx::Shape& shape) {
  OrderedJson out = OrderedJson::array();
  for (size_t dim : shape) {
    out.push_back(dim);
  }
  return out;
}

static OrderedJson capture_json_scalar_from_array(const mx::array& array) {
  switch (array.dtype()) {
    case mx::bool_:
      return OrderedJson(array.item<bool>());
    case mx::uint8:
      return OrderedJson(array.item<uint8_t>());
    case mx::uint16:
      return OrderedJson(array.item<uint16_t>());
    case mx::uint32:
      return OrderedJson(array.item<uint32_t>());
    case mx::uint64:
      return OrderedJson(array.item<uint64_t>());
    case mx::int8:
      return OrderedJson(array.item<int8_t>());
    case mx::int16:
      return OrderedJson(array.item<int16_t>());
    case mx::int32:
      return OrderedJson(array.item<int32_t>());
    case mx::int64:
      return OrderedJson(array.item<int64_t>());
    case mx::float16:
      return OrderedJson(static_cast<double>(array.item<mx::float16_t>()));
    case mx::bfloat16:
      return OrderedJson(static_cast<double>(array.item<mx::bfloat16_t>()));
    case mx::float32:
      return OrderedJson(static_cast<double>(array.item<float>()));
    case mx::float64:
      return OrderedJson(array.item<double>());
    default:
      throw std::runtime_error("unsupported dtype for graph ir constant conversion");
  }
}

static OrderedJson capture_json_values_from_array(const mx::array& source) {
  mx::array array = source;
  if (array.ndim() == 0) {
    array.eval();
    return capture_json_scalar_from_array(array);
  }

  if (array.ndim() == 1) {
    array.eval();
    const size_t size = array.size();
    switch (array.dtype()) {
      case mx::bool_: {
        const bool* data = array.data<bool>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint8: {
        const uint8_t* data = array.data<uint8_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint16: {
        const uint16_t* data = array.data<uint16_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint32: {
        const uint32_t* data = array.data<uint32_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::uint64: {
        const uint64_t* data = array.data<uint64_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int8: {
        const int8_t* data = array.data<int8_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int16: {
        const int16_t* data = array.data<int16_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int32: {
        const int32_t* data = array.data<int32_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::int64: {
        const int64_t* data = array.data<int64_t>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      case mx::float16: {
        const mx::float16_t* data = array.data<mx::float16_t>();
        return capture_build_flat_json_array(
            size,
            [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
      }
      case mx::bfloat16: {
        const mx::bfloat16_t* data = array.data<mx::bfloat16_t>();
        return capture_build_flat_json_array(
            size,
            [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
      }
      case mx::float32: {
        const float* data = array.data<float>();
        return capture_build_flat_json_array(
            size,
            [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
      }
      case mx::float64: {
        const double* data = array.data<double>();
        return capture_build_flat_json_array(size, [&](size_t i) { return OrderedJson(data[i]); });
      }
      default:
        throw std::runtime_error("unsupported dtype for graph ir constant conversion");
    }
  }

  const mx::Shape shape = array.shape();
  mx::array flat = mx::reshape(array, mx::Shape{static_cast<mx::ShapeElem>(array.size())});
  flat.eval();

  size_t idx = 0;
  switch (flat.dtype()) {
    case mx::bool_: {
      const bool* data = flat.data<bool>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint8: {
      const uint8_t* data = flat.data<uint8_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint16: {
      const uint16_t* data = flat.data<uint16_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint32: {
      const uint32_t* data = flat.data<uint32_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::uint64: {
      const uint64_t* data = flat.data<uint64_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int8: {
      const int8_t* data = flat.data<int8_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int16: {
      const int16_t* data = flat.data<int16_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int32: {
      const int32_t* data = flat.data<int32_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::int64: {
      const int64_t* data = flat.data<int64_t>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    case mx::float16: {
      const mx::float16_t* data = flat.data<mx::float16_t>();
      return capture_build_nested_json_array(
          shape,
          0,
          idx,
          [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
    }
    case mx::bfloat16: {
      const mx::bfloat16_t* data = flat.data<mx::bfloat16_t>();
      return capture_build_nested_json_array(
          shape,
          0,
          idx,
          [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
    }
    case mx::float32: {
      const float* data = flat.data<float>();
      return capture_build_nested_json_array(
          shape,
          0,
          idx,
          [&](size_t i) { return OrderedJson(static_cast<double>(data[i])); });
    }
    case mx::float64: {
      const double* data = flat.data<double>();
      return capture_build_nested_json_array(shape, 0, idx, [&](size_t i) { return OrderedJson(data[i]); });
    }
    default:
      throw std::runtime_error("unsupported dtype for graph ir constant conversion");
  }
}

static OrderedJson capture_json_tensor_info_from_graph_tensor(const GraphTensorInfo& info) {
  OrderedJson out = OrderedJson::object();
  out["name"] = std::get<0>(info);
  out["shape"] = capture_json_shape_from_mx_shape(std::get<1>(info));
  out["dtype"] = dtype_to_string(std::get<2>(info));
  return out;
}

static OrderedJson capture_json_tensor_infos_from_graph_tensors(const std::vector<GraphTensorInfo>& infos) {
  OrderedJson out = OrderedJson::array();
  for (const auto& info : infos) {
    out.push_back(capture_json_tensor_info_from_graph_tensor(info));
  }
  return out;
}

static OrderedJson capture_json_tensor_names_from_graph_tensors(const std::vector<GraphTensorInfo>& infos) {
  OrderedJson out = OrderedJson::array();
  for (const auto& info : infos) {
    out.push_back(std::get<0>(info));
  }
  return out;
}

static OrderedJson capture_json_state_value_from_mx_state(const mx::StateT& value) {
  if (std::holds_alternative<bool>(value)) {
    return OrderedJson(std::get<bool>(value));
  }
  if (std::holds_alternative<int>(value)) {
    return OrderedJson(std::get<int>(value));
  }
  if (std::holds_alternative<size_t>(value)) {
    return OrderedJson(std::get<size_t>(value));
  }
  if (std::holds_alternative<float>(value)) {
    return OrderedJson(static_cast<double>(std::get<float>(value)));
  }
  if (std::holds_alternative<double>(value)) {
    return OrderedJson(std::get<double>(value));
  }
  if (std::holds_alternative<mx::Dtype>(value)) {
    return OrderedJson(dtype_to_string(std::get<mx::Dtype>(value)));
  }
  if (std::holds_alternative<mx::Shape>(value)) {
    return capture_json_shape_from_mx_shape(std::get<mx::Shape>(value));
  }
  if (std::holds_alternative<mx::Strides>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& strides = std::get<mx::Strides>(value);
    for (auto stride : strides) {
      out.push_back(static_cast<long long>(stride));
    }
    return out;
  }
  if (std::holds_alternative<std::vector<int>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& values = std::get<std::vector<int>>(value);
    for (int item : values) {
      out.push_back(item);
    }
    return out;
  }
  if (std::holds_alternative<std::vector<size_t>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& values = std::get<std::vector<size_t>>(value);
    for (size_t item : values) {
      out.push_back(item);
    }
    return out;
  }
  if (std::holds_alternative<std::vector<std::tuple<bool, bool, bool>>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& tuples = std::get<std::vector<std::tuple<bool, bool, bool>>>(value);
    for (const auto& item : tuples) {
      out.push_back(OrderedJson::array({std::get<0>(item), std::get<1>(item), std::get<2>(item)}));
    }
    return out;
  }
  if (std::holds_alternative<std::vector<std::variant<bool, int, float>>>(value)) {
    OrderedJson out = OrderedJson::array();
    const auto& vars = std::get<std::vector<std::variant<bool, int, float>>>(value);
    for (const auto& item : vars) {
      if (std::holds_alternative<bool>(item)) {
        out.push_back(std::get<bool>(item));
      } else if (std::holds_alternative<int>(item)) {
        out.push_back(std::get<int>(item));
      } else {
        out.push_back(static_cast<double>(std::get<float>(item)));
      }
    }
    return out;
  }
  if (std::holds_alternative<std::optional<float>>(value)) {
    const auto& opt = std::get<std::optional<float>>(value);
    if (!opt.has_value()) {
      return nullptr;
    }
    return OrderedJson(static_cast<double>(opt.value()));
  }
  return OrderedJson(std::get<std::string>(value));
}

static OrderedJson capture_json_state_values_from_mx_states(const std::vector<mx::StateT>& values) {
  OrderedJson out = OrderedJson::array();
  for (const auto& value : values) {
    out.push_back(capture_json_state_value_from_mx_state(value));
  }
  return out;
}

template <typename T>
static const T* export_callback_field(
    const mx::ExportCallbackInput& data,
    const std::string& key) {
  for (const auto& [candidate_key, candidate_value] : data) {
    if (candidate_key == key && std::holds_alternative<T>(candidate_value)) {
      return &std::get<T>(candidate_value);
    }
  }
  return nullptr;
}

static OrderedJson export_graph_ir_payload(
    const GraphIrExportInvocation& invocation,
    GraphIrExportTimingStats* timing_stats = nullptr) {
  OrderedJson graph_inputs = OrderedJson::array();
  OrderedJson keyword_inputs = OrderedJson::array();
  OrderedJson graph_outputs = OrderedJson::array();
  OrderedJson graph_constants = OrderedJson::array();
  OrderedJson graph_nodes = OrderedJson::array();

  const auto trace_started_at = std::chrono::steady_clock::now();
  mx::export_function(
      [&graph_inputs, &keyword_inputs, &graph_outputs, &graph_constants, &graph_nodes, timing_stats](
          const mx::ExportCallbackInput& data) {
        const auto* record_type = export_callback_field<std::string>(data, "type");
        if (record_type == nullptr) {
          return;
        }

        if (*record_type == "inputs") {
          const auto* inputs = export_callback_field<std::vector<GraphTensorInfo>>(data, "inputs");
          if (inputs != nullptr) {
            graph_inputs = capture_json_tensor_infos_from_graph_tensors(*inputs);
          }
          return;
        }

        if (*record_type == "keyword_inputs") {
          const auto* keywords =
              export_callback_field<std::vector<std::pair<std::string, std::string>>>(
                  data,
                  "keywords");
          if (keywords != nullptr) {
            keyword_inputs = OrderedJson::array();
            for (const auto& [name, tensor] : *keywords) {
              OrderedJson entry = OrderedJson::object();
              entry["name"] = name;
              entry["tensor"] = tensor;
              keyword_inputs.push_back(std::move(entry));
            }
          }
          return;
        }

        if (*record_type == "outputs") {
          const auto* outputs = export_callback_field<std::vector<GraphTensorInfo>>(data, "outputs");
          if (outputs != nullptr) {
            graph_outputs = capture_json_tensor_infos_from_graph_tensors(*outputs);
          }
          return;
        }

        if (*record_type == "constants") {
          const auto* constants =
              export_callback_field<std::vector<std::pair<std::string, mx::array>>>(
                  data,
                  "constants");
          if (constants != nullptr) {
            graph_constants = OrderedJson::array();
            if (timing_stats != nullptr) {
              timing_stats->constants_count += constants->size();
            }
            for (const auto& [name, arr] : *constants) {
              OrderedJson entry = OrderedJson::object();
              entry["name"] = name;
              entry["shape"] = capture_json_shape_from_mx_shape(arr.shape());
              entry["dtype"] = dtype_to_string(arr.dtype());
              if (timing_stats != nullptr) {
                timing_stats->constant_elements += static_cast<size_t>(arr.size());
                const auto capture_started_at = std::chrono::steady_clock::now();
                entry["values"] = capture_json_values_from_array(arr);
                timing_stats->constants_capture_ms += elapsed_millis(capture_started_at);
              } else {
                entry["values"] = capture_json_values_from_array(arr);
              }
              graph_constants.push_back(std::move(entry));
            }
          }
          return;
        }

        if (*record_type != "primitive") {
          return;
        }

        const auto* op_name = export_callback_field<std::string>(data, "name");
        if (op_name == nullptr) {
          return;
        }

        OrderedJson node = OrderedJson::object();
        node["op"] = *op_name;

        OrderedJson node_inputs = OrderedJson::array();
        const auto* node_input_infos = export_callback_field<std::vector<GraphTensorInfo>>(data, "inputs");
        if (node_input_infos != nullptr) {
          node_inputs = capture_json_tensor_names_from_graph_tensors(*node_input_infos);
        }
        node["inputs"] = std::move(node_inputs);

        OrderedJson node_outputs = OrderedJson::array();
        const auto* node_output_infos =
            export_callback_field<std::vector<GraphTensorInfo>>(data, "outputs");
        if (node_output_infos != nullptr) {
          node_outputs = capture_json_tensor_names_from_graph_tensors(*node_output_infos);
        }
        node["outputs"] = std::move(node_outputs);

        OrderedJson node_arguments = OrderedJson::array();
        const auto* arguments = export_callback_field<std::vector<mx::StateT>>(data, "arguments");
        if (arguments != nullptr) {
          node_arguments = capture_json_state_values_from_mx_states(*arguments);
        }
        node["arguments"] = std::move(node_arguments);

        graph_nodes.push_back(std::move(node));
      },
      graph_ir_args_kwargs_function_from_callable(invocation.fun),
      invocation.args,
      invocation.kwargs,
      invocation.shapeless);
  if (timing_stats != nullptr) {
    timing_stats->export_function_ms += elapsed_millis(trace_started_at);
  }

  OrderedJson payload = OrderedJson::object();
  payload["ir_version"] = 1;
  payload["shapeless"] = invocation.shapeless;
  payload["inputs"] = std::move(graph_inputs);
  payload["keyword_inputs"] = std::move(keyword_inputs);
  payload["outputs"] = std::move(graph_outputs);
  payload["constants"] = std::move(graph_constants);
  payload["nodes"] = std::move(graph_nodes);
  return payload;
}

VALUE core_native_export_graph_ir_json(int argc, VALUE* argv, VALUE) {
  try {
    auto invocation = parse_graph_ir_export_invocation(argc, argv, "native_export_graph_ir_json");
    OrderedJson payload = export_graph_ir_payload(invocation);

    const std::string content = payload.dump();
    return rb_str_new(content.data(), static_cast<long>(content.size()));
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

// ============================================================================
// Section: Ruby <-> OrderedJson Conversion and Source Parsing
// ============================================================================

static std::string std_string_from_ruby(VALUE value) {
  VALUE str = rb_obj_as_string(value);
  return std::string(RSTRING_PTR(str), static_cast<size_t>(RSTRING_LEN(str)));
}

static VALUE ruby_string_from_std(const std::string& value) {
  return rb_str_new(value.data(), static_cast<long>(value.size()));
}

static OrderedJson ordered_json_from_ruby(VALUE value);
static OrderedJson ordered_json_complex_from_ruby(VALUE value);

static OrderedJson ordered_json_integer_from_ruby(VALUE value) {
  VALUE text_value = rb_funcall(value, rb_intern("to_s"), 0);
  const std::string text =
      std::string(RSTRING_PTR(text_value), static_cast<size_t>(RSTRING_LEN(text_value)));
  if (text.empty()) {
    throw std::invalid_argument("failed to convert Integer to JSON number");
  }

  const bool negative = text.front() == '-';
  try {
    if (negative) {
      return static_cast<int64_t>(std::stoll(text));
    }

    const auto raw = std::stoull(text);
    if (raw <= static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
      return static_cast<int64_t>(raw);
    }
    return static_cast<uint64_t>(raw);
  } catch (const std::out_of_range&) {
    try {
      return static_cast<double>(std::stold(text));
    } catch (const std::exception&) {
      throw std::invalid_argument("Integer is too large to convert into JSON numeric range");
    }
  } catch (const std::invalid_argument&) {
    throw std::invalid_argument("failed to parse Integer while converting to JSON");
  }
}

static OrderedJson ordered_json_object_from_ruby_hash(VALUE hash) {
  OrderedJson out = OrderedJson::object();
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  const long len = RARRAY_LEN(keys);
  for (long i = 0; i < len; ++i) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE item = rb_hash_aref(hash, key);
    VALUE key_str = rb_obj_as_string(key);
    out[std::string(RSTRING_PTR(key_str), static_cast<size_t>(RSTRING_LEN(key_str)))] =
        ordered_json_from_ruby(item);
  }
  return out;
}

static OrderedJson ordered_json_complex_from_ruby(VALUE value) {
  VALUE real_value = rb_funcall(value, rb_intern("real"), 0);
  VALUE imag_value = rb_funcall(value, rb_intern("imag"), 0);
  double real = NUM2DBL(real_value);
  double imag = NUM2DBL(imag_value);
  OrderedJson pair = OrderedJson::array();
  pair.push_back(real);
  pair.push_back(imag);
  OrderedJson out = OrderedJson::object();
  out["__mlx_complex__"] = std::move(pair);
  return out;
}

static OrderedJson ordered_json_from_ruby(VALUE value) {
  if (NIL_P(value)) {
    return nullptr;
  }
  if (value == Qtrue || value == Qfalse) {
    return value == Qtrue;
  }
  if (rb_obj_is_kind_of(value, rb_cComplex)) {
    return ordered_json_complex_from_ruby(value);
  }
  if (RB_INTEGER_TYPE_P(value)) {
    return ordered_json_integer_from_ruby(value);
  }
  if (RB_FLOAT_TYPE_P(value)) {
    return NUM2DBL(value);
  }
  if (RB_TYPE_P(value, T_STRING)) {
    return std::string(RSTRING_PTR(value), static_cast<size_t>(RSTRING_LEN(value)));
  }
  if (RB_TYPE_P(value, T_ARRAY)) {
    OrderedJson out = OrderedJson::array();
    const long len = RARRAY_LEN(value);
    for (long i = 0; i < len; ++i) {
      out.push_back(ordered_json_from_ruby(rb_ary_entry(value, i)));
    }
    return out;
  }
  if (RB_TYPE_P(value, T_HASH)) {
    return ordered_json_object_from_ruby_hash(value);
  }
  if (SYMBOL_P(value)) {
    VALUE symbol_str = rb_sym2str(value);
    return std::string(RSTRING_PTR(symbol_str), static_cast<size_t>(RSTRING_LEN(symbol_str)));
  }

  VALUE converted = rb_obj_as_string(value);
  return std::string(RSTRING_PTR(converted), static_cast<size_t>(RSTRING_LEN(converted)));
}

static VALUE ruby_value_from_ordered_json(const OrderedJson& value) {
  if (value.is_null()) {
    return Qnil;
  }
  if (value.is_boolean()) {
    return value.get<bool>() ? Qtrue : Qfalse;
  }
  if (value.is_number_integer()) {
    return LL2NUM(value.get<int64_t>());
  }
  if (value.is_number_unsigned()) {
    return ULL2NUM(value.get<uint64_t>());
  }
  if (value.is_number_float()) {
    return rb_float_new(value.get<double>());
  }
  if (value.is_string()) {
    const auto& text = value.get_ref<const std::string&>();
    return rb_str_new(text.data(), static_cast<long>(text.size()));
  }
  if (value.is_array()) {
    VALUE out = rb_ary_new_capa(static_cast<long>(value.size()));
    for (const auto& item : value) {
      rb_ary_push(out, ruby_value_from_ordered_json(item));
    }
    return out;
  }
  if (value.is_object()) {
    VALUE out = rb_hash_new();
    for (auto it = value.begin(); it != value.end(); ++it) {
      const auto& key = it.key();
      rb_hash_aset(
          out,
          rb_str_new(key.data(), static_cast<long>(key.size())),
          ruby_value_from_ordered_json(it.value()));
    }
    return out;
  }

  rb_raise(rb_eTypeError, "unsupported JSON value type");
  return Qnil;
}

static std::string read_file_to_string(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::ostringstream out;
    out << "failed to read file: " << path;
    throw std::runtime_error(out.str());
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

static OrderedJson parse_json_payload_from_string(const std::string& raw, const std::string& label) {
  try {
    return OrderedJson::parse(raw);
  } catch (const std::exception& error) {
    std::ostringstream out;
    out << "failed to parse " << label << ": " << error.what();
    throw std::invalid_argument(out.str());
  }
}

static OrderedJson parse_graph_ir_source_payload(VALUE source) {
  if (RB_TYPE_P(source, T_HASH)) {
    return ordered_json_from_ruby(source);
  }

  if (RB_TYPE_P(source, T_STRING)) {
    const std::string raw = std_string_from_ruby(source);
    bool treat_as_file = false;
    try {
      treat_as_file = std::filesystem::is_regular_file(raw);
    } catch (const std::filesystem::filesystem_error&) {
      treat_as_file = false;
    }
    if (treat_as_file) {
      return parse_json_payload_from_string(read_file_to_string(raw), "graph ir file");
    }
    return parse_json_payload_from_string(raw, "graph ir string");
  }

  if (rb_respond_to(source, rb_intern("to_path"))) {
    VALUE path_value = rb_funcall(source, rb_intern("to_path"), 0);
    const std::string path = std_string_from_ruby(path_value);
    if (path.empty()) {
      throw std::invalid_argument("graph ir path-like source must be non-empty");
    }
    if (!std::filesystem::is_regular_file(path)) {
      std::ostringstream out;
      out << "graph ir path does not exist: " << path;
      throw std::invalid_argument(out.str());
    }
    return parse_json_payload_from_string(read_file_to_string(path), "graph ir file");
  }

  if (rb_respond_to(source, rb_intern("read"))) {
    VALUE io_raw = rb_funcall(source, rb_intern("read"), 0);
    return parse_json_payload_from_string(std_string_from_ruby(io_raw), "graph ir IO");
  }

  throw std::invalid_argument(
      "graph ir source must be a Hash, JSON String, file path, or IO-like object");
}

static std::string ruby_path_string(VALUE value, const char* label) {
  if (rb_respond_to(value, rb_intern("write"))) {
    std::ostringstream out;
    out << label << " requires a path-like target, not an IO-like target";
    throw std::invalid_argument(out.str());
  }

  VALUE raw = value;
  if (rb_respond_to(value, rb_intern("to_path"))) {
    raw = rb_funcall(value, rb_intern("to_path"), 0);
  }
  const std::string path = std_string_from_ruby(raw);
  if (path.empty()) {
    std::ostringstream out;
    out << label << " target must be a non-empty path-like value";
    throw std::invalid_argument(out.str());
  }
  return path;
}

// ============================================================================
// Section: ONNX Lowering Utilities and Shape/Dtype Inference
// ============================================================================

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

static std::string onnx_dtype_symbol(const std::string& dtype) {
  const char* symbol = lookup_string_pair(kOnnxDtypePairs, dtype);
  if (symbol != nullptr) {
    return std::string(symbol);
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported dtype " << dtype;
  throw std::runtime_error(out.str());
}

static std::string onnx_op_name(const std::string& op) {
  const char* mapped = lookup_string_pair(kOnnxOpPairs, op);
  if (mapped != nullptr) {
    return std::string(mapped);
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

  const char* mapped = lookup_int_string_pair(kReduceCodeToOnnxOpPairs, reduce_code);
  if (mapped != nullptr) {
    return std::string(mapped);
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

  const char* mapped = lookup_int_string_pair(kArgReduceCodeToOnnxOpPairs, reduce_code);
  if (mapped != nullptr) {
    return std::string(mapped);
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

  const auto lhs_rank = lookup_string_int_pair(kDtypePromotionRankPairs, lhs.value());
  const auto rhs_rank = lookup_string_int_pair(kDtypePromotionRankPairs, rhs.value());
  if (!lhs_rank.has_value() || !rhs_rank.has_value()) {
    return lhs;
  }

  return lhs_rank.value() >= rhs_rank.value() ? lhs : rhs;
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
    const bool valid_dtype =
        target.is_string() &&
        lookup_string_pair(kOnnxDtypePairs, target.get<std::string>()) != nullptr;
    if (!valid_dtype) {
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
  if (lookup_int_string_pair(kArgReduceCodeToOnnxOpPairs, mode) == nullptr) {
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

  std::set<size_t> index_filter;
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

  const char* mapped = lookup_string_pair(kOnnxOpPairs, op);
  if (mapped != nullptr) {
    return std::string(mapped);
  }

  if (!strict) {
    return std::nullopt;
  }

  std::ostringstream out;
  out << "[graph_ir_to_onnx_stub] unsupported op " << op;
  throw std::runtime_error(out.str());
}

struct LoweringContext {
  OrderedJson& initializers;
  NameSet& used_tensor_names;
  ShapeMap& known_shapes;
  DtypeMap& known_dtypes;
};

struct ParsedLoweringNode {
  std::string op;
  std::string op_type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  OrderedJson attributes;
  OrderedJson arguments;
};

static ParsedLoweringNode parse_lowering_node(
    const OrderedJson& node,
    const ShapeMap& known_shapes) {
  ParsedLoweringNode parsed;
  parsed.op = node.at("op").get<std::string>();
  parsed.op_type = onnx_op_type_for_node(node, true, &known_shapes).value();
  parsed.inputs = json_string_vector(node.at("inputs"), "node inputs");
  parsed.outputs = json_string_vector(node.at("outputs"), "node outputs");
  parsed.attributes = onnx_node_attributes(node);
  parsed.arguments = node.contains("arguments") ? node.at("arguments") : OrderedJson::array();
  return parsed;
}

static void assign_known_shape_if_present(
    ShapeMap& known_shapes,
    const std::vector<std::string>& names,
    const std::optional<Shape>& shape) {
  if (!shape.has_value()) {
    return;
  }
  for (const auto& name : names) {
    known_shapes[name] = shape.value();
  }
}

static void assign_known_dtype_if_present(
    DtypeMap& known_dtypes,
    const std::vector<std::string>& names,
    const std::optional<std::string>& dtype) {
  if (!dtype.has_value()) {
    return;
  }
  for (const auto& name : names) {
    known_dtypes[name] = dtype.value();
  }
}

static std::optional<std::vector<OrderedJson>> maybe_lower_with_promoted_cast(
    size_t node_index,
    const std::string& op,
    const std::string& op_type,
    std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const OrderedJson& attributes,
    const std::optional<std::string>& promoted_dtype,
    const std::optional<std::vector<size_t>>& cast_indices,
    size_t lhs_input_index,
    size_t rhs_input_index,
    const std::optional<std::string>& output_dtype_override,
    LoweringContext& lowering) {
  if (!promoted_dtype.has_value()) {
    return std::nullopt;
  }

  auto [casted_inputs, cast_nodes] = cast_inputs_to_dtype(
      node_index,
      op,
      inputs,
      promoted_dtype.value(),
      lowering.known_shapes,
      lowering.known_dtypes,
      lowering.used_tensor_names,
      cast_indices);
  if (cast_nodes.empty()) {
    return std::nullopt;
  }

  inputs = casted_inputs;
  const auto inferred_output_shape = infer_elementwise_output_shape(
      known_shape_for(lowering.known_shapes, inputs.at(lhs_input_index)),
      known_shape_for(lowering.known_shapes, inputs.at(rhs_input_index)));
  const auto inferred_output_dtype = output_dtype_override.has_value()
      ? output_dtype_override
      : promoted_dtype;
  assign_known_shape_if_present(lowering.known_shapes, outputs, inferred_output_shape);
  assign_known_dtype_if_present(lowering.known_dtypes, outputs, inferred_output_dtype);

  cast_nodes.push_back(build_onnx_node_spec(
      "node_" + std::to_string(node_index) + "_" + op_type,
      op_type,
      inputs,
      outputs,
      attributes));
  return cast_nodes;
}

static std::vector<OrderedJson> lower_onnx_arange_node(
    const std::vector<std::string>& outputs,
    const OrderedJson& arguments,
    LoweringContext& lowering) {
  const auto parsed = arange_arguments(arguments);
  const auto values = arange_values(parsed);

  OrderedJson tensor = OrderedJson::object();
  tensor["name"] = outputs.at(0);
  tensor["shape"] = json_from_int_vector({static_cast<int64_t>(values.size())});
  tensor["dtype"] = parsed.dtype;
  tensor["values"] = values;

  lowering.initializers.push_back(onnx_initializer_info(tensor));
  lowering.known_shapes[outputs.at(0)] = {static_cast<int64_t>(values.size())};
  lowering.known_dtypes[outputs.at(0)] = parsed.dtype;
  return {};
}

static std::vector<OrderedJson> lower_onnx_convolution_node(
    size_t node_index,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const OrderedJson& arguments,
    LoweringContext& lowering) {
  auto& used_tensor_names = lowering.used_tensor_names;
  auto& known_shapes = lowering.known_shapes;
  auto& known_dtypes = lowering.known_dtypes;
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
    auto inferred_output_shape = infer_convolution_transpose_output_shape(
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

  auto inferred_output_shape = infer_convolution_output_shape(
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

static std::vector<OrderedJson> lower_onnx_node_default(
    const OrderedJson& node,
    size_t node_index,
    LoweringContext& lowering) {
  auto& initializers = lowering.initializers;
  auto& used_tensor_names = lowering.used_tensor_names;
  auto& known_shapes = lowering.known_shapes;
  auto& known_dtypes = lowering.known_dtypes;

  auto parsed_node = parse_lowering_node(node, known_shapes);
  const auto op = std::move(parsed_node.op);
  const auto op_type = std::move(parsed_node.op_type);
  auto inputs = std::move(parsed_node.inputs);
  auto outputs = std::move(parsed_node.outputs);
  OrderedJson attributes = std::move(parsed_node.attributes);
  const OrderedJson arguments = std::move(parsed_node.arguments);

  std::optional<Shape> inferred_output_shape;
  std::optional<std::string> inferred_output_dtype;

  if (op == "Arange") {
    return lower_onnx_arange_node(outputs, arguments, lowering);
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
    return lower_onnx_convolution_node(
        node_index,
        inputs,
        outputs,
        arguments,
        lowering);
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
      const char* reduce_type_symbol = lookup_int_string_pair(kReduceCodeToOnnxOpPairs, reduce_code);
      if (reduce_type_symbol == nullptr) {
        std::ostringstream out;
        out << "[graph_ir_to_onnx_stub] unsupported Reduce code " << reduce_code;
        throw std::runtime_error(out.str());
      }
      const std::string reduce_type(reduce_type_symbol);

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

    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::nullopt,
            0,
            1,
            std::nullopt,
            lowering);
        lowered.has_value()) {
      return lowered.value();
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
    const char* arg_op_symbol = lookup_int_string_pair(kArgReduceCodeToOnnxOpPairs, arg_mode);
    if (arg_op_symbol == nullptr) {
      std::ostringstream out;
      out << "[graph_ir_to_onnx_stub] unsupported ArgReduce code " << arg_mode;
      throw std::runtime_error(out.str());
    }
    const std::string arg_op(arg_op_symbol);
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
    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::nullopt,
            0,
            1,
            std::optional<std::string>("bool"),
            lowering);
        lowered.has_value()) {
      return lowered.value();
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

    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::nullopt,
            0,
            1,
            std::optional<std::string>("bool"),
            lowering);
        lowered.has_value()) {
      return lowered.value();
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

    if (const auto lowered = maybe_lower_with_promoted_cast(
            node_index,
            op,
            op_type,
            inputs,
            outputs,
            attributes,
            promoted_dtype,
            std::optional<std::vector<size_t>>{{1, 2}},
            1,
            2,
            std::nullopt,
            lowering);
        lowered.has_value()) {
      return lowered.value();
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

  assign_known_shape_if_present(known_shapes, outputs, inferred_output_shape);
  assign_known_dtype_if_present(known_dtypes, outputs, inferred_output_dtype);

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
  LoweringContext lowering{
      initializers,
      used_tensor_names,
      known_shapes,
      known_dtypes};

  OrderedJson nodes = OrderedJson::array();
  const auto& source_nodes = payload.at("nodes");
  for (size_t index = 0; index < source_nodes.size(); ++index) {
    auto lowered = lower_onnx_node_default(
        source_nodes.at(index),
        index,
        lowering);
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

// ============================================================================
// Section: ONNX Protobuf Encoding
// ============================================================================

enum class PbWireType : uint8_t {
  kVarint = 0,
  kFixed64 = 1,
  kLengthDelimited = 2,
  kFixed32 = 5,
};

static void pb_write_varint(std::string& out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7fU) | 0x80U));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value));
}

static void pb_write_key(std::string& out, int field_number, PbWireType wire_type) {
  const uint64_t key =
      (static_cast<uint64_t>(field_number) << 3) | static_cast<uint64_t>(wire_type);
  pb_write_varint(out, key);
}

static void pb_write_varint_field(std::string& out, int field_number, uint64_t value) {
  pb_write_key(out, field_number, PbWireType::kVarint);
  pb_write_varint(out, value);
}

static void pb_write_int64_field(std::string& out, int field_number, int64_t value) {
  pb_write_varint_field(out, field_number, static_cast<uint64_t>(value));
}

static void pb_write_string_field(std::string& out, int field_number, const std::string& value) {
  pb_write_key(out, field_number, PbWireType::kLengthDelimited);
  pb_write_varint(out, static_cast<uint64_t>(value.size()));
  out.append(value);
}

static void pb_write_bytes_field(std::string& out, int field_number, const std::string& value) {
  pb_write_string_field(out, field_number, value);
}

static void pb_write_message_field(std::string& out, int field_number, const std::string& message) {
  pb_write_key(out, field_number, PbWireType::kLengthDelimited);
  pb_write_varint(out, static_cast<uint64_t>(message.size()));
  out.append(message);
}

static void pb_write_fixed32_field(std::string& out, int field_number, uint32_t value) {
  pb_write_key(out, field_number, PbWireType::kFixed32);
  std::array<char, 4> bytes = {
      static_cast<char>(value & 0xffU),
      static_cast<char>((value >> 8) & 0xffU),
      static_cast<char>((value >> 16) & 0xffU),
      static_cast<char>((value >> 24) & 0xffU)};
  out.append(bytes.data(), bytes.size());
}

static int onnx_elem_type_from_symbol(const std::string& symbol) {
  static const std::map<std::string, int> kSymbolToOnnxElemType = {
      {"UNDEFINED", 0},
      {"FLOAT", 1},
      {"UINT8", 2},
      {"INT8", 3},
      {"UINT16", 4},
      {"INT16", 5},
      {"INT32", 6},
      {"INT64", 7},
      {"STRING", 8},
      {"BOOL", 9},
      {"FLOAT16", 10},
      {"DOUBLE", 11},
      {"UINT32", 12},
      {"UINT64", 13},
      {"COMPLEX64", 14},
      {"COMPLEX128", 15},
      {"BFLOAT16", 16},
  };
  const auto it = kSymbolToOnnxElemType.find(symbol);
  if (it != kSymbolToOnnxElemType.end()) {
    return it->second;
  }
  std::ostringstream out;
  out << "unsupported ONNX element type symbol: " << symbol;
  throw std::invalid_argument(out.str());
}

static int onnx_elem_type_from_dtype(const std::string& dtype) {
  return onnx_elem_type_from_symbol(onnx_dtype_symbol(onnx_effective_dtype(dtype)));
}

static bool json_integer_like(const OrderedJson& value) {
  if (value.is_number_integer() || value.is_number_unsigned()) {
    return true;
  }
  if (value.is_number_float()) {
    const double v = value.get<double>();
    return std::isfinite(v) && std::trunc(v) == v;
  }
  return false;
}

static std::vector<int64_t> shape_vector_from_json(const OrderedJson& shape, const std::string& label) {
  if (!shape.is_array()) {
    std::ostringstream out;
    out << label << " shape must be an Array";
    throw std::invalid_argument(out.str());
  }
  std::vector<int64_t> out;
  out.reserve(shape.size());
  for (const auto& dim : shape) {
    out.push_back(normalized_integer_scalar(dim, label + " shape dim"));
  }
  return out;
}

static size_t expected_initializer_value_count(const std::vector<int64_t>& dims) {
  if (dims.empty()) {
    return 1;
  }
  size_t total = 1;
  for (const auto dim : dims) {
    if (dim < 0) {
      throw std::invalid_argument("initializer shape values must be non-negative");
    }
    total *= static_cast<size_t>(dim);
  }
  return total;
}

static void collect_initializer_leaves(const OrderedJson& value, std::vector<const OrderedJson*>& out) {
  if (value.is_array()) {
    for (const auto& item : value) {
      collect_initializer_leaves(item, out);
    }
    return;
  }
  out.push_back(&value);
}

static uint16_t float32_to_float16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint32_t half_mantissa = mantissa >> shift;
    if ((mantissa >> (shift - 1)) & 1U) {
      half_mantissa += 1U;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }

  if (exponent >= 0x1f) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }

  uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
  if (mantissa & 0x00001000U) {
    half = static_cast<uint16_t>(half + 1U);
  }
  return half;
}

static uint16_t float32_to_bfloat16_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x00008000U) >> 16);
}

template <typename T>
static void append_le_bytes(std::string& out, T value) {
  std::array<char, sizeof(T)> bytes{};
  std::memcpy(bytes.data(), &value, sizeof(T));
  out.append(bytes.data(), bytes.size());
}

static void raise_invalid_complex_literal(const std::string& label) {
  std::ostringstream out;
  out << label << " unsupported complex literal";
  throw std::invalid_argument(out.str());
}

static std::string normalize_complex_literal(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const char ch : value) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      normalized.push_back(ch);
    }
  }
  return normalized;
}

static std::size_t find_complex_literal_split(std::string_view value) {
  for (std::size_t idx = value.size(); idx > 0; --idx) {
    const std::size_t pos = idx - 1;
    if (pos == 0) {
      continue;
    }
    const char ch = value[pos];
    if ((ch == '+' || ch == '-') && value[pos - 1] != 'e' && value[pos - 1] != 'E') {
      return pos;
    }
  }
  return std::string_view::npos;
}

static double parse_complex_literal_double(const std::string& text, const std::string& label) {
  if (text.empty()) {
    raise_invalid_complex_literal(label);
  }
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(text.c_str(), &end);
  if (text.c_str() == end) {
    raise_invalid_complex_literal(label);
  }
  if (errno == ERANGE) {
    raise_invalid_complex_literal(label);
  }
  if (static_cast<size_t>(end - text.c_str()) != text.size()) {
    raise_invalid_complex_literal(label);
  }
  return value;
}

static std::pair<float, float> complex64_pair_from_string(const std::string& raw, const std::string& label) {
  const std::string normalized = normalize_complex_literal(raw);
  if (normalized.empty()) {
    raise_invalid_complex_literal(label);
  }
  const char last = normalized.back();
  if (last != 'i' && last != 'I') {
    raise_invalid_complex_literal(label);
  }
  std::string_view remaining(normalized.data(), normalized.size() - 1);
  if (remaining.empty()) {
    return {0.0f, 1.0f};
  }
  const std::size_t split = find_complex_literal_split(remaining);
  std::string real_text;
  std::string imag_text;
  if (split == std::string_view::npos) {
    real_text = "0";
    if (remaining == "+" || remaining == "-") {
      imag_text = remaining == "+" ? "1" : "-1";
    } else {
      imag_text = std::string(remaining);
    }
  } else {
    real_text = std::string(remaining.substr(0, split));
    imag_text = std::string(remaining.substr(split));
  }
  const double real = parse_complex_literal_double(real_text, label);
  const double imag = parse_complex_literal_double(imag_text, label);
  return {static_cast<float>(real), static_cast<float>(imag)};
}

static std::pair<float, float> complex64_pair_from_json(const OrderedJson& value, const std::string& label) {
  if (value.is_object() && value.contains("__mlx_complex__")) {
    const auto& pair = value.at("__mlx_complex__");
    if (!pair.is_array() || pair.size() != 2 || !json_is_numeric(pair.at(0)) || !json_is_numeric(pair.at(1))) {
      std::ostringstream out;
      out << label << " invalid complex marker";
      throw std::invalid_argument(out.str());
    }
    return {static_cast<float>(pair.at(0).get<double>()), static_cast<float>(pair.at(1).get<double>())};
  }
  if (value.is_string()) {
    return complex64_pair_from_string(value.get_ref<const std::string&>(), label);
  }
  if (value.is_boolean()) {
    return {value.get<bool>() ? 1.0f : 0.0f, 0.0f};
  }
  if (json_is_numeric(value)) {
    return {static_cast<float>(value.get<double>()), 0.0f};
  }
  std::ostringstream out;
  out << label << " unsupported complex64 initializer leaf";
  throw std::invalid_argument(out.str());
}

template <typename IntegerType>
static std::string tensor_raw_integer_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected,
    const std::string& label) {
  std::string raw;
  raw.reserve(expected * sizeof(IntegerType));
  for (const auto* item : leaves) {
    append_le_bytes<IntegerType>(raw, static_cast<IntegerType>(normalized_integer_scalar(*item, label)));
  }
  return raw;
}

static double numeric_initializer_leaf(const OrderedJson& value, const std::string& numeric_error_message) {
  if (!json_is_numeric(value)) {
    throw std::invalid_argument(numeric_error_message);
  }
  return value.get<double>();
}

template <typename FloatType>
static std::string tensor_raw_float_initializer(
    const std::vector<const OrderedJson*>& leaves,
    size_t expected,
    const std::string& numeric_error_message) {
  std::string raw;
  raw.reserve(expected * sizeof(FloatType));
  for (const auto* item : leaves) {
    append_le_bytes<FloatType>(raw, static_cast<FloatType>(numeric_initializer_leaf(*item, numeric_error_message)));
  }
  return raw;
}

static std::string tensor_raw_bool_initializer(const std::vector<const OrderedJson*>& leaves, size_t expected) {
  std::string raw;
  raw.reserve(expected);
  for (const auto* item : leaves) {
    uint8_t value = 0;
    if (item->is_boolean()) {
      value = item->get<bool>() ? 1 : 0;
    } else if (json_integer_like(*item)) {
      value = normalized_integer_scalar(*item, "bool initializer leaf") == 0 ? 0 : 1;
    } else if (item->is_number_float()) {
      value = item->get<double>() == 0.0 ? 0 : 1;
    } else {
      throw std::invalid_argument("bool initializer values must be numeric/boolean");
    }
    raw.push_back(static_cast<char>(value));
  }
  return raw;
}

static std::string tensor_raw_float16_initializer(const std::vector<const OrderedJson*>& leaves, size_t expected) {
  std::string raw;
  raw.reserve(expected * sizeof(uint16_t));
  for (const auto* item : leaves) {
    const float value = static_cast<float>(numeric_initializer_leaf(*item, "float16 initializer values must be numeric"));
    append_le_bytes<uint16_t>(raw, float32_to_float16_bits(value));
  }
  return raw;
}

static std::string tensor_raw_bfloat16_initializer(const std::vector<const OrderedJson*>& leaves, size_t expected) {
  std::string raw;
  raw.reserve(expected * sizeof(uint16_t));
  for (const auto* item : leaves) {
    const float value = static_cast<float>(numeric_initializer_leaf(*item, "bfloat16 initializer values must be numeric"));
    append_le_bytes<uint16_t>(raw, float32_to_bfloat16_bits(value));
  }
  return raw;
}

static std::string tensor_raw_complex64_initializer(const std::vector<const OrderedJson*>& leaves, size_t expected) {
  std::string raw;
  raw.reserve(expected * sizeof(float) * 2);
  for (const auto* item : leaves) {
    auto [real, imag] = complex64_pair_from_json(*item, "complex64 initializer leaf");
    append_le_bytes<float>(raw, real);
    append_le_bytes<float>(raw, imag);
  }
  return raw;
}

static std::string tensor_raw_bytes_from_initializer(const OrderedJson& initializer) {
  const std::string dtype = initializer.at("dtype").get<std::string>();
  const auto dims = shape_vector_from_json(initializer.at("shape"), "initializer");
  const size_t expected = expected_initializer_value_count(dims);

  std::vector<const OrderedJson*> leaves;
  collect_initializer_leaves(initializer.at("values"), leaves);
  if (leaves.size() != expected) {
    std::ostringstream out;
    out << "initializer " << initializer.at("name").get<std::string>() << " has " << leaves.size()
        << " values but expected " << expected;
    throw std::invalid_argument(out.str());
  }

  if (dtype == "bool" || dtype == "bool_") {
    return tensor_raw_bool_initializer(leaves, expected);
  }

  if (dtype == "uint8") {
    return tensor_raw_integer_initializer<uint8_t>(leaves, expected, "uint8 initializer leaf");
  }
  if (dtype == "uint16") {
    return tensor_raw_integer_initializer<uint16_t>(leaves, expected, "uint16 initializer leaf");
  }
  if (dtype == "uint32") {
    return tensor_raw_integer_initializer<uint32_t>(leaves, expected, "uint32 initializer leaf");
  }
  if (dtype == "uint64") {
    return tensor_raw_integer_initializer<uint64_t>(leaves, expected, "uint64 initializer leaf");
  }
  if (dtype == "int8") {
    return tensor_raw_integer_initializer<int8_t>(leaves, expected, "int8 initializer leaf");
  }
  if (dtype == "int16") {
    return tensor_raw_integer_initializer<int16_t>(leaves, expected, "int16 initializer leaf");
  }
  if (dtype == "int32") {
    return tensor_raw_integer_initializer<int32_t>(leaves, expected, "int32 initializer leaf");
  }
  if (dtype == "int64") {
    return tensor_raw_integer_initializer<int64_t>(leaves, expected, "int64 initializer leaf");
  }
  if (dtype == "float16") {
    return tensor_raw_float16_initializer(leaves, expected);
  }
  if (dtype == "bfloat16") {
    return tensor_raw_bfloat16_initializer(leaves, expected);
  }
  if (dtype == "float32") {
    return tensor_raw_float_initializer<float>(leaves, expected, "float32 initializer values must be numeric");
  }
  if (dtype == "float64") {
    return tensor_raw_float_initializer<double>(leaves, expected, "float64 initializer values must be numeric");
  }
  if (dtype == "complex64") {
    return tensor_raw_complex64_initializer(leaves, expected);
  }

  std::ostringstream out;
  out << "unsupported initializer dtype for native ONNX binary export: " << dtype;
  throw std::invalid_argument(out.str());
}

static std::string pb_encode_string_string_entry(const std::string& key, const std::string& value) {
  std::string out;
  pb_write_string_field(out, 1, key);
  pb_write_string_field(out, 2, value);
  return out;
}

static std::string pb_encode_tensor_shape(const std::vector<int64_t>& shape) {
  std::string out;
  for (const auto dim : shape) {
    std::string dim_msg;
    pb_write_int64_field(dim_msg, 1, dim);
    pb_write_message_field(out, 1, dim_msg);
  }
  return out;
}

static std::string pb_encode_tensor_type_proto(int elem_type, const std::vector<int64_t>& shape) {
  std::string tensor_type;
  pb_write_varint_field(tensor_type, 1, static_cast<uint64_t>(elem_type));
  pb_write_message_field(tensor_type, 2, pb_encode_tensor_shape(shape));

  std::string type_proto;
  pb_write_message_field(type_proto, 1, tensor_type);
  return type_proto;
}

static std::string pb_encode_value_info(const OrderedJson& info) {
  const std::string name = info.at("name").get<std::string>();
  const auto shape = shape_vector_from_json(info.at("shape"), "value_info");
  std::string onnx_elem_symbol;
  if (info.contains("onnx_elem_type") && info.at("onnx_elem_type").is_string()) {
    onnx_elem_symbol = info.at("onnx_elem_type").get<std::string>();
  } else {
    onnx_elem_symbol = onnx_dtype_symbol(onnx_effective_dtype(info.at("dtype").get<std::string>()));
  }
  const int elem_type = onnx_elem_type_from_symbol(onnx_elem_symbol);

  std::string out;
  pb_write_string_field(out, 1, name);
  pb_write_message_field(out, 2, pb_encode_tensor_type_proto(elem_type, shape));
  return out;
}

static std::string pb_encode_attribute(
    const std::string& op_type,
    const std::string& name,
    const OrderedJson& value) {
  std::string out;
  pb_write_string_field(out, 1, name);

  if (op_type == "Cast" && name == "to" && value.is_string()) {
    const int cast_to = onnx_elem_type_from_symbol(value.get<std::string>());
    pb_write_varint_field(out, 20, 2);
    pb_write_int64_field(out, 3, cast_to);
    return out;
  }

  if (value.is_boolean()) {
    pb_write_varint_field(out, 20, 2);
    pb_write_int64_field(out, 3, value.get<bool>() ? 1 : 0);
    return out;
  }
  if (value.is_number_integer() || value.is_number_unsigned()) {
    pb_write_varint_field(out, 20, 2);
    pb_write_int64_field(out, 3, normalized_integer_scalar(value, "attribute"));
    return out;
  }
  if (value.is_number_float()) {
    pb_write_varint_field(out, 20, 1);
    pb_write_fixed32_field(out, 2, std::bit_cast<uint32_t>(static_cast<float>(value.get<double>())));
    return out;
  }
  if (value.is_string()) {
    pb_write_varint_field(out, 20, 3);
    pb_write_bytes_field(out, 4, value.get<std::string>());
    return out;
  }
  if (value.is_array()) {
    bool all_integer_typed = true;
    bool all_numeric = true;
    bool all_string = true;
    for (const auto& item : value) {
      all_integer_typed =
          all_integer_typed &&
          (item.is_boolean() || item.is_number_integer() || item.is_number_unsigned());
      all_numeric = all_numeric && json_is_numeric(item);
      all_string = all_string && item.is_string();
    }
    if (value.empty() || all_integer_typed) {
      pb_write_varint_field(out, 20, 7);
      for (const auto& item : value) {
        pb_write_int64_field(out, 8, normalized_integer_scalar(item, "attribute vector"));
      }
      return out;
    }
    if (all_numeric) {
      pb_write_varint_field(out, 20, 6);
      for (const auto& item : value) {
        pb_write_fixed32_field(out, 7, std::bit_cast<uint32_t>(static_cast<float>(item.get<double>())));
      }
      return out;
    }
    if (all_string) {
      pb_write_varint_field(out, 20, 8);
      for (const auto& item : value) {
        pb_write_bytes_field(out, 9, item.get<std::string>());
      }
      return out;
    }
  }

  std::ostringstream msg;
  msg << "unsupported ONNX attribute type for " << op_type << "." << name;
  throw std::invalid_argument(msg.str());
}

static std::string pb_encode_node(const OrderedJson& node) {
  const auto op_type = node.at("op_type").get<std::string>();
  std::string out;
  for (const auto& input : node.at("inputs")) {
    pb_write_string_field(out, 1, input.get<std::string>());
  }
  for (const auto& output : node.at("outputs")) {
    pb_write_string_field(out, 2, output.get<std::string>());
  }
  pb_write_string_field(out, 3, node.at("name").get<std::string>());
  pb_write_string_field(out, 4, op_type);

  const auto& attributes = node.at("attributes");
  for (auto it = attributes.begin(); it != attributes.end(); ++it) {
    pb_write_message_field(out, 5, pb_encode_attribute(op_type, it.key(), it.value()));
  }
  return out;
}

static void pb_encode_tensor_header(
    std::string& out,
    const std::vector<int64_t>& shape,
    int elem_type,
    const std::string& name) {
  for (const auto dim : shape) {
    pb_write_int64_field(out, 1, dim);
  }
  pb_write_varint_field(out, 2, static_cast<uint64_t>(elem_type));
  pb_write_string_field(out, 8, name);
}

static uint32_t raw_bytes_u32_le_at(const std::string& raw, size_t offset) {
  const auto b0 = static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 0]));
  const auto b1 = static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 1]));
  const auto b2 = static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 2]));
  const auto b3 = static_cast<uint32_t>(static_cast<unsigned char>(raw[offset + 3]));
  return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

static void pb_encode_tensor_inline_complex64_data(std::string& out, const std::string& raw) {
  if ((raw.size() % sizeof(float)) != 0) {
    throw std::invalid_argument("complex64 initializer raw byte count must be divisible by 4");
  }
  for (size_t offset = 0; offset < raw.size(); offset += sizeof(float)) {
    pb_write_fixed32_field(out, 4, raw_bytes_u32_le_at(raw, offset));
  }
}

static void pb_encode_tensor_inline_data(std::string& out, const std::string& dtype, const std::string& raw) {
  if (dtype == "complex64") {
    pb_encode_tensor_inline_complex64_data(out, raw);
    return;
  }
  pb_write_bytes_field(out, 9, raw);
}

static void pb_encode_tensor_external_data_entries(
    std::string& out,
    const std::string& external_data_file,
    uint64_t external_offset,
    size_t raw_size) {
  pb_write_message_field(out, 13, pb_encode_string_string_entry("location", external_data_file));
  pb_write_message_field(out, 13, pb_encode_string_string_entry("offset", std::to_string(external_offset)));
  pb_write_message_field(out, 13, pb_encode_string_string_entry("length", std::to_string(raw_size)));
  pb_write_varint_field(out, 14, 1);
}

static void append_tensor_external_data(std::string& external_data, uint64_t& external_offset, const std::string& raw) {
  external_data.append(raw);
  external_offset += static_cast<uint64_t>(raw.size());
}

static bool should_externalize_tensor_raw_bytes(
    const OnnxBinaryWriteOptions& options,
    const std::string& raw) {
  return options.external_data &&
      static_cast<int64_t>(raw.size()) >= options.external_data_size_threshold;
}

static std::string pb_encode_tensor(
    const OrderedJson& tensor,
    const OnnxBinaryWriteOptions& options,
    std::string& external_data,
    uint64_t& external_offset,
    bool& has_external_data) {
  const auto name = tensor.at("name").get<std::string>();
  const auto shape = shape_vector_from_json(tensor.at("shape"), "initializer");
  const auto dtype = tensor.at("dtype").get<std::string>();
  const int elem_type = onnx_elem_type_from_dtype(dtype);
  const std::string raw = tensor_raw_bytes_from_initializer(tensor);

  std::string out;
  pb_encode_tensor_header(out, shape, elem_type, name);

  const bool externalize = should_externalize_tensor_raw_bytes(options, raw);

  if (!externalize) {
    pb_encode_tensor_inline_data(out, dtype, raw);
    return out;
  }

  has_external_data = true;
  pb_encode_tensor_external_data_entries(out, options.external_data_file, external_offset, raw.size());
  append_tensor_external_data(external_data, external_offset, raw);
  return out;
}

static std::string pb_encode_graph(
    const OrderedJson& graph,
    const OnnxBinaryWriteOptions& options,
    std::string& external_data,
    bool& has_external_data) {
  std::string out;
  const auto name = graph.at("name").get<std::string>();
  pb_write_string_field(out, 2, name);

  for (const auto& node : graph.at("nodes")) {
    pb_write_message_field(out, 1, pb_encode_node(node));
  }
  uint64_t external_offset = 0;
  for (const auto& initializer : graph.at("initializers")) {
    pb_write_message_field(
        out,
        5,
        pb_encode_tensor(initializer, options, external_data, external_offset, has_external_data));
  }
  for (const auto& input : graph.at("inputs")) {
    pb_write_message_field(out, 11, pb_encode_value_info(input));
  }
  for (const auto& output : graph.at("outputs")) {
    pb_write_message_field(out, 12, pb_encode_value_info(output));
  }
  return out;
}

static std::string pb_encode_opset_import(int64_t opset) {
  std::string out;
  pb_write_string_field(out, 1, "");
  pb_write_int64_field(out, 2, opset);
  return out;
}

static OnnxBinaryArtifact build_onnx_binary_artifact_from_stub(
    const OrderedJson& onnx_stub,
    const OnnxBinaryWriteOptions& options) {
  if (!onnx_stub.is_object()) {
    throw std::invalid_argument("onnx stub must be a JSON object");
  }
  if (!onnx_stub.contains("graph") || !onnx_stub.at("graph").is_object()) {
    throw std::invalid_argument("onnx stub must include graph object");
  }
  const int64_t opset = normalized_integer_scalar(onnx_stub.at("opset"), "onnx_stub opset");
  const std::string producer_name =
      onnx_stub.contains("producer_name") && onnx_stub.at("producer_name").is_string()
      ? onnx_stub.at("producer_name").get<std::string>()
      : "mlx-ruby";

  std::string external_data;
  bool has_external_data = false;
  const std::string graph_message = pb_encode_graph(onnx_stub.at("graph"), options, external_data, has_external_data);

  std::string model;
  pb_write_int64_field(model, 1, 10);
  pb_write_string_field(model, 2, producer_name);
  pb_write_message_field(model, 7, graph_message);
  pb_write_message_field(model, 8, pb_encode_opset_import(opset));

  OnnxBinaryArtifact artifact;
  artifact.model_bytes = std::move(model);
  artifact.external_data_bytes = std::move(external_data);
  artifact.has_external_data = has_external_data;
  return artifact;
}

// ============================================================================
// Section: Binary Artifact IO and Error Translation
// ============================================================================

static void write_binary_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary);
  if (!output.good()) {
    std::ostringstream out;
    out << "failed to open file for write: " << path.string();
    throw std::runtime_error(out.str());
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output.good()) {
    std::ostringstream out;
    out << "failed to write file: " << path.string();
    throw std::runtime_error(out.str());
  }
}

static OnnxBinaryWriteOptions normalize_onnx_binary_write_options(
    const std::string& target_path,
    VALUE external_data,
    VALUE external_data_file,
    VALUE external_data_size_threshold) {
  if (!(external_data == Qtrue || external_data == Qfalse)) {
    throw std::invalid_argument("external_data must be true or false");
  }

  OnnxBinaryWriteOptions options;
  options.external_data = (external_data == Qtrue);
  options.external_data_size_threshold = 1024;
  options.external_data_file = "weights.bin";

  if (!options.external_data) {
    return options;
  }

  const int64_t threshold = NUM2LL(external_data_size_threshold);
  if (threshold < 0) {
    throw std::invalid_argument("external_data_size_threshold must be a non-negative Integer");
  }
  options.external_data_size_threshold = threshold;

  std::string location;
  if (NIL_P(external_data_file)) {
    std::filesystem::path path(target_path);
    std::string base = path.stem().string();
    if (base.empty()) {
      base = "weights";
    }
    location = base + ".data";
  } else {
    location = std_string_from_ruby(external_data_file);
  }
  if (location.empty()) {
    throw std::invalid_argument("external_data_file must be a non-empty filename");
  }
  std::filesystem::path location_path(location);
  if (location_path.has_parent_path() || location.find('/') != std::string::npos ||
      location.find('\\') != std::string::npos) {
    throw std::invalid_argument("external_data_file must be a filename without path separators");
  }
  options.external_data_file = location;
  return options;
}

static VALUE write_onnx_binary_artifact_to_target(
    const std::string& target_path,
    const OnnxBinaryArtifact& artifact,
    const OnnxBinaryWriteOptions& options) {
  std::filesystem::path path(target_path);
  if (!path.has_parent_path()) {
    path = std::filesystem::absolute(path);
  }
  const auto parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  write_binary_file(path, artifact.model_bytes);

  if (options.external_data && artifact.has_external_data) {
    write_binary_file(parent / options.external_data_file, artifact.external_data_bytes);
  }
  return ruby_string_from_std(path.string());
}

static bool graph_ir_is_unsupported_error_message(const std::string& message) {
  return message.rfind("[graph_ir_to_onnx_stub] unsupported", 0) == 0;
}

[[noreturn]] static void raise_graph_ir_native_exception(const std::exception& error) {
  const std::string message(error.what());
  if (!NIL_P(eGraphIRNativeUnsupportedError) && graph_ir_is_unsupported_error_message(message)) {
    VALUE exc = rb_exc_new_str(
        eGraphIRNativeUnsupportedError,
        rb_str_new(message.data(), static_cast<long>(message.size())));
    rb_exc_raise(exc);
  }

  rb_raise(rb_eRuntimeError, "%s", message.c_str());
}

static OrderedJson parse_graph_ir_json_payload(VALUE graph_ir_json) {
  VALUE graph_ir_json_str = StringValue(graph_ir_json);
  return parse_json_payload_from_string(std_string_from_ruby(graph_ir_json_str), "graph ir json");
}

static OrderedJson graph_ir_compatibility_report_payload(const OrderedJson& payload) {
  OrderedJson probe_initializers = OrderedJson::array();
  auto probe_used_tensor_names = collect_payload_tensor_names(payload);
  auto probe_known_shapes = collect_known_tensor_shapes(payload);
  auto probe_known_dtypes = collect_known_tensor_dtypes(payload);

  OrderedJson node_support = OrderedJson::array();
  size_t unsupported_nodes = 0;
  std::set<std::string> unsupported_ops;

  const auto& source_nodes = payload.at("nodes");
  for (size_t index = 0; index < source_nodes.size(); ++index) {
    const auto& node = source_nodes.at(index);
    const auto op = node.at("op").get<std::string>();

    bool supported = false;
    std::optional<std::string> mapped;

    try {
      auto trial_initializers = probe_initializers;
      auto trial_used_tensor_names = probe_used_tensor_names;
      auto trial_known_shapes = probe_known_shapes;
      auto trial_known_dtypes = probe_known_dtypes;
      LoweringContext trial_lowering{
          trial_initializers,
          trial_used_tensor_names,
          trial_known_shapes,
          trial_known_dtypes};

      auto lowered = lower_onnx_node_default(
          node,
          index,
          trial_lowering);
      if (!lowered.empty() && lowered.front().contains("op_type") && lowered.front().at("op_type").is_string()) {
        mapped = lowered.front().at("op_type").get<std::string>();
      } else {
        mapped = onnx_op_type_for_node(node, false, &trial_known_shapes);
      }

      probe_initializers = std::move(trial_initializers);
      probe_used_tensor_names = std::move(trial_used_tensor_names);
      probe_known_shapes = std::move(trial_known_shapes);
      probe_known_dtypes = std::move(trial_known_dtypes);
      supported = true;
    } catch (const std::exception&) {
      try {
        mapped = onnx_op_type_for_node(node, false, &probe_known_shapes);
      } catch (const std::exception&) {
        mapped = std::nullopt;
      }
    }

    OrderedJson entry = OrderedJson::object();
    entry["index"] = index;
    entry["op"] = op;
    entry["supported"] = supported;
    if (mapped.has_value()) {
      entry["onnx_op_type"] = mapped.value();
    } else {
      entry["onnx_op_type"] = nullptr;
    }
    node_support.push_back(std::move(entry));

    if (!supported) {
      ++unsupported_nodes;
      unsupported_ops.insert(op);
    }
  }

  int64_t ir_version = kGraphIrVersion;
  if (payload.contains("ir_version")) {
    const auto& value = payload.at("ir_version");
    if (value.is_number_integer()) {
      ir_version = value.get<int64_t>();
    } else if (value.is_number_unsigned()) {
      const auto raw = value.get<uint64_t>();
      if (raw <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        ir_version = static_cast<int64_t>(raw);
      }
    }
  }

  OrderedJson unsupported_ops_json = OrderedJson::array();
  for (const auto& op : unsupported_ops) {
    unsupported_ops_json.push_back(op);
  }

  OrderedJson report = OrderedJson::object();
  report["format"] = "webgpu_compat_report_v1";
  report["ir_version"] = ir_version;
  report["total_nodes"] = source_nodes.size();
  report["supported_nodes"] = source_nodes.size() - unsupported_nodes;
  report["unsupported_nodes"] = unsupported_nodes;
  report["unsupported_ops"] = std::move(unsupported_ops_json);
  report["ready_for_stub_conversion"] = unsupported_nodes == 0;
  report["nodes"] = std::move(node_support);
  return report;
}

// ============================================================================
// Section: Ruby-Callable Native Entry Helpers
// ============================================================================

static VALUE graph_ir_to_onnx_json_from_source(VALUE graph_ir_source, VALUE opset, VALUE model_name) {
  const bool timing_enabled = graph_ir_native_timing_enabled();
  const auto started_at = std::chrono::steady_clock::now();
  const auto opset_int = normalize_positive_integer(opset, "opset");
  const auto model_name_str = non_empty_model_name(model_name);
  const auto parse_started_at = std::chrono::steady_clock::now();
  const auto payload = parse_graph_ir_source_payload(graph_ir_source);
  const double parse_json_ms = elapsed_millis(parse_started_at);

  const auto lower_started_at = std::chrono::steady_clock::now();
  const auto onnx_payload = graph_ir_to_onnx_json_payload(payload, opset_int, model_name_str);
  const double lower_onnx_ms = elapsed_millis(lower_started_at);
  const auto dump_started_at = std::chrono::steady_clock::now();
  const auto content = onnx_payload.dump();
  const double dump_json_ms = elapsed_millis(dump_started_at);
  if (timing_enabled) {
    emit_graph_ir_to_onnx_json_timing_line(
        opset_int,
        model_name_str,
        parse_json_ms,
        lower_onnx_ms,
        dump_json_ms,
        elapsed_millis(started_at),
        content.size());
  }
  return ruby_string_from_std(content);
}

static VALUE graph_ir_compatibility_report_json_from_source(VALUE graph_ir_source) {
  const auto payload = parse_graph_ir_source_payload(graph_ir_source);
  const auto report = graph_ir_compatibility_report_payload(payload);
  return ruby_string_from_std(report.dump());
}

static GraphIrExportInvocation parse_graph_ir_export_invocation_from_structured_args(
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless,
    const char* method_name) {
  if (!RB_TYPE_P(args_array, T_ARRAY)) {
    std::ostringstream out;
    out << method_name << " args_array must be an Array";
    throw std::invalid_argument(out.str());
  }
  if (!(NIL_P(kwargs_hash) || RB_TYPE_P(kwargs_hash, T_HASH))) {
    std::ostringstream out;
    out << method_name << " kwargs_hash must be a Hash or nil";
    throw std::invalid_argument(out.str());
  }
  if (!(shapeless == Qtrue || shapeless == Qfalse)) {
    std::ostringstream out;
    out << method_name << " shapeless must be true or false";
    throw std::invalid_argument(out.str());
  }

  mx::Args args;
  const long args_len = RARRAY_LEN(args_array);
  args.reserve(static_cast<size_t>(args_len));
  for (long i = 0; i < args_len; ++i) {
    args.push_back(graph_ir_array_from_ruby(rb_ary_entry(args_array, i)));
  }

  mx::Kwargs kwargs = NIL_P(kwargs_hash) ? mx::Kwargs{} : graph_ir_array_map_from_ruby_hash(kwargs_hash);
  if (args.empty() && kwargs.empty()) {
    std::ostringstream out;
    out << "[" << method_name << "] Inputs must include at least one positional or keyword array";
    throw std::invalid_argument(out.str());
  }

  GraphIrExportInvocation invocation;
  invocation.fun = fun;
  invocation.args = std::move(args);
  invocation.kwargs = std::move(kwargs);
  invocation.shapeless = RTEST(shapeless);
  return invocation;
}

// ============================================================================
// Section: Ruby Singleton Method Entry Points
// ============================================================================

static VALUE graph_ir_native_export_graph_ir(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless) {
  try {
    auto invocation = parse_graph_ir_export_invocation_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_graph_ir");
    auto payload = export_graph_ir_payload(invocation);
    return ruby_value_from_ordered_json(payload);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

static VALUE graph_ir_native_export_graph_ir_json(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless) {
  try {
    auto invocation = parse_graph_ir_export_invocation_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_graph_ir_json");
    auto payload = export_graph_ir_payload(invocation);
    return ruby_string_from_std(payload.dump());
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

static VALUE graph_ir_native_graph_ir_to_onnx_json(VALUE, VALUE graph_ir_source, VALUE opset, VALUE model_name) {
  try {
    return graph_ir_to_onnx_json_from_source(graph_ir_source, opset, model_name);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
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
    const bool timing_enabled = graph_ir_native_timing_enabled();
    const auto started_at = std::chrono::steady_clock::now();
    const auto decode_started_at = std::chrono::steady_clock::now();
    auto invocation = parse_graph_ir_export_invocation_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_onnx_json");
    const double args_decode_ms = elapsed_millis(decode_started_at);

    const auto opset_int = normalize_positive_integer(opset, "opset");
    const auto model_name_str = non_empty_model_name(model_name);

    GraphIrExportTimingStats export_stats;
    const auto export_started_at = std::chrono::steady_clock::now();
    const auto payload = export_graph_ir_payload(invocation, timing_enabled ? &export_stats : nullptr);
    const double export_graph_ir_ms = elapsed_millis(export_started_at);
    const auto lower_started_at = std::chrono::steady_clock::now();
    const auto onnx_payload = graph_ir_to_onnx_json_payload(payload, opset_int, model_name_str);
    const double lower_onnx_ms = elapsed_millis(lower_started_at);
    const auto dump_started_at = std::chrono::steady_clock::now();
    const auto content = onnx_payload.dump();
    const double dump_json_ms = elapsed_millis(dump_started_at);

    if (timing_enabled) {
      emit_export_onnx_json_timing_line(
          invocation,
          opset_int,
          model_name_str,
          export_stats,
          args_decode_ms,
          export_graph_ir_ms,
          lower_onnx_ms,
          dump_json_ms,
          elapsed_millis(started_at),
          content.size());
    }

    return ruby_string_from_std(content);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

static VALUE graph_ir_native_export_onnx_compatibility_report(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless) {
  try {
    auto invocation = parse_graph_ir_export_invocation_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_onnx_compatibility_report");
    const auto payload = export_graph_ir_payload(invocation);
    const auto report = graph_ir_compatibility_report_payload(payload);
    return ruby_value_from_ordered_json(report);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

static VALUE graph_ir_native_export_onnx(
    VALUE,
    VALUE target_path,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless,
    VALUE opset,
    VALUE model_name,
    VALUE external_data,
    VALUE external_data_file,
    VALUE external_data_size_threshold) {
  try {
    const auto target = ruby_path_string(target_path, "export_onnx");
    const auto options = normalize_onnx_binary_write_options(
        target,
        external_data,
        external_data_file,
        external_data_size_threshold);
    auto invocation = parse_graph_ir_export_invocation_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_onnx");
    const auto opset_int = normalize_positive_integer(opset, "opset");
    const auto model_name_str = non_empty_model_name(model_name);

    const auto payload = export_graph_ir_payload(invocation);
    const auto onnx_payload = graph_ir_to_onnx_json_payload(payload, opset_int, model_name_str);
    const auto artifact = build_onnx_binary_artifact_from_stub(onnx_payload, options);
    return write_onnx_binary_artifact_to_target(target, artifact, options);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

static VALUE graph_ir_native_graph_ir_to_onnx(
    VALUE,
    VALUE target_path,
    VALUE graph_ir_source,
    VALUE opset,
    VALUE model_name,
    VALUE external_data,
    VALUE external_data_file,
    VALUE external_data_size_threshold) {
  try {
    const auto target = ruby_path_string(target_path, "graph_ir_to_onnx");
    const auto options = normalize_onnx_binary_write_options(
        target,
        external_data,
        external_data_file,
        external_data_size_threshold);
    const auto opset_int = normalize_positive_integer(opset, "opset");
    const auto model_name_str = non_empty_model_name(model_name);
    const auto payload = parse_graph_ir_source_payload(graph_ir_source);
    const auto onnx_payload = graph_ir_to_onnx_json_payload(payload, opset_int, model_name_str);
    const auto artifact = build_onnx_binary_artifact_from_stub(onnx_payload, options);
    return write_onnx_binary_artifact_to_target(target, artifact, options);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

static VALUE graph_ir_native_graph_ir_compatibility_report_json(VALUE, VALUE graph_ir_source) {
  try {
    return graph_ir_compatibility_report_json_from_source(graph_ir_source);
  } catch (const std::exception& error) {
    raise_graph_ir_native_exception(error);
    return Qnil;
  }
}

// ============================================================================
// Section: Ruby Method Binding Registration
// ============================================================================

} // namespace

extern "C" void init_graph_ir_native_bindings(VALUE mMLX) {
  mGraphIR = rb_define_module_under(mMLX, "GraphIR");
  mGraphIRNative = rb_define_module_under(mGraphIR, "Native");
  eGraphIRNativeUnsupportedError =
      rb_define_class_under(mGraphIRNative, "UnsupportedError", rb_eRuntimeError);

  rb_define_singleton_method(
      mGraphIRNative,
      "export_graph_ir",
      RUBY_METHOD_FUNC(graph_ir_native_export_graph_ir),
      4);
  rb_define_singleton_method(
      mGraphIRNative,
      "export_graph_ir_json",
      RUBY_METHOD_FUNC(graph_ir_native_export_graph_ir_json),
      4);
  rb_define_singleton_method(
      mGraphIRNative,
      "graph_ir_to_onnx_json",
      RUBY_METHOD_FUNC(graph_ir_native_graph_ir_to_onnx_json),
      3);
  rb_define_singleton_method(
      mGraphIRNative,
      "graph_ir_to_onnx",
      RUBY_METHOD_FUNC(graph_ir_native_graph_ir_to_onnx),
      7);
  rb_define_singleton_method(
      mGraphIRNative,
      "export_onnx_json",
      RUBY_METHOD_FUNC(graph_ir_native_export_onnx_json),
      6);
  rb_define_singleton_method(
      mGraphIRNative,
      "export_onnx_compatibility_report",
      RUBY_METHOD_FUNC(graph_ir_native_export_onnx_compatibility_report),
      4);
  rb_define_singleton_method(
      mGraphIRNative,
      "export_onnx",
      RUBY_METHOD_FUNC(graph_ir_native_export_onnx),
      10);
  rb_define_singleton_method(
      mGraphIRNative,
      "graph_ir_compatibility_report_json",
      RUBY_METHOD_FUNC(graph_ir_native_graph_ir_compatibility_report_json),
      1);
}
