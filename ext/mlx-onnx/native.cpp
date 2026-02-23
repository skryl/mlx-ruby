#include "native.hpp"

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

#include "json.hpp"
#include "mlx/export.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

using OrderedJson = nlohmann::ordered_json;

namespace {

// Ruby binding/front-end for IR capture and argument/source normalization.
//
// Heavyweight ONNX lowering, compatibility probing, and protobuf encoding live
// in ir_core.{hpp,cpp}. This file keeps Ruby VALUE conversion, tracing
// invocation decoding, and exception translation.

// ============================================================================
// Section: Binding State and Capture Types
// ============================================================================

static VALUE mONNX;
static VALUE mONNXNative;
static VALUE eOnnxNativeUnsupportedError = Qnil;

constexpr int64_t kGraphIrVersion = mlx::onnx::kGraphIrVersion;

using GraphTensorInfo = std::tuple<std::string, mx::Shape, mx::Dtype>;

// Decoded Ruby invocation used by export entry points.
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

// ============================================================================
// Section: Timing and Diagnostics Helpers
// ============================================================================

static bool onnx_native_timing_enabled() {
  // Accept explicit false-like values; everything else enables timing.
  const char* raw = std::getenv("MLX_IR_NATIVE_TIMING");
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

static void emit_onnx_native_timing_line(const std::string& line) {
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
}

static void emit_export_onnx_json_timing_line(
    const GraphIrExportInvocation& invocation,
    int64_t opset,
    const std::string& model_name,
    const GraphIrExportTimingStats& export_stats,
    double args_decode_ms,
    double export_ir_ms,
    double lower_onnx_ms,
    double dump_json_ms,
    double total_ms,
    size_t onnx_json_bytes) {
  // Key=value single-line logs are easier to parse in CI logs.
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << "[mlx.onnx.native.timing] export_onnx_json";
  out << " total_ms=" << total_ms;
  out << " args_decode_ms=" << args_decode_ms;
  out << " export_ir_ms=" << export_ir_ms;
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
  emit_onnx_native_timing_line(out.str());
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
  out << "[mlx.onnx.native.timing] graph_ir_to_onnx_json";
  out << " total_ms=" << total_ms;
  out << " parse_json_ms=" << parse_json_ms;
  out << " lower_onnx_ms=" << lower_onnx_ms;
  out << " json_dump_ms=" << dump_json_ms;
  out << " onnx_json_bytes=" << onnx_json_bytes;
  out << " opset=" << opset;
  out << " model_name=" << model_name;
  emit_onnx_native_timing_line(out.str());
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
// Section: IR Export Capture and Trace Conversion
// ============================================================================

[[noreturn]] static void raise_onnx_native_exception(const std::exception& error);

template <typename ValueAt>
static OrderedJson capture_build_nested_json_array(
    const mx::Shape& shape,
    size_t dim,
    size_t& flat_index,
    ValueAt value_at) {
  // Rebuild N-D JSON nesting from a flat buffer cursor.
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
  // Scalar conversion preserves integer types and normalizes floats to JSON
  // numbers (double) for stable serialization.
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
  // Constants are eagerly materialized so later lowering/encoding has a single
  // JSON representation independent of backend/device buffers.
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
  // Export callback state arguments use a tagged variant; serialize every
  // supported alternative into JSON values consumed by lowering.
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
  // Callback records are heterogenous key/value pairs; this helper combines
  // key lookup and variant type-check in one place.
  for (const auto& [candidate_key, candidate_value] : data) {
    if (candidate_key == key && std::holds_alternative<T>(candidate_value)) {
      return &std::get<T>(candidate_value);
    }
  }
  return nullptr;
}

static OrderedJson export_ir_payload(
    const GraphIrExportInvocation& invocation,
    GraphIrExportTimingStats* timing_stats = nullptr) {
  // Single capture pass: collect graph metadata and primitive nodes from
  // export_function callback records and normalize them into IR JSON.
  OrderedJson graph_inputs = OrderedJson::array();
  OrderedJson keyword_inputs = OrderedJson::array();
  OrderedJson graph_outputs = OrderedJson::array();
  OrderedJson graph_constants = OrderedJson::array();
  OrderedJson graph_nodes = OrderedJson::array();

  const auto trace_started_at = std::chrono::steady_clock::now();
  mx::export_function(
      [&graph_inputs, &keyword_inputs, &graph_outputs, &graph_constants, &graph_nodes, timing_stats](
          const mx::ExportCallbackInput& data) {
        // Record schema comes from mlx::export_function "type" discriminator.
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
              // Constant payloads are embedded by value so that ONNX export can
              // be run later without re-tracing.
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
      onnx_args_kwargs_function_from_callable(invocation.fun),
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
  // Preserve Integer magnitude when possible (int64/uint64), then degrade to
  // double only when outside 64-bit integer JSON range.
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
  // Complex values round-trip via an explicit marker object.
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
  // This conversion intentionally accepts more than strict JSON scalar classes
  // to keep Ruby-side ergonomics predictable.
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
  // Reverse bridge for native methods that return structured JSON payloads.
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

static std::string tagged_ir_api_error(const std::string& message) {
  if (!message.empty() && message.front() == '[') {
    return message;
  }
  return std::string("[ir.api] ") + message;
}

static std::string read_file_to_string(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.good()) {
    std::ostringstream out;
    out << "failed to read file: " << path;
    throw std::runtime_error(tagged_ir_api_error(out.str()));
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
    throw std::invalid_argument(tagged_ir_api_error(out.str()));
  }
}

static OrderedJson parse_ir_source_payload(VALUE source) {
  // Unified source parser for Hash / JSON string / file path / IO-like object.
  // Heuristic for String:
  // - existing regular file path -> read and parse file contents
  // - otherwise -> parse as JSON literal string
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
      throw std::invalid_argument("[ir.api] graph ir path-like source must be non-empty");
    }
    if (!std::filesystem::is_regular_file(path)) {
      std::ostringstream out;
      out << "graph ir path does not exist: " << path;
      throw std::invalid_argument(std::string("[ir.api] ") + out.str());
    }
    return parse_json_payload_from_string(read_file_to_string(path), "graph ir file");
  }

  if (rb_respond_to(source, rb_intern("read"))) {
    VALUE io_raw = rb_funcall(source, rb_intern("read"), 0);
    return parse_json_payload_from_string(std_string_from_ruby(io_raw), "graph ir IO");
  }

  throw std::invalid_argument(
      "[ir.api] graph ir source must be a Hash, JSON String, file path, or IO-like object");
}

static std::string ruby_path_string(VALUE value, const char* label) {
  // Binary export paths are path-like only; IO-like objects are rejected to
  // keep external-data colocated path handling deterministic.
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

static mlx::onnx::OnnxBinaryWriteOptions normalize_onnx_binary_write_options(
    const std::string& target_path,
    VALUE external_data,
    VALUE external_data_file,
    VALUE external_data_size_threshold) {
  // Normalize Ruby kwargs into strict binary writer options.
  if (!(external_data == Qtrue || external_data == Qfalse)) {
    throw std::invalid_argument("external_data must be true or false");
  }

  mlx::onnx::OnnxBinaryWriteOptions options;
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

[[noreturn]] static void raise_onnx_native_exception(const std::exception& error) {
  // Promote lowering "unsupported" errors to typed Ruby exception so callers
  // can distinguish unsupported coverage from generic runtime failures.
  const std::string message(error.what());
  if (!NIL_P(eOnnxNativeUnsupportedError) &&
      mlx::onnx::ir_is_unsupported_error_message(message)) {
    VALUE exc = rb_exc_new_str(
        eOnnxNativeUnsupportedError,
        rb_str_new(message.data(), static_cast<long>(message.size())));
    rb_exc_raise(exc);
  }

  rb_raise(rb_eRuntimeError, "%s", message.c_str());
}

// ============================================================================
// Section: Ruby-Callable Native Entry Helpers
// ============================================================================

static VALUE graph_ir_to_onnx_json_from_source(VALUE ir_source, VALUE opset, VALUE model_name) {
  // Direct source->ONNX JSON entry used by Ruby API and tests.
  const bool timing_enabled = onnx_native_timing_enabled();
  const auto started_at = std::chrono::steady_clock::now();
  const auto opset_int = normalize_positive_integer(opset, "opset");
  const auto model_name_str = non_empty_model_name(model_name);
  const auto parse_started_at = std::chrono::steady_clock::now();
  const auto payload = parse_ir_source_payload(ir_source);
  const double parse_json_ms = elapsed_millis(parse_started_at);

  const auto lower_started_at = std::chrono::steady_clock::now();
  const auto onnx_payload =
      mlx::onnx::ir_to_onnx_json_payload(payload, opset_int, model_name_str);
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

static VALUE graph_ir_compatibility_report_json_from_source(VALUE ir_source) {
  const auto payload = parse_ir_source_payload(ir_source);
  const auto report = mlx::onnx::ir_compatibility_report_payload(payload);
  return ruby_string_from_std(report.dump());
}

static GraphIrExportInvocation parse_ir_export_invocation_from_structured_args(
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless,
    const char* method_name) {
  // Structured parser used by public singleton methods that pass args/kwargs
  // explicitly instead of variadic flattening.
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
    args.push_back(onnx_array_from_ruby(rb_ary_entry(args_array, i)));
  }

  mx::Kwargs kwargs = NIL_P(kwargs_hash) ? mx::Kwargs{} : onnx_array_map_from_ruby_hash(kwargs_hash);
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

struct ParsedExportPayload {
  GraphIrExportInvocation invocation;
  OrderedJson payload;
};

static ParsedExportPayload parse_and_export_payload_from_structured_args(
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless,
    const char* method_name,
    GraphIrExportTimingStats* timing_stats = nullptr) {
  ParsedExportPayload out;
  out.invocation = parse_ir_export_invocation_from_structured_args(
      fun,
      args_array,
      kwargs_hash,
      shapeless,
      method_name);
  out.payload = export_ir_payload(out.invocation, timing_stats);
  return out;
}

static OrderedJson build_onnx_stub_payload(
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name) {
  return mlx::onnx::ir_to_onnx_json_payload(
      payload, opset, model_name);
}

static std::string write_onnx_binary_from_payload(
    const std::string& target,
    const OrderedJson& payload,
    int64_t opset,
    const std::string& model_name,
    const mlx::onnx::OnnxBinaryWriteOptions& options) {
  const auto onnx_payload = build_onnx_stub_payload(payload, opset, model_name);
  const auto artifact =
      mlx::onnx::build_onnx_binary_artifact_from_stub(
          onnx_payload, options);
  return mlx::onnx::write_onnx_binary_artifact_to_path(
      target, artifact, options);
}

// ============================================================================
// Section: Ruby Singleton Method Entry Points
// ============================================================================

static VALUE onnx_native_export_graph_ir(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless) {
  // Entry points are thin wrappers so exception translation remains uniform.
  try {
    auto exported = parse_and_export_payload_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_ir");
    return ruby_value_from_ordered_json(exported.payload);
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_export_graph_ir_json(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless) {
  try {
    auto exported = parse_and_export_payload_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_ir_json");
    return ruby_string_from_std(exported.payload.dump());
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_graph_ir_to_onnx_json(VALUE, VALUE ir_source, VALUE opset, VALUE model_name) {
  try {
    return graph_ir_to_onnx_json_from_source(ir_source, opset, model_name);
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_export_onnx_json(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless,
    VALUE opset,
    VALUE model_name) {
  try {
    const bool timing_enabled = onnx_native_timing_enabled();
    const auto started_at = std::chrono::steady_clock::now();
    const auto decode_started_at = std::chrono::steady_clock::now();
    auto invocation = parse_ir_export_invocation_from_structured_args(
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
    const auto payload = export_ir_payload(
        invocation, timing_enabled ? &export_stats : nullptr);
    const double export_ir_ms = elapsed_millis(export_started_at);
    const auto lower_started_at = std::chrono::steady_clock::now();
    const auto onnx_payload = build_onnx_stub_payload(
        payload, opset_int, model_name_str);
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
          export_ir_ms,
          lower_onnx_ms,
          dump_json_ms,
          elapsed_millis(started_at),
          content.size());
    }

    return ruby_string_from_std(content);
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_export_onnx_compatibility_report(
    VALUE,
    VALUE fun,
    VALUE args_array,
    VALUE kwargs_hash,
    VALUE shapeless) {
  try {
    auto exported = parse_and_export_payload_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_onnx_compatibility_report");
    const auto report =
        mlx::onnx::ir_compatibility_report_payload(
            exported.payload);
    return ruby_value_from_ordered_json(report);
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_export_onnx(
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
    auto exported = parse_and_export_payload_from_structured_args(
        fun,
        args_array,
        kwargs_hash,
        shapeless,
        "native_export_onnx");
    const auto opset_int = normalize_positive_integer(opset, "opset");
    const auto model_name_str = non_empty_model_name(model_name);

    return ruby_string_from_std(write_onnx_binary_from_payload(
        target, exported.payload, opset_int, model_name_str, options));
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_graph_ir_to_onnx(
    VALUE,
    VALUE target_path,
    VALUE ir_source,
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
    const auto payload = parse_ir_source_payload(ir_source);
    return ruby_string_from_std(write_onnx_binary_from_payload(
        target, payload, opset_int, model_name_str, options));
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

static VALUE onnx_native_graph_ir_compatibility_report_json(VALUE, VALUE ir_source) {
  try {
    return graph_ir_compatibility_report_json_from_source(ir_source);
  } catch (const std::exception& error) {
    raise_onnx_native_exception(error);
    return Qnil;
  }
}

// ============================================================================
// Section: Ruby Method Binding Registration
// ============================================================================

} // namespace

extern "C" void init_onnx_native_bindings(VALUE mMLX) {
  mONNX = rb_define_module_under(mMLX, "ONNX");
  mONNXNative = rb_define_module_under(mONNX, "Native");
  eOnnxNativeUnsupportedError =
      rb_define_class_under(mONNXNative, "UnsupportedError", rb_eRuntimeError);

  rb_define_singleton_method(
      mONNXNative,
      "export_graph_ir",
      RUBY_METHOD_FUNC(onnx_native_export_graph_ir),
      4);
  rb_define_singleton_method(
      mONNXNative,
      "export_graph_ir_json",
      RUBY_METHOD_FUNC(onnx_native_export_graph_ir_json),
      4);
  rb_define_singleton_method(
      mONNXNative,
      "graph_ir_to_onnx_json",
      RUBY_METHOD_FUNC(onnx_native_graph_ir_to_onnx_json),
      3);
  rb_define_singleton_method(
      mONNXNative,
      "graph_ir_to_onnx",
      RUBY_METHOD_FUNC(onnx_native_graph_ir_to_onnx),
      7);
  rb_define_singleton_method(
      mONNXNative,
      "export_onnx_json",
      RUBY_METHOD_FUNC(onnx_native_export_onnx_json),
      6);
  rb_define_singleton_method(
      mONNXNative,
      "export_onnx_compatibility_report",
      RUBY_METHOD_FUNC(onnx_native_export_onnx_compatibility_report),
      4);
  rb_define_singleton_method(
      mONNXNative,
      "export_onnx",
      RUBY_METHOD_FUNC(onnx_native_export_onnx),
      10);
  rb_define_singleton_method(
      mONNXNative,
      "ir_compatibility_report_json",
      RUBY_METHOD_FUNC(onnx_native_graph_ir_compatibility_report_json),
      1);
}
