#include <ruby.h>
#include <ruby/thread.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"
#include "mlx/dtype.h"
#include "mlx/einsum.h"
#include "mlx/fft.h"
#include "mlx/fast.h"
#include "mlx/export.h"
#include "mlx/graph_utils.h"
#include "mlx/io.h"
#include "mlx/linalg.h"
#include "mlx/memory.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"
#include "mlx/version.h"

namespace mx = mlx::core;
namespace mxfft = mlx::core::fft;
namespace mxfast = mlx::core::fast;
namespace mxlinalg = mlx::core::linalg;
namespace mxmetal = mlx::core::metal;
namespace mxdist = mlx::core::distributed;

static VALUE mMLX;
static VALUE mNative;
static VALUE mCore;
static VALUE cDtype;
static VALUE cArray;
static VALUE cDevice;
static VALUE cStream;
static VALUE cFunction;
static VALUE cFunctionExporter;
static VALUE cGroup;
static VALUE cKernel;

struct DtypeWrapper {
  mx::Dtype dtype;

  DtypeWrapper() : dtype(mx::bool_) {}
};

struct DeviceWrapper {
  mx::Device device;

  DeviceWrapper() : device(mx::Device::cpu, 0) {}
};

struct ArrayWrapper {
  mx::array array;

  ArrayWrapper() : array(0.0f) {}
};

struct StreamWrapper {
  mx::Stream stream;

  StreamWrapper() : stream(0, mx::Device(mx::Device::cpu, 0)) {}
};

struct GroupWrapper {
  std::optional<mxdist::Group> group;
};

struct FunctionWrapper {
  std::function<std::vector<mx::array>(const std::vector<mx::array>&)> vector_fn;
  std::function<std::vector<mx::array>(const mx::Args&, const mx::Kwargs&)> args_kwargs_fn;
  std::function<std::pair<std::vector<mx::array>, std::vector<mx::array>>(
      const std::vector<mx::array>&)>
      value_grad_fn;
  bool accepts_args_kwargs;
  bool returns_value_and_grad;
  bool always_array_output;
  bool release_gvl;
  VALUE refs;

  FunctionWrapper()
      : accepts_args_kwargs(false),
        returns_value_and_grad(false),
        always_array_output(false),
        release_gvl(false),
        refs(Qnil) {}
};

struct FunctionExporterWrapper {
  std::optional<mx::FunctionExporter> exporter;
  VALUE refs;

  FunctionExporterWrapper() : refs(Qnil) {}
};

struct KernelWrapper {
  mxfast::CustomKernelFunction kernel;
  VALUE refs;
};

static void dtype_free(void* ptr) {
  delete static_cast<DtypeWrapper*>(ptr);
}

static size_t dtype_memsize(const void*) {
  return sizeof(DtypeWrapper);
}

static const rb_data_type_t dtype_data_type = {
    "MLX::Core::Dtype",
    {nullptr, dtype_free, dtype_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void array_free(void* ptr) {
  delete static_cast<ArrayWrapper*>(ptr);
}

static size_t array_memsize(const void*) {
  return sizeof(ArrayWrapper);
}

static const rb_data_type_t array_data_type = {
    "MLX::Core::Array",
    {nullptr, array_free, array_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void device_free(void* ptr) {
  delete static_cast<DeviceWrapper*>(ptr);
}

static size_t device_memsize(const void*) {
  return sizeof(DeviceWrapper);
}

static const rb_data_type_t device_data_type = {
    "MLX::Core::Device",
    {nullptr, device_free, device_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void stream_free(void* ptr) {
  delete static_cast<StreamWrapper*>(ptr);
}

static size_t stream_memsize(const void*) {
  return sizeof(StreamWrapper);
}

static const rb_data_type_t stream_data_type = {
    "MLX::Core::Stream",
    {nullptr, stream_free, stream_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void group_free(void* ptr) {
  delete static_cast<GroupWrapper*>(ptr);
}

static size_t group_memsize(const void*) {
  return sizeof(GroupWrapper);
}

static const rb_data_type_t group_data_type = {
    "MLX::Core::Group",
    {nullptr, group_free, group_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void function_mark(void* ptr) {
  auto* wrapper = static_cast<FunctionWrapper*>(ptr);
  if (wrapper != nullptr) {
    rb_gc_mark(wrapper->refs);
  }
}

static void function_free(void* ptr) {
  delete static_cast<FunctionWrapper*>(ptr);
}

static size_t function_memsize(const void*) {
  return sizeof(FunctionWrapper);
}

static const rb_data_type_t function_data_type = {
    "MLX::Core::Function",
    {function_mark, function_free, function_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void function_exporter_mark(void* ptr) {
  auto* wrapper = static_cast<FunctionExporterWrapper*>(ptr);
  if (wrapper != nullptr) {
    rb_gc_mark(wrapper->refs);
  }
}

static void function_exporter_free(void* ptr) {
  delete static_cast<FunctionExporterWrapper*>(ptr);
}

static size_t function_exporter_memsize(const void*) {
  return sizeof(FunctionExporterWrapper);
}

static const rb_data_type_t function_exporter_data_type = {
    "MLX::Core::FunctionExporter",
    {function_exporter_mark, function_exporter_free, function_exporter_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

static void kernel_mark(void* ptr) {
  auto* wrapper = static_cast<KernelWrapper*>(ptr);
  if (wrapper != nullptr) {
    rb_gc_mark(wrapper->refs);
  }
}

static void kernel_free(void* ptr) {
  delete static_cast<KernelWrapper*>(ptr);
}

static size_t kernel_memsize(const void*) {
  return sizeof(KernelWrapper);
}

static const rb_data_type_t kernel_data_type = {
    "MLX::Core::Kernel",
    {kernel_mark, kernel_free, kernel_memsize, nullptr, nullptr},
    nullptr,
    nullptr,
    RUBY_TYPED_FREE_IMMEDIATELY,
};

// Forward declarations for helpers with mutually dependent ordering.
static VALUE array_wrap(const mx::array& array);
static mx::array array_unwrap(VALUE object);
static std::unordered_map<std::string, mx::array> array_map_from_ruby_hash(VALUE value);

static void raise_std_exception(const std::exception& error) {
  rb_raise(rb_eRuntimeError, "%s", error.what());
}

struct SymbolIdCache {
  ID cpu;
  ID gpu;
  ID bool_;
  ID uint8;
  ID uint16;
  ID uint32;
  ID uint64;
  ID int8;
  ID int16;
  ID int32;
  ID int64;
  ID float16;
  ID float32;
  ID float64;
  ID bfloat16;
  ID complex64;
  ID complexfloating;
  ID floating;
  ID inexact;
  ID signedinteger;
  ID unsignedinteger;
  ID integer;
  ID number;
  ID generic;
};

static const SymbolIdCache& symbol_ids() {
  static const SymbolIdCache cache = {
      rb_intern("cpu"),
      rb_intern("gpu"),
      rb_intern("bool_"),
      rb_intern("uint8"),
      rb_intern("uint16"),
      rb_intern("uint32"),
      rb_intern("uint64"),
      rb_intern("int8"),
      rb_intern("int16"),
      rb_intern("int32"),
      rb_intern("int64"),
      rb_intern("float16"),
      rb_intern("float32"),
      rb_intern("float64"),
      rb_intern("bfloat16"),
      rb_intern("complex64"),
      rb_intern("complexfloating"),
      rb_intern("floating"),
      rb_intern("inexact"),
      rb_intern("signedinteger"),
      rb_intern("unsignedinteger"),
      rb_intern("integer"),
      rb_intern("number"),
      rb_intern("generic"),
  };
  return cache;
}

static VALUE id_to_symbol(ID id) {
  return ID2SYM(id);
}

static mx::Device::DeviceType device_type_from_value(VALUE value) {
  const auto& ids = symbol_ids();
  VALUE symbol = value;
  if (RB_TYPE_P(value, T_STRING)) {
    symbol = rb_str_intern(value);
  }

  if (SYMBOL_P(symbol)) {
    const ID sid = SYM2ID(symbol);
    if (sid == ids.cpu) {
      return mx::Device::cpu;
    }
    if (sid == ids.gpu) {
      return mx::Device::gpu;
    }
  }

  rb_raise(rb_eArgError, "device type must be :cpu or :gpu");
  return mx::Device::cpu;
}

static VALUE device_type_to_symbol(mx::Device::DeviceType type) {
  const auto& ids = symbol_ids();
  switch (type) {
    case mx::Device::cpu:
      return id_to_symbol(ids.cpu);
    case mx::Device::gpu:
      return id_to_symbol(ids.gpu);
    default:
      rb_raise(rb_eRuntimeError, "unknown MLX device type");
      return Qnil;
  }
}

static mx::Dtype dtype_from_symbol(VALUE symbol) {
  const auto& ids = symbol_ids();
  if (!SYMBOL_P(symbol)) {
    rb_raise(rb_eArgError, "unsupported dtype symbol");
  }

  const ID sid = SYM2ID(symbol);
  if (sid == ids.bool_) return mx::bool_;
  if (sid == ids.uint8) return mx::uint8;
  if (sid == ids.uint16) return mx::uint16;
  if (sid == ids.uint32) return mx::uint32;
  if (sid == ids.uint64) return mx::uint64;
  if (sid == ids.int8) return mx::int8;
  if (sid == ids.int16) return mx::int16;
  if (sid == ids.int32) return mx::int32;
  if (sid == ids.int64) return mx::int64;
  if (sid == ids.float16) return mx::float16;
  if (sid == ids.float32) return mx::float32;
  if (sid == ids.float64) return mx::float64;
  if (sid == ids.bfloat16) return mx::bfloat16;
  if (sid == ids.complex64) return mx::complex64;

  rb_raise(rb_eArgError, "unsupported dtype symbol");
  return mx::bool_;
}

static bool symbol_is_dtype(VALUE symbol) {
  if (!SYMBOL_P(symbol)) {
    return false;
  }

  const auto& ids = symbol_ids();
  const ID sid = SYM2ID(symbol);
  return sid == ids.bool_ || sid == ids.uint8 || sid == ids.uint16 || sid == ids.uint32 ||
      sid == ids.uint64 || sid == ids.int8 || sid == ids.int16 || sid == ids.int32 ||
      sid == ids.int64 || sid == ids.float16 || sid == ids.float32 || sid == ids.float64 ||
      sid == ids.bfloat16 || sid == ids.complex64;
}

static bool value_looks_like_dtype(VALUE value) {
  if (rb_obj_is_kind_of(value, cDtype)) {
    return true;
  }
  VALUE symbol = value;
  if (RB_TYPE_P(value, T_STRING)) {
    symbol = rb_str_intern(value);
  }
  return symbol_is_dtype(symbol);
}

static VALUE dtype_to_symbol(const mx::Dtype& dtype) {
  const auto& ids = symbol_ids();
  switch (dtype.val()) {
    case mx::Dtype::Val::bool_:
      return id_to_symbol(ids.bool_);
    case mx::Dtype::Val::uint8:
      return id_to_symbol(ids.uint8);
    case mx::Dtype::Val::uint16:
      return id_to_symbol(ids.uint16);
    case mx::Dtype::Val::uint32:
      return id_to_symbol(ids.uint32);
    case mx::Dtype::Val::uint64:
      return id_to_symbol(ids.uint64);
    case mx::Dtype::Val::int8:
      return id_to_symbol(ids.int8);
    case mx::Dtype::Val::int16:
      return id_to_symbol(ids.int16);
    case mx::Dtype::Val::int32:
      return id_to_symbol(ids.int32);
    case mx::Dtype::Val::int64:
      return id_to_symbol(ids.int64);
    case mx::Dtype::Val::float16:
      return id_to_symbol(ids.float16);
    case mx::Dtype::Val::float32:
      return id_to_symbol(ids.float32);
    case mx::Dtype::Val::float64:
      return id_to_symbol(ids.float64);
    case mx::Dtype::Val::bfloat16:
      return id_to_symbol(ids.bfloat16);
    case mx::Dtype::Val::complex64:
      return id_to_symbol(ids.complex64);
    default:
      rb_raise(rb_eRuntimeError, "unknown MLX dtype value");
      return Qnil;
  }
}

static mx::Dtype::Category category_from_symbol(VALUE symbol) {
  const auto& ids = symbol_ids();
  if (!SYMBOL_P(symbol)) {
    rb_raise(rb_eArgError, "unsupported dtype category symbol");
  }

  const ID sid = SYM2ID(symbol);
  if (sid == ids.complexfloating) return mx::complexfloating;
  if (sid == ids.floating) return mx::floating;
  if (sid == ids.inexact) return mx::inexact;
  if (sid == ids.signedinteger) return mx::signedinteger;
  if (sid == ids.unsignedinteger) return mx::unsignedinteger;
  if (sid == ids.integer) return mx::integer;
  if (sid == ids.number) return mx::number;
  if (sid == ids.generic) return mx::generic;

  rb_raise(rb_eArgError, "unsupported dtype category symbol");
  return mx::generic;
}

static VALUE category_to_symbol(mx::Dtype::Category category) {
  const auto& ids = symbol_ids();
  switch (category) {
    case mx::Dtype::Category::complexfloating:
      return id_to_symbol(ids.complexfloating);
    case mx::Dtype::Category::floating:
      return id_to_symbol(ids.floating);
    case mx::Dtype::Category::inexact:
      return id_to_symbol(ids.inexact);
    case mx::Dtype::Category::signedinteger:
      return id_to_symbol(ids.signedinteger);
    case mx::Dtype::Category::unsignedinteger:
      return id_to_symbol(ids.unsignedinteger);
    case mx::Dtype::Category::integer:
      return id_to_symbol(ids.integer);
    case mx::Dtype::Category::number:
      return id_to_symbol(ids.number);
    case mx::Dtype::Category::generic:
      return id_to_symbol(ids.generic);
    default:
      rb_raise(rb_eRuntimeError, "unknown MLX dtype category");
      return Qnil;
  }
}

static VALUE dtype_wrap(const mx::Dtype& dtype) {
  auto* wrapper = new DtypeWrapper();
  wrapper->dtype = dtype;
  return TypedData_Wrap_Struct(cDtype, &dtype_data_type, wrapper);
}

static mx::Dtype dtype_unwrap(VALUE object) {
  if (!rb_obj_is_kind_of(object, cDtype)) {
    rb_raise(rb_eTypeError, "expected MLX::Core::Dtype");
  }

  DtypeWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, DtypeWrapper, &dtype_data_type, wrapper);
  return wrapper->dtype;
}

static std::optional<mx::Dtype> optional_dtype_from_value(VALUE value) {
  if (NIL_P(value)) {
    return std::nullopt;
  }
  if (rb_obj_is_kind_of(value, cDtype)) {
    return dtype_unwrap(value);
  }

  VALUE symbol = value;
  if (RB_TYPE_P(value, T_STRING)) {
    symbol = rb_str_intern(value);
  }
  if (SYMBOL_P(symbol)) {
    return dtype_from_symbol(symbol);
  }

  rb_raise(rb_eArgError, "dtype must be nil, MLX::Core::Dtype, symbol, or string");
  return std::nullopt;
}

static mx::array cast_if_needed(mx::array a, const std::optional<mx::Dtype>& dtype) {
  if (dtype.has_value() && a.dtype() != dtype.value()) {
    return mx::astype(std::move(a), dtype.value());
  }
  return a;
}

static mx::array scalar_array_from_ruby(VALUE value, const std::optional<mx::Dtype>& dtype) {
  if (RB_TYPE_P(value, T_TRUE) || RB_TYPE_P(value, T_FALSE)) {
    auto a = mx::array(value == Qtrue);
    return cast_if_needed(std::move(a), dtype);
  }
  if (RB_INTEGER_TYPE_P(value)) {
    auto a = mx::array(NUM2LL(value));
    return cast_if_needed(std::move(a), dtype);
  }
  if (RB_FLOAT_TYPE_P(value)) {
    auto a = mx::array(NUM2DBL(value));
    return cast_if_needed(std::move(a), dtype);
  }
  rb_raise(rb_eTypeError, "expected boolean, integer, or float");
  return mx::array(0.0f);
}

static mx::array array_from_ruby(VALUE value, const std::optional<mx::Dtype>& dtype);

static bool is_numeric_scalar(VALUE value) {
  return RB_INTEGER_TYPE_P(value) || RB_FLOAT_TYPE_P(value) ||
      RB_TYPE_P(value, T_TRUE) || RB_TYPE_P(value, T_FALSE);
}

template <typename T>
static T scalar_value_from_ruby(VALUE value) {
  if (RB_INTEGER_TYPE_P(value)) {
    return static_cast<T>(NUM2LL(value));
  }
  if (RB_FLOAT_TYPE_P(value)) {
    return static_cast<T>(NUM2DBL(value));
  }
  if (RB_TYPE_P(value, T_TRUE) || RB_TYPE_P(value, T_FALSE)) {
    return static_cast<T>(value == Qtrue ? 1.0 : 0.0);
  }
  rb_raise(rb_eTypeError, "expected numeric/boolean scalar");
  return static_cast<T>(0);
}

template <typename T>
static void infer_shape_and_flatten_typed(
    VALUE value,
    size_t depth,
    mx::Shape& shape,
    std::vector<T>& flat) {
  if (RB_TYPE_P(value, T_ARRAY)) {
    const long len = RARRAY_LEN(value);
    if (shape.size() == depth) {
      shape.push_back(static_cast<mx::ShapeElem>(len));
    } else if (shape[depth] != len) {
      rb_raise(rb_eArgError, "ragged array input is not supported");
    }
    for (long i = 0; i < len; ++i) {
      infer_shape_and_flatten_typed<T>(rb_ary_entry(value, i), depth + 1, shape, flat);
    }
    return;
  }

  if (!is_numeric_scalar(value)) {
    rb_raise(rb_eTypeError, "nested arrays must contain only numeric/boolean scalars");
  }
  if (shape.size() != depth) {
    rb_raise(rb_eArgError, "inconsistent nested array depth");
  }
  flat.push_back(scalar_value_from_ruby<T>(value));
}

static void infer_shape_and_flatten(
    VALUE value,
    size_t depth,
    mx::Shape& shape,
    std::vector<double>& flat) {
  infer_shape_and_flatten_typed<double>(value, depth, shape, flat);
}

static mx::array tensor_array_from_ruby(VALUE value, const std::optional<mx::Dtype>& dtype) {
  mx::Dtype target_dtype = dtype.value_or(mx::float32);
  mx::Dtype build_dtype = target_dtype;

  // MLX does not support float64 on GPU. Build with float32 and cast only
  // when a different target dtype was explicitly requested.
  if (build_dtype == mx::float64) {
    build_dtype = mx::float32;
  }

  if (build_dtype == mx::float32) {
    mx::Shape shape;
    std::vector<float> data;
    infer_shape_and_flatten_typed<float>(value, 0, shape, data);
    mx::array a(data.begin(), shape, build_dtype);
    if (target_dtype != build_dtype) {
      a = mx::astype(std::move(a), target_dtype);
    }
    return a;
  } else {
    mx::Shape shape;
    std::vector<double> data;
    infer_shape_and_flatten(value, 0, shape, data);
    mx::array a(data.begin(), shape, build_dtype);
    if (target_dtype != build_dtype) {
      a = mx::astype(std::move(a), target_dtype);
    }
    return a;
  }
}

static mx::array array_from_ruby(VALUE value, const std::optional<mx::Dtype>& dtype) {
  if (rb_obj_is_kind_of(value, cArray)) {
    ArrayWrapper* wrapper = nullptr;
    TypedData_Get_Struct(value, ArrayWrapper, &array_data_type, wrapper);
    return cast_if_needed(wrapper->array, dtype);
  }
  if (RB_TYPE_P(value, T_ARRAY)) {
    return tensor_array_from_ruby(value, dtype);
  }
  return scalar_array_from_ruby(value, dtype);
}

static VALUE array_wrap(const mx::array& array) {
  auto* wrapper = new ArrayWrapper();
  wrapper->array = array;
  return TypedData_Wrap_Struct(cArray, &array_data_type, wrapper);
}

static mx::array array_unwrap(VALUE object) {
  if (!rb_obj_is_kind_of(object, cArray)) {
    rb_raise(rb_eTypeError, "expected MLX::Core::Array");
  }

  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, ArrayWrapper, &array_data_type, wrapper);
  return wrapper->array;
}

static mx::Shape shape_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "shape must be an Array of integers");
  }

  const long len = RARRAY_LEN(value);
  mx::Shape shape;
  shape.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    VALUE dim = rb_ary_entry(value, i);
    if (!RB_INTEGER_TYPE_P(dim)) {
      rb_raise(rb_eTypeError, "shape dimensions must be integers");
    }
    const int v = NUM2INT(dim);
    if (v < 0) {
      rb_raise(rb_eArgError, "shape dimensions must be non-negative");
    }
    shape.push_back(static_cast<mx::ShapeElem>(v));
  }
  return shape;
}

static std::vector<int> int_vector_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "expected Array of integers");
  }

  const long len = RARRAY_LEN(value);
  std::vector<int> out;
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    VALUE item = rb_ary_entry(value, i);
    if (!RB_INTEGER_TYPE_P(item)) {
      rb_raise(rb_eTypeError, "array entries must be integers");
    }
    out.push_back(NUM2INT(item));
  }
  return out;
}

static std::optional<std::vector<int>> optional_int_vector_from_value(VALUE value) {
  if (NIL_P(value)) {
    return std::nullopt;
  }
  return int_vector_from_ruby(value);
}

static std::optional<std::vector<int>> optional_axis_vector_from_value(VALUE value) {
  if (NIL_P(value)) {
    return std::nullopt;
  }
  if (RB_INTEGER_TYPE_P(value)) {
    return std::vector<int>{NUM2INT(value)};
  }
  return int_vector_from_ruby(value);
}

static std::vector<int> int_vector_from_ruby_or_scalar(
    VALUE value,
    const std::vector<int>& default_value,
    const char* name) {
  if (NIL_P(value)) {
    return default_value;
  }
  if (RB_INTEGER_TYPE_P(value)) {
    const int v = NUM2INT(value);
    return {v};
  }
  if (RB_TYPE_P(value, T_ARRAY)) {
    return int_vector_from_ruby(value);
  }
  rb_raise(rb_eTypeError, "%s must be an integer or an Array of integers", name);
  return default_value;
}

static std::pair<int, int> int_pair_from_ruby_or_scalar(
    VALUE value,
    std::pair<int, int> default_value,
    const char* name) {
  if (NIL_P(value)) {
    return default_value;
  }
  if (RB_INTEGER_TYPE_P(value)) {
    const int v = NUM2INT(value);
    return {v, v};
  }
  if (!RB_TYPE_P(value, T_ARRAY) || RARRAY_LEN(value) != 2) {
    rb_raise(rb_eTypeError, "%s must be an integer or a 2-element Array", name);
  }

  VALUE first = rb_ary_entry(value, 0);
  VALUE second = rb_ary_entry(value, 1);
  if (!RB_INTEGER_TYPE_P(first) || !RB_INTEGER_TYPE_P(second)) {
    rb_raise(rb_eTypeError, "%s entries must be integers", name);
  }
  return {NUM2INT(first), NUM2INT(second)};
}

static std::tuple<int, int, int> int_triple_from_ruby_or_scalar(
    VALUE value,
    std::tuple<int, int, int> default_value,
    const char* name) {
  if (NIL_P(value)) {
    return default_value;
  }
  if (RB_INTEGER_TYPE_P(value)) {
    const int v = NUM2INT(value);
    return {v, v, v};
  }
  if (!RB_TYPE_P(value, T_ARRAY) || RARRAY_LEN(value) != 3) {
    rb_raise(rb_eTypeError, "%s must be an integer or a 3-element Array", name);
  }

  VALUE x = rb_ary_entry(value, 0);
  VALUE y = rb_ary_entry(value, 1);
  VALUE z = rb_ary_entry(value, 2);
  if (!RB_INTEGER_TYPE_P(x) || !RB_INTEGER_TYPE_P(y) || !RB_INTEGER_TYPE_P(z)) {
    rb_raise(rb_eTypeError, "%s entries must be integers", name);
  }
  return {NUM2INT(x), NUM2INT(y), NUM2INT(z)};
}

static std::pair<std::vector<int>, std::vector<int>>
conv_general_padding_from_ruby(VALUE value) {
  if (NIL_P(value)) {
    return {{0}, {0}};
  }
  if (RB_INTEGER_TYPE_P(value)) {
    const int v = NUM2INT(value);
    return {{v}, {v}};
  }
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(
        rb_eTypeError,
        "padding must be an integer, an Array of integers, or [low, high]");
  }

  if (RARRAY_LEN(value) == 2 &&
      RB_TYPE_P(rb_ary_entry(value, 0), T_ARRAY) &&
      RB_TYPE_P(rb_ary_entry(value, 1), T_ARRAY)) {
    return {
        int_vector_from_ruby(rb_ary_entry(value, 0)),
        int_vector_from_ruby(rb_ary_entry(value, 1))};
  }

  auto symmetric = int_vector_from_ruby(value);
  return {symmetric, symmetric};
}

static mx::Strides strides_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "expected Array of integers");
  }

  const long len = RARRAY_LEN(value);
  mx::Strides out;
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    VALUE item = rb_ary_entry(value, i);
    if (!RB_INTEGER_TYPE_P(item)) {
      rb_raise(rb_eTypeError, "array entries must be integers");
    }
    out.push_back(NUM2LL(item));
  }
  return out;
}

static std::vector<mx::array> array_vector_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "expected Array of MLX::Core::Array values");
  }

  const long len = RARRAY_LEN(value);
  std::vector<mx::array> out;
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    VALUE item = rb_ary_entry(value, i);
    out.push_back(array_unwrap(item));
  }
  return out;
}

static VALUE ruby_array_of_arrays(const std::vector<mx::array>& arrays) {
  VALUE out = rb_ary_new_capa(static_cast<long>(arrays.size()));
  for (const auto& a : arrays) {
    rb_ary_push(out, array_wrap(a));
  }
  return out;
}

static std::optional<mx::array> optional_array_from_value(VALUE value) {
  if (NIL_P(value)) {
    return std::nullopt;
  }
  return array_from_ruby(value, std::nullopt);
}

static std::string string_from_ruby(VALUE value) {
  VALUE as_string = rb_obj_as_string(value);
  return std::string(StringValueCStr(as_string));
}

static std::vector<mx::array> array_sequence_from_ruby(VALUE value, bool* was_scalar) {
  if (rb_obj_is_kind_of(value, cArray)) {
    if (was_scalar != nullptr) {
      *was_scalar = true;
    }
    return {array_unwrap(value)};
  }

  if (RB_TYPE_P(value, T_ARRAY)) {
    if (was_scalar != nullptr) {
      *was_scalar = false;
    }
    return array_vector_from_ruby(value);
  }

  rb_raise(rb_eTypeError, "expected MLX::Core::Array or Array of MLX::Core::Array");
  return {};
}

struct ArrayCollector {
  std::vector<mx::array>* arrays;
};

static void collect_arrays_from_tree(VALUE value, std::vector<mx::array>& arrays);

static int hash_collect_arrays_iter(VALUE, VALUE value, VALUE arg) {
  auto* collector = reinterpret_cast<ArrayCollector*>(arg);
  collect_arrays_from_tree(value, *collector->arrays);
  return ST_CONTINUE;
}

static void collect_arrays_from_tree(VALUE value, std::vector<mx::array>& arrays) {
  if (rb_obj_is_kind_of(value, cArray)) {
    arrays.push_back(array_unwrap(value));
    return;
  }
  if (RB_TYPE_P(value, T_ARRAY)) {
    const long len = RARRAY_LEN(value);
    for (long i = 0; i < len; ++i) {
      collect_arrays_from_tree(rb_ary_entry(value, i), arrays);
    }
    return;
  }
  if (RB_TYPE_P(value, T_HASH)) {
    ArrayCollector collector{&arrays};
    rb_hash_foreach(value, hash_collect_arrays_iter, reinterpret_cast<VALUE>(&collector));
  }
}

struct ArrayMapBuilder {
  std::unordered_map<std::string, mx::array> map;
};

static int hash_to_array_map_iter(VALUE key, VALUE value, VALUE arg) {
  auto* builder = reinterpret_cast<ArrayMapBuilder*>(arg);
  builder->map.insert_or_assign(string_from_ruby(key), array_unwrap(value));
  return ST_CONTINUE;
}

static std::unordered_map<std::string, mx::array> array_map_from_ruby_hash(VALUE value) {
  if (!RB_TYPE_P(value, T_HASH)) {
    rb_raise(rb_eTypeError, "expected Hash mapping String/Symbol keys to MLX::Core::Array");
  }
  ArrayMapBuilder builder;
  rb_hash_foreach(value, hash_to_array_map_iter, reinterpret_cast<VALUE>(&builder));
  return builder.map;
}

static VALUE ruby_hash_of_arrays(const std::unordered_map<std::string, mx::array>& map) {
  VALUE out = rb_hash_new();
  for (const auto& [key, value] : map) {
    VALUE ruby_key = rb_utf8_str_new(key.c_str(), static_cast<long>(key.size()));
    rb_hash_aset(out, ruby_key, array_wrap(value));
  }
  return out;
}

static VALUE ruby_hash_of_strings(const std::unordered_map<std::string, std::string>& map) {
  VALUE out = rb_hash_new();
  for (const auto& [key, value] : map) {
    VALUE ruby_key = rb_utf8_str_new(key.c_str(), static_cast<long>(key.size()));
    VALUE ruby_value = rb_utf8_str_new(value.c_str(), static_cast<long>(value.size()));
    rb_hash_aset(out, ruby_key, ruby_value);
  }
  return out;
}

struct StringMapBuilder {
  std::unordered_map<std::string, std::string> map;
};

static int hash_to_string_map_iter(VALUE key, VALUE value, VALUE arg) {
  auto* builder = reinterpret_cast<StringMapBuilder*>(arg);
  builder->map.insert_or_assign(string_from_ruby(key), string_from_ruby(value));
  return ST_CONTINUE;
}

static std::unordered_map<std::string, std::string> string_map_from_ruby_hash(VALUE value) {
  if (NIL_P(value)) {
    return {};
  }
  if (!RB_TYPE_P(value, T_HASH)) {
    rb_raise(rb_eTypeError, "expected Hash mapping String/Symbol keys to String values");
  }
  StringMapBuilder builder;
  rb_hash_foreach(value, hash_to_string_map_iter, reinterpret_cast<VALUE>(&builder));
  return builder.map;
}

static mx::GGUFMetaData gguf_metadata_from_ruby(VALUE value) {
  if (NIL_P(value)) {
    return std::monostate{};
  }
  if (rb_obj_is_kind_of(value, cArray)) {
    return array_unwrap(value);
  }
  if (RB_TYPE_P(value, T_STRING) || SYMBOL_P(value)) {
    return string_from_ruby(value);
  }
  if (RB_TYPE_P(value, T_ARRAY)) {
    const long len = RARRAY_LEN(value);
    std::vector<std::string> out;
    out.reserve(static_cast<size_t>(len));
    for (long i = 0; i < len; ++i) {
      out.push_back(string_from_ruby(rb_ary_entry(value, i)));
    }
    return out;
  }

  rb_raise(
      rb_eTypeError,
      "GGUF metadata values must be nil, MLX::Core::Array, String/Symbol, or Array of strings");
  return std::monostate{};
}

struct GGUFMetaMapBuilder {
  std::unordered_map<std::string, mx::GGUFMetaData> map;
};

static int hash_to_gguf_meta_map_iter(VALUE key, VALUE value, VALUE arg) {
  auto* builder = reinterpret_cast<GGUFMetaMapBuilder*>(arg);
  builder->map.insert_or_assign(string_from_ruby(key), gguf_metadata_from_ruby(value));
  return ST_CONTINUE;
}

static std::unordered_map<std::string, mx::GGUFMetaData> gguf_meta_map_from_ruby_hash(VALUE value) {
  if (NIL_P(value)) {
    return {};
  }
  if (!RB_TYPE_P(value, T_HASH)) {
    rb_raise(rb_eTypeError, "expected Hash for GGUF metadata");
  }
  GGUFMetaMapBuilder builder;
  rb_hash_foreach(value, hash_to_gguf_meta_map_iter, reinterpret_cast<VALUE>(&builder));
  return builder.map;
}

static VALUE gguf_metadata_to_ruby(const mx::GGUFMetaData& value) {
  if (std::holds_alternative<std::monostate>(value)) {
    return Qnil;
  }
  if (std::holds_alternative<mx::array>(value)) {
    return array_wrap(std::get<mx::array>(value));
  }
  if (std::holds_alternative<std::string>(value)) {
    const auto& s = std::get<std::string>(value);
    return rb_utf8_str_new(s.c_str(), static_cast<long>(s.size()));
  }

  const auto& strings = std::get<std::vector<std::string>>(value);
  VALUE out = rb_ary_new_capa(static_cast<long>(strings.size()));
  for (const auto& s : strings) {
    rb_ary_push(out, rb_utf8_str_new(s.c_str(), static_cast<long>(s.size())));
  }
  return out;
}

static VALUE ruby_hash_of_gguf_metadata(const std::unordered_map<std::string, mx::GGUFMetaData>& map) {
  VALUE out = rb_hash_new();
  for (const auto& [key, value] : map) {
    VALUE ruby_key = rb_utf8_str_new(key.c_str(), static_cast<long>(key.size()));
    rb_hash_aset(out, ruby_key, gguf_metadata_to_ruby(value));
  }
  return out;
}

static VALUE device_wrap(const mx::Device& device) {
  auto* wrapper = new DeviceWrapper();
  wrapper->device = device;
  return TypedData_Wrap_Struct(cDevice, &device_data_type, wrapper);
}

static mx::Device device_unwrap(VALUE object) {
  if (!rb_obj_is_kind_of(object, cDevice)) {
    rb_raise(rb_eTypeError, "expected MLX::Core::Device");
  }

  DeviceWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, DeviceWrapper, &device_data_type, wrapper);
  return wrapper->device;
}

static mx::Device device_from_object_or_type(VALUE object) {
  if (rb_obj_is_kind_of(object, cDevice)) {
    return device_unwrap(object);
  }
  return mx::Device(device_type_from_value(object), 0);
}

static VALUE stream_wrap(const mx::Stream& stream) {
  auto* wrapper = new StreamWrapper();
  wrapper->stream = stream;
  return TypedData_Wrap_Struct(cStream, &stream_data_type, wrapper);
}

static mx::Stream stream_unwrap(VALUE object) {
  if (!rb_obj_is_kind_of(object, cStream)) {
    rb_raise(rb_eTypeError, "expected MLX::Core::Stream");
  }

  StreamWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, StreamWrapper, &stream_data_type, wrapper);
  return wrapper->stream;
}

static VALUE group_wrap(const mxdist::Group& group) {
  auto* wrapper = new GroupWrapper();
  wrapper->group = group;
  return TypedData_Wrap_Struct(cGroup, &group_data_type, wrapper);
}

static mxdist::Group group_unwrap(VALUE object) {
  if (!rb_obj_is_kind_of(object, cGroup)) {
    rb_raise(rb_eTypeError, "expected MLX::Core::Group");
  }
  GroupWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, GroupWrapper, &group_data_type, wrapper);
  if (wrapper == nullptr || !wrapper->group.has_value()) {
    rb_raise(rb_eRuntimeError, "invalid MLX::Core::Group");
  }
  return wrapper->group.value();
}

static std::optional<mxdist::Group> optional_group_from_value(VALUE value) {
  if (NIL_P(value)) {
    return std::nullopt;
  }
  return group_unwrap(value);
}

static mx::StreamOrDevice stream_or_device_from_value(VALUE value) {
  if (NIL_P(value)) {
    return mx::StreamOrDevice{};
  }
  if (rb_obj_is_kind_of(value, cStream)) {
    return stream_unwrap(value);
  }
  return device_from_object_or_type(value);
}

static VALUE function_alloc(VALUE klass) {
  auto* wrapper = new FunctionWrapper();
  return TypedData_Wrap_Struct(klass, &function_data_type, wrapper);
}

static VALUE ruby_from_array_vector_auto(const std::vector<mx::array>& arrays) {
  if (arrays.size() == 1) {
    return array_wrap(arrays.at(0));
  }
  return ruby_array_of_arrays(arrays);
}

static void rethrow_captured_exception(const std::exception_ptr& error) {
  if (error) {
    std::rethrow_exception(error);
  }
}

struct FunctionVectorCallPayload {
  FunctionWrapper* wrapper;
  const std::vector<mx::array>* inputs;
  std::vector<mx::array>* outputs;
  std::exception_ptr error;
};

static void* function_vector_call_without_gvl(void* arg) {
  auto* payload = reinterpret_cast<FunctionVectorCallPayload*>(arg);
  try {
    *payload->outputs = payload->wrapper->vector_fn(*payload->inputs);
  } catch (...) {
    payload->error = std::current_exception();
  }
  return nullptr;
}

struct FunctionArgsKwCallPayload {
  FunctionWrapper* wrapper;
  const mx::Args* args;
  const mx::Kwargs* kwargs;
  std::vector<mx::array>* outputs;
  std::exception_ptr error;
};

static void* function_args_kwargs_call_without_gvl(void* arg) {
  auto* payload = reinterpret_cast<FunctionArgsKwCallPayload*>(arg);
  try {
    *payload->outputs = payload->wrapper->args_kwargs_fn(*payload->args, *payload->kwargs);
  } catch (...) {
    payload->error = std::current_exception();
  }
  return nullptr;
}

struct FunctionValueGradCallPayload {
  FunctionWrapper* wrapper;
  const std::vector<mx::array>* inputs;
  std::pair<std::vector<mx::array>, std::vector<mx::array>>* outputs;
  std::exception_ptr error;
};

static void* function_value_grad_call_without_gvl(void* arg) {
  auto* payload = reinterpret_cast<FunctionValueGradCallPayload*>(arg);
  try {
    *payload->outputs = payload->wrapper->value_grad_fn(*payload->inputs);
  } catch (...) {
    payload->error = std::current_exception();
  }
  return nullptr;
}

static VALUE function_call(int argc, VALUE* argv, VALUE self) {
  try {
    FunctionWrapper* wrapper = nullptr;
    TypedData_Get_Struct(self, FunctionWrapper, &function_data_type, wrapper);
    if (wrapper == nullptr) {
      rb_raise(rb_eRuntimeError, "invalid MLX::Core::Function");
    }

    if (wrapper->accepts_args_kwargs) {
      int positional_argc = argc;
      VALUE kwargs_hash = Qnil;
      if (argc > 0 && RB_TYPE_P(argv[argc - 1], T_HASH)) {
        positional_argc = argc - 1;
        kwargs_hash = argv[argc - 1];
      }

      mx::Args args;
      args.reserve(static_cast<size_t>(positional_argc));
      for (int i = 0; i < positional_argc; ++i) {
        args.push_back(array_from_ruby(argv[i], std::nullopt));
      }
      mx::Kwargs kwargs = NIL_P(kwargs_hash) ? mx::Kwargs{} : array_map_from_ruby_hash(kwargs_hash);
      std::vector<mx::array> outputs;
      if (wrapper->release_gvl) {
        FunctionArgsKwCallPayload payload{wrapper, &args, &kwargs, &outputs, nullptr};
        rb_thread_call_without_gvl(
            function_args_kwargs_call_without_gvl,
            &payload,
            RUBY_UBF_IO,
            nullptr);
        rethrow_captured_exception(payload.error);
      } else {
        outputs = wrapper->args_kwargs_fn(args, kwargs);
      }
      if (wrapper->always_array_output) {
        return ruby_array_of_arrays(outputs);
      }
      return ruby_from_array_vector_auto(outputs);
    }

    std::vector<mx::array> inputs;
    inputs.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
      inputs.push_back(array_from_ruby(argv[i], std::nullopt));
    }

    if (wrapper->returns_value_and_grad) {
      std::pair<std::vector<mx::array>, std::vector<mx::array>> result;
      if (wrapper->release_gvl) {
        FunctionValueGradCallPayload payload{wrapper, &inputs, &result, nullptr};
        rb_thread_call_without_gvl(
            function_value_grad_call_without_gvl,
            &payload,
            RUBY_UBF_IO,
            nullptr);
        rethrow_captured_exception(payload.error);
      } else {
        result = wrapper->value_grad_fn(inputs);
      }
      VALUE out = rb_ary_new_capa(2);
      rb_ary_push(out, ruby_from_array_vector_auto(result.first));
      rb_ary_push(out, ruby_from_array_vector_auto(result.second));
      return out;
    }

    std::vector<mx::array> outputs;
    if (wrapper->release_gvl) {
      FunctionVectorCallPayload payload{wrapper, &inputs, &outputs, nullptr};
      rb_thread_call_without_gvl(
          function_vector_call_without_gvl,
          &payload,
          RUBY_UBF_IO,
          nullptr);
      rethrow_captured_exception(payload.error);
    } else {
      outputs = wrapper->vector_fn(inputs);
    }

    return ruby_from_array_vector_auto(outputs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE function_wrap_vector(
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> fn,
    VALUE refs) {
  VALUE object = function_alloc(cFunction);
  FunctionWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, FunctionWrapper, &function_data_type, wrapper);
  wrapper->vector_fn = std::move(fn);
  wrapper->accepts_args_kwargs = false;
  wrapper->returns_value_and_grad = false;
  wrapper->always_array_output = false;
  wrapper->refs = refs;
  return object;
}

static VALUE function_wrap_args_kwargs(
    std::function<std::vector<mx::array>(const mx::Args&, const mx::Kwargs&)> fn,
    VALUE refs,
    bool always_array_output = false) {
  VALUE object = function_alloc(cFunction);
  FunctionWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, FunctionWrapper, &function_data_type, wrapper);
  wrapper->args_kwargs_fn = std::move(fn);
  wrapper->accepts_args_kwargs = true;
  wrapper->returns_value_and_grad = false;
  wrapper->always_array_output = always_array_output;
  wrapper->refs = refs;
  return object;
}

static VALUE function_wrap_value_grad(
    std::function<std::pair<std::vector<mx::array>, std::vector<mx::array>>(
        const std::vector<mx::array>&)>
        fn,
    VALUE refs) {
  VALUE object = function_alloc(cFunction);
  FunctionWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, FunctionWrapper, &function_data_type, wrapper);
  wrapper->value_grad_fn = std::move(fn);
  wrapper->accepts_args_kwargs = false;
  wrapper->returns_value_and_grad = true;
  wrapper->always_array_output = false;
  wrapper->refs = refs;
  return object;
}

struct RubyCallableCallPayload {
  VALUE callable;
  const std::vector<VALUE>* argv;
  int kw_splat;
};

static VALUE ruby_callable_call_protected(VALUE arg) {
  auto* payload = reinterpret_cast<RubyCallableCallPayload*>(arg);
  if (payload->kw_splat == RB_PASS_KEYWORDS) {
    return rb_funcallv_kw(
        payload->callable,
        rb_intern("call"),
        static_cast<int>(payload->argv->size()),
        payload->argv->data(),
        RB_PASS_KEYWORDS);
  }
  return rb_funcallv(
      payload->callable,
      rb_intern("call"),
      static_cast<int>(payload->argv->size()),
      payload->argv->data());
}

static std::vector<mx::array> call_ruby_callable_as_array_vector(
    VALUE callable,
    const std::vector<mx::array>& inputs) {
  std::vector<VALUE> ruby_args;
  ruby_args.reserve(inputs.size());
  for (const auto& input : inputs) {
    ruby_args.push_back(array_wrap(input));
  }

  RubyCallableCallPayload payload{callable, &ruby_args, RB_NO_KEYWORDS};
  int state = 0;
  VALUE out = rb_protect(
      ruby_callable_call_protected, reinterpret_cast<VALUE>(&payload), &state);
  if (state != 0) {
    VALUE err = rb_errinfo();
    VALUE msg = rb_obj_as_string(err);
    rb_set_errinfo(Qnil);
    throw std::runtime_error(StringValueCStr(msg));
  }

  bool was_scalar = false;
  return array_sequence_from_ruby(out, &was_scalar);
}

static VALUE ruby_keyword_hash_from_arrays(const mx::Kwargs& kwargs) {
  VALUE out = rb_hash_new();
  for (const auto& [key, value] : kwargs) {
    VALUE ruby_key = ID2SYM(rb_intern(key.c_str()));
    rb_hash_aset(out, ruby_key, array_wrap(value));
  }
  return out;
}

static std::vector<mx::array> call_ruby_callable_as_array_vector(
    VALUE callable,
    const mx::Args& args,
    const mx::Kwargs& kwargs) {
  std::vector<VALUE> ruby_args;
  ruby_args.reserve(args.size() + (kwargs.empty() ? 0 : 1));
  for (const auto& arg : args) {
    ruby_args.push_back(array_wrap(arg));
  }
  int kw_splat = RB_NO_KEYWORDS;
  if (!kwargs.empty()) {
    ruby_args.push_back(ruby_keyword_hash_from_arrays(kwargs));
    kw_splat = RB_PASS_KEYWORDS;
  }

  RubyCallableCallPayload payload{callable, &ruby_args, kw_splat};
  int state = 0;
  VALUE out = rb_protect(
      ruby_callable_call_protected, reinterpret_cast<VALUE>(&payload), &state);
  if (state != 0) {
    VALUE err = rb_errinfo();
    VALUE msg = rb_obj_as_string(err);
    rb_set_errinfo(Qnil);
    throw std::runtime_error(StringValueCStr(msg));
  }

  bool was_scalar = false;
  return array_sequence_from_ruby(out, &was_scalar);
}

static std::function<std::vector<mx::array>(const std::vector<mx::array>&)>
vector_function_from_callable(VALUE callable) {
  if (!rb_respond_to(callable, rb_intern("call"))) {
    rb_raise(rb_eTypeError, "expected callable object");
  }
  return [callable](const std::vector<mx::array>& inputs) {
    return call_ruby_callable_as_array_vector(callable, inputs);
  };
}

static std::function<std::vector<mx::array>(const mx::Args&, const mx::Kwargs&)>
args_kwargs_function_from_callable(VALUE callable) {
  if (!rb_respond_to(callable, rb_intern("call"))) {
    rb_raise(rb_eTypeError, "expected callable object");
  }
  return [callable](const mx::Args& args, const mx::Kwargs& kwargs) {
    return call_ruby_callable_as_array_vector(callable, args, kwargs);
  };
}

static std::vector<int> argnums_from_value(VALUE value) {
  if (NIL_P(value)) {
    return {0};
  }
  if (RB_INTEGER_TYPE_P(value)) {
    return {NUM2INT(value)};
  }
  return int_vector_from_ruby(value);
}

static std::vector<int> vmap_axes_from_value(VALUE value) {
  if (NIL_P(value)) {
    return {};
  }
  if (RB_INTEGER_TYPE_P(value)) {
    return {NUM2INT(value)};
  }
  return int_vector_from_ruby(value);
}

static VALUE function_exporter_alloc(VALUE klass) {
  auto* wrapper = new FunctionExporterWrapper();
  return TypedData_Wrap_Struct(klass, &function_exporter_data_type, wrapper);
}

static VALUE function_exporter_wrap(mx::FunctionExporter exporter, VALUE refs) {
  VALUE object = function_exporter_alloc(cFunctionExporter);
  FunctionExporterWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, FunctionExporterWrapper, &function_exporter_data_type, wrapper);
  wrapper->exporter.emplace(std::move(exporter));
  wrapper->refs = refs;
  return object;
}

static VALUE function_exporter_call(int argc, VALUE* argv, VALUE self) {
  try {
    FunctionExporterWrapper* wrapper = nullptr;
    TypedData_Get_Struct(self, FunctionExporterWrapper, &function_exporter_data_type, wrapper);
    if (wrapper == nullptr || !wrapper->exporter.has_value()) {
      rb_raise(rb_eRuntimeError, "invalid MLX::Core::FunctionExporter");
    }

    int positional_argc = argc;
    VALUE kwargs_hash = Qnil;
    if (argc > 0 && RB_TYPE_P(argv[argc - 1], T_HASH)) {
      positional_argc = argc - 1;
      kwargs_hash = argv[argc - 1];
    }

    mx::Args args;
    args.reserve(static_cast<size_t>(positional_argc));
    for (int i = 0; i < positional_argc; ++i) {
      args.push_back(array_from_ruby(argv[i], std::nullopt));
    }
    mx::Kwargs kwargs = NIL_P(kwargs_hash) ? mx::Kwargs{} : array_map_from_ruby_hash(kwargs_hash);
    wrapper->exporter.value()(args, kwargs);
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE function_exporter_close(VALUE self) {
  try {
    FunctionExporterWrapper* wrapper = nullptr;
    TypedData_Get_Struct(self, FunctionExporterWrapper, &function_exporter_data_type, wrapper);
    if (wrapper != nullptr && wrapper->exporter.has_value()) {
      wrapper->exporter.value().close();
    }
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE kernel_alloc(VALUE klass) {
  auto* wrapper = new KernelWrapper();
  wrapper->refs = Qnil;
  return TypedData_Wrap_Struct(klass, &kernel_data_type, wrapper);
}

static VALUE kernel_wrap(mxfast::CustomKernelFunction kernel, VALUE refs) {
  VALUE object = kernel_alloc(cKernel);
  KernelWrapper* wrapper = nullptr;
  TypedData_Get_Struct(object, KernelWrapper, &kernel_data_type, wrapper);
  wrapper->kernel = std::move(kernel);
  wrapper->refs = refs;
  return object;
}

static VALUE hash_fetch_optional(VALUE hash, const char* key) {
  if (!RB_TYPE_P(hash, T_HASH)) {
    rb_raise(rb_eTypeError, "expected Hash");
  }
  VALUE symbol_key = ID2SYM(rb_intern(key));
  VALUE value = rb_hash_lookup2(hash, symbol_key, Qundef);
  if (value != Qundef) {
    return value;
  }
  VALUE string_key = rb_utf8_str_new_cstr(key);
  return rb_hash_lookup2(hash, string_key, Qundef);
}

static VALUE hash_fetch_required(VALUE hash, const char* key) {
  VALUE value = hash_fetch_optional(hash, key);
  if (value == Qundef || NIL_P(value)) {
    rb_raise(rb_eArgError, "missing required keyword: %s", key);
  }
  return value;
}

static std::vector<std::string> string_vector_from_ruby(VALUE value, const char* name) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "%s must be an Array of strings", name);
  }
  std::vector<std::string> out;
  const long len = RARRAY_LEN(value);
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    out.push_back(string_from_ruby(rb_ary_entry(value, i)));
  }
  return out;
}

static std::vector<mx::Shape> shape_vector_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "output_shapes must be an Array of shape Arrays");
  }
  std::vector<mx::Shape> out;
  const long len = RARRAY_LEN(value);
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    out.push_back(shape_from_ruby(rb_ary_entry(value, i)));
  }
  return out;
}

static std::vector<mx::Dtype> dtype_vector_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "output_dtypes must be an Array of dtypes");
  }
  std::vector<mx::Dtype> out;
  const long len = RARRAY_LEN(value);
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    auto dtype = optional_dtype_from_value(rb_ary_entry(value, i));
    if (!dtype.has_value()) {
      rb_raise(rb_eArgError, "dtype entries cannot be nil");
    }
    out.push_back(dtype.value());
  }
  return out;
}

static std::vector<mx::array> array_inputs_from_ruby(VALUE value) {
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "inputs must be an Array");
  }
  std::vector<mx::array> out;
  const long len = RARRAY_LEN(value);
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    out.push_back(array_from_ruby(rb_ary_entry(value, i), std::nullopt));
  }
  return out;
}

static std::vector<std::pair<std::string, mxfast::TemplateArg>> template_args_from_ruby(
    VALUE value) {
  if (NIL_P(value) || value == Qundef) {
    return {};
  }
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "template must be an Array of [name, value] pairs");
  }

  std::vector<std::pair<std::string, mxfast::TemplateArg>> out;
  const long len = RARRAY_LEN(value);
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    VALUE pair = rb_ary_entry(value, i);
    if (!RB_TYPE_P(pair, T_ARRAY) || RARRAY_LEN(pair) != 2) {
      rb_raise(rb_eTypeError, "template entries must be 2-element Arrays");
    }
    VALUE name = rb_ary_entry(pair, 0);
    VALUE val = rb_ary_entry(pair, 1);

    if (RB_TYPE_P(val, T_TRUE) || RB_TYPE_P(val, T_FALSE)) {
      out.push_back({string_from_ruby(name), RTEST(val)});
      continue;
    }
    if (RB_INTEGER_TYPE_P(val)) {
      out.push_back({string_from_ruby(name), NUM2INT(val)});
      continue;
    }
    auto dtype = optional_dtype_from_value(val);
    if (dtype.has_value()) {
      out.push_back({string_from_ruby(name), dtype.value()});
      continue;
    }
    rb_raise(rb_eTypeError, "template values must be bool, int, or dtype");
  }
  return out;
}

static std::vector<mxfast::ScalarArg> scalar_args_from_ruby(VALUE value) {
  if (NIL_P(value) || value == Qundef) {
    return {};
  }
  if (!RB_TYPE_P(value, T_ARRAY)) {
    rb_raise(rb_eTypeError, "scalars must be an Array");
  }
  std::vector<mxfast::ScalarArg> out;
  const long len = RARRAY_LEN(value);
  out.reserve(static_cast<size_t>(len));
  for (long i = 0; i < len; ++i) {
    VALUE item = rb_ary_entry(value, i);
    if (RB_TYPE_P(item, T_TRUE) || RB_TYPE_P(item, T_FALSE)) {
      out.push_back(RTEST(item));
    } else if (RB_INTEGER_TYPE_P(item)) {
      out.push_back(NUM2INT(item));
    } else if (RB_FLOAT_TYPE_P(item)) {
      out.push_back(static_cast<float>(NUM2DBL(item)));
    } else {
      rb_raise(rb_eTypeError, "scalar args must be bool, int, or float");
    }
  }
  return out;
}

static VALUE kernel_call(int argc, VALUE* argv, VALUE self) {
  try {
    if (argc != 1 || !RB_TYPE_P(argv[0], T_HASH)) {
      rb_raise(rb_eArgError, "Kernel#call expects a single keyword Hash argument");
    }
    VALUE kwargs = argv[0];

    KernelWrapper* wrapper = nullptr;
    TypedData_Get_Struct(self, KernelWrapper, &kernel_data_type, wrapper);
    if (wrapper == nullptr) {
      rb_raise(rb_eRuntimeError, "invalid MLX::Core::Kernel");
    }

    auto inputs = array_inputs_from_ruby(hash_fetch_required(kwargs, "inputs"));
    auto output_shapes = shape_vector_from_ruby(hash_fetch_required(kwargs, "output_shapes"));
    auto output_dtypes = dtype_vector_from_ruby(hash_fetch_required(kwargs, "output_dtypes"));
    auto grid = int_triple_from_ruby_or_scalar(hash_fetch_required(kwargs, "grid"), {1, 1, 1}, "grid");
    auto threadgroup =
        int_triple_from_ruby_or_scalar(hash_fetch_required(kwargs, "threadgroup"), {1, 1, 1}, "threadgroup");
    auto template_args = template_args_from_ruby(hash_fetch_optional(kwargs, "template"));
    VALUE init_value = hash_fetch_optional(kwargs, "init_value");
    VALUE verbose = hash_fetch_optional(kwargs, "verbose");
    VALUE stream = hash_fetch_optional(kwargs, "stream");

    std::optional<float> init_value_v = std::nullopt;
    if (init_value != Qundef && !NIL_P(init_value)) {
      init_value_v = static_cast<float>(NUM2DBL(init_value));
    }
    const bool verbose_v = (verbose != Qundef) && RTEST(verbose);
    auto stream_v = stream == Qundef ? mx::StreamOrDevice{} : stream_or_device_from_value(stream);

    auto outputs = wrapper->kernel(
        inputs,
        output_shapes,
        output_dtypes,
        grid,
        threadgroup,
        template_args,
        init_value_v,
        verbose_v,
        stream_v);
    return ruby_array_of_arrays(outputs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE dtype_alloc(VALUE klass) {
  auto* wrapper = new DtypeWrapper();
  return TypedData_Wrap_Struct(klass, &dtype_data_type, wrapper);
}

static VALUE dtype_initialize(VALUE self, VALUE value) {
  VALUE symbol = value;
  if (RB_TYPE_P(value, T_STRING)) {
    symbol = rb_str_intern(value);
  }
  if (!SYMBOL_P(symbol)) {
    rb_raise(rb_eArgError, "dtype initializer expects symbol or string");
  }

  DtypeWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DtypeWrapper, &dtype_data_type, wrapper);
  wrapper->dtype = dtype_from_symbol(symbol);
  return self;
}

static VALUE dtype_size(VALUE self) {
  DtypeWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DtypeWrapper, &dtype_data_type, wrapper);
  return INT2NUM(wrapper->dtype.size());
}

static VALUE dtype_name(VALUE self) {
  DtypeWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DtypeWrapper, &dtype_data_type, wrapper);
  return dtype_to_symbol(wrapper->dtype);
}

static VALUE dtype_equal(VALUE self, VALUE other) {
  if (!rb_obj_is_kind_of(other, cDtype)) {
    return Qfalse;
  }

  DtypeWrapper* lhs = nullptr;
  DtypeWrapper* rhs = nullptr;
  TypedData_Get_Struct(self, DtypeWrapper, &dtype_data_type, lhs);
  TypedData_Get_Struct(other, DtypeWrapper, &dtype_data_type, rhs);

  return lhs->dtype == rhs->dtype ? Qtrue : Qfalse;
}

static VALUE dtype_hash(VALUE self) {
  DtypeWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DtypeWrapper, &dtype_data_type, wrapper);
  return INT2NUM(static_cast<int>(wrapper->dtype.val()));
}

static VALUE dtype_to_s(VALUE self) {
  DtypeWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DtypeWrapper, &dtype_data_type, wrapper);

  std::ostringstream out;
  out << "#<MLX::Core::Dtype :"
      << rb_id2name(SYM2ID(dtype_to_symbol(wrapper->dtype)))
      << " size=" << static_cast<int>(wrapper->dtype.size())
      << ">";

  const auto value = out.str();
  return rb_utf8_str_new(value.c_str(), static_cast<long>(value.size()));
}

static VALUE device_alloc(VALUE klass) {
  auto* wrapper = new DeviceWrapper();
  return TypedData_Wrap_Struct(klass, &device_data_type, wrapper);
}

static VALUE device_initialize(int argc, VALUE* argv, VALUE self) {
  VALUE type;
  VALUE index;
  rb_scan_args(argc, argv, "11", &type, &index);

  const auto device_type = device_type_from_value(type);
  const int device_index = NIL_P(index) ? 0 : NUM2INT(index);

  DeviceWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DeviceWrapper, &device_data_type, wrapper);
  wrapper->device = mx::Device(device_type, device_index);
  return self;
}

static VALUE device_type(VALUE self) {
  DeviceWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DeviceWrapper, &device_data_type, wrapper);
  return device_type_to_symbol(wrapper->device.type);
}

static VALUE device_index(VALUE self) {
  DeviceWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DeviceWrapper, &device_data_type, wrapper);
  return INT2NUM(wrapper->device.index);
}

static VALUE device_equal(VALUE self, VALUE other) {
  if (!rb_obj_is_kind_of(other, cDevice)) {
    return Qfalse;
  }

  DeviceWrapper* lhs = nullptr;
  DeviceWrapper* rhs = nullptr;
  TypedData_Get_Struct(self, DeviceWrapper, &device_data_type, lhs);
  TypedData_Get_Struct(other, DeviceWrapper, &device_data_type, rhs);

  return lhs->device == rhs->device ? Qtrue : Qfalse;
}

static VALUE device_to_s(VALUE self) {
  DeviceWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, DeviceWrapper, &device_data_type, wrapper);

  std::ostringstream out;
  out << "#<MLX::Core::Device type=:"
      << (wrapper->device.type == mx::Device::cpu ? "cpu" : "gpu")
      << " index=" << wrapper->device.index << ">";

  const auto value = out.str();
  return rb_utf8_str_new(value.c_str(), static_cast<long>(value.size()));
}

static VALUE stream_alloc(VALUE klass) {
  auto* wrapper = new StreamWrapper();
  return TypedData_Wrap_Struct(klass, &stream_data_type, wrapper);
}

static VALUE stream_initialize(int argc, VALUE* argv, VALUE self) {
  VALUE index;
  VALUE device;
  rb_scan_args(argc, argv, "11", &index, &device);

  const int stream_index = NIL_P(index) ? 0 : NUM2INT(index);
  mx::Device stream_device = NIL_P(device)
      ? mx::Device(mx::Device::cpu, 0)
      : device_from_object_or_type(device);

  StreamWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, StreamWrapper, &stream_data_type, wrapper);
  wrapper->stream = mx::Stream(stream_index, stream_device);
  return self;
}

static VALUE stream_index(VALUE self) {
  StreamWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, StreamWrapper, &stream_data_type, wrapper);
  return INT2NUM(wrapper->stream.index);
}

static VALUE stream_device(VALUE self) {
  StreamWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, StreamWrapper, &stream_data_type, wrapper);
  return device_wrap(wrapper->stream.device);
}

static VALUE stream_equal(VALUE self, VALUE other) {
  if (!rb_obj_is_kind_of(other, cStream)) {
    return Qfalse;
  }

  StreamWrapper* lhs = nullptr;
  StreamWrapper* rhs = nullptr;
  TypedData_Get_Struct(self, StreamWrapper, &stream_data_type, lhs);
  TypedData_Get_Struct(other, StreamWrapper, &stream_data_type, rhs);

  return lhs->stream == rhs->stream ? Qtrue : Qfalse;
}

static VALUE stream_to_s(VALUE self) {
  StreamWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, StreamWrapper, &stream_data_type, wrapper);

  std::ostringstream out;
  out << "#<MLX::Core::Stream index=" << wrapper->stream.index
      << " device=:"
      << (wrapper->stream.device.type == mx::Device::cpu ? "cpu" : "gpu")
      << ">";

  const auto value = out.str();
  return rb_utf8_str_new(value.c_str(), static_cast<long>(value.size()));
}

static VALUE group_alloc(VALUE klass) {
  auto* wrapper = new GroupWrapper();
  return TypedData_Wrap_Struct(klass, &group_data_type, wrapper);
}

static VALUE group_rank(VALUE self) {
  GroupWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, GroupWrapper, &group_data_type, wrapper);
  if (wrapper == nullptr || !wrapper->group.has_value()) {
    rb_raise(rb_eRuntimeError, "invalid MLX::Core::Group");
  }
  return INT2NUM(wrapper->group->rank());
}

static VALUE group_size(VALUE self) {
  GroupWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, GroupWrapper, &group_data_type, wrapper);
  if (wrapper == nullptr || !wrapper->group.has_value()) {
    rb_raise(rb_eRuntimeError, "invalid MLX::Core::Group");
  }
  return INT2NUM(wrapper->group->size());
}

static VALUE group_split(int argc, VALUE* argv, VALUE self) {
  VALUE color;
  VALUE key;
  rb_scan_args(argc, argv, "11", &color, &key);
  const int color_v = NUM2INT(color);
  const int key_v = NIL_P(key) ? -1 : NUM2INT(key);

  GroupWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, GroupWrapper, &group_data_type, wrapper);
  if (wrapper == nullptr || !wrapper->group.has_value()) {
    rb_raise(rb_eRuntimeError, "invalid MLX::Core::Group");
  }
  return group_wrap(wrapper->group->split(color_v, key_v));
}

static VALUE group_to_s(VALUE self) {
  GroupWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, GroupWrapper, &group_data_type, wrapper);
  if (wrapper == nullptr || !wrapper->group.has_value()) {
    return rb_utf8_str_new_cstr("#<MLX::Core::Group invalid>");
  }
  std::ostringstream out;
  out << "#<MLX::Core::Group rank=" << wrapper->group->rank()
      << " size=" << wrapper->group->size() << ">";
  const auto value = out.str();
  return rb_utf8_str_new(value.c_str(), static_cast<long>(value.size()));
}

static VALUE array_alloc(VALUE klass) {
  auto* wrapper = new ArrayWrapper();
  return TypedData_Wrap_Struct(klass, &array_data_type, wrapper);
}

static VALUE array_initialize(int argc, VALUE* argv, VALUE self) {
  try {
    VALUE value;
    VALUE dtype;
    rb_scan_args(argc, argv, "11", &value, &dtype);

    ArrayWrapper* wrapper = nullptr;
    TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);
    wrapper->array = array_from_ruby(value, optional_dtype_from_value(dtype));
    return self;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE array_ndim(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);
  return INT2NUM(static_cast<int>(wrapper->array.ndim()));
}

static VALUE array_size(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);
  return ULL2NUM(static_cast<unsigned long long>(wrapper->array.size()));
}

static VALUE array_shape(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);

  VALUE shape = rb_ary_new_capa(static_cast<long>(wrapper->array.ndim()));
  for (auto dim : wrapper->array.shape()) {
    rb_ary_push(shape, INT2NUM(dim));
  }
  return shape;
}

static VALUE array_dtype(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);
  return dtype_wrap(wrapper->array.dtype());
}

static VALUE ruby_scalar_from_array(const mx::array& array) {
  switch (array.dtype()) {
    case mx::bool_:
      return array.item<bool>() ? Qtrue : Qfalse;
    case mx::uint8:
      return UINT2NUM(array.item<uint8_t>());
    case mx::uint16:
      return UINT2NUM(array.item<uint16_t>());
    case mx::uint32:
      return UINT2NUM(array.item<uint32_t>());
    case mx::uint64:
      return ULL2NUM(array.item<uint64_t>());
    case mx::int8:
      return INT2NUM(array.item<int8_t>());
    case mx::int16:
      return INT2NUM(array.item<int16_t>());
    case mx::int32:
      return INT2NUM(array.item<int32_t>());
    case mx::int64:
      return LL2NUM(array.item<int64_t>());
    case mx::float16:
      return DBL2NUM(static_cast<double>(array.item<mx::float16_t>()));
    case mx::bfloat16:
      return DBL2NUM(static_cast<double>(array.item<mx::bfloat16_t>()));
    case mx::float32:
      return DBL2NUM(static_cast<double>(array.item<float>()));
    case mx::float64:
      return DBL2NUM(array.item<double>());
    default:
      rb_raise(rb_eTypeError, "unsupported dtype for scalar conversion");
      return Qnil;
  }
}

static VALUE array_item(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);
  if (wrapper->array.size() != 1) {
    rb_raise(rb_eRuntimeError, "item is only available for size-1 arrays");
  }

  wrapper->array.eval();
  return ruby_scalar_from_array(wrapper->array);
}

template <typename ValueAt>
static VALUE build_nested_ruby_array(
    const mx::Shape& shape,
    size_t dim,
    size_t& flat_index,
    ValueAt value_at) {
  if (dim == shape.size()) {
    return value_at(flat_index++);
  }

  const long n = static_cast<long>(shape[dim]);
  VALUE out = rb_ary_new_capa(n);
  for (long i = 0; i < n; ++i) {
    rb_ary_push(out, build_nested_ruby_array(shape, dim + 1, flat_index, value_at));
  }
  return out;
}

static VALUE array_to_a(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);

  if (wrapper->array.ndim() == 0) {
    wrapper->array.eval();
    return ruby_scalar_from_array(wrapper->array);
  }

  mx::array flat = mx::reshape(
      wrapper->array, mx::Shape{static_cast<mx::ShapeElem>(wrapper->array.size())});
  flat.eval();

  size_t idx = 0;
  switch (flat.dtype()) {
    case mx::bool_: {
      const bool* data = flat.data<bool>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return data[i] ? Qtrue : Qfalse; });
    }
    case mx::uint8: {
      const uint8_t* data = flat.data<uint8_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return UINT2NUM(data[i]); });
    }
    case mx::uint16: {
      const uint16_t* data = flat.data<uint16_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return UINT2NUM(data[i]); });
    }
    case mx::uint32: {
      const uint32_t* data = flat.data<uint32_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return UINT2NUM(data[i]); });
    }
    case mx::uint64: {
      const uint64_t* data = flat.data<uint64_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return ULL2NUM(data[i]); });
    }
    case mx::int8: {
      const int8_t* data = flat.data<int8_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return INT2NUM(data[i]); });
    }
    case mx::int16: {
      const int16_t* data = flat.data<int16_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return INT2NUM(data[i]); });
    }
    case mx::int32: {
      const int32_t* data = flat.data<int32_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return INT2NUM(data[i]); });
    }
    case mx::int64: {
      const int64_t* data = flat.data<int64_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return LL2NUM(data[i]); });
    }
    case mx::float16: {
      const mx::float16_t* data = flat.data<mx::float16_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return DBL2NUM(static_cast<double>(data[i])); });
    }
    case mx::bfloat16: {
      const mx::bfloat16_t* data = flat.data<mx::bfloat16_t>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return DBL2NUM(static_cast<double>(data[i])); });
    }
    case mx::float32: {
      const float* data = flat.data<float>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return DBL2NUM(static_cast<double>(data[i])); });
    }
    case mx::float64: {
      const double* data = flat.data<double>();
      return build_nested_ruby_array(
          wrapper->array.shape(),
          0,
          idx,
          [&](size_t i) { return DBL2NUM(data[i]); });
    }
    default:
      rb_raise(rb_eTypeError, "to_a unsupported for current dtype in this phase");
      return Qnil;
  }
}

static VALUE array_to_s(VALUE self) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);

  std::ostringstream out;
  out << "#<MLX::Core::Array shape=[";
  for (size_t i = 0; i < wrapper->array.ndim(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << wrapper->array.shape(i);
  }
  out << "] dtype=:" << rb_id2name(SYM2ID(dtype_to_symbol(wrapper->array.dtype()))) << ">";

  const auto value = out.str();
  return rb_utf8_str_new(value.c_str(), static_cast<long>(value.size()));
}

static VALUE array_binary_op(VALUE self, VALUE other, const std::function<mx::array(mx::array, mx::array)>& op) {
  ArrayWrapper* wrapper = nullptr;
  TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);

  auto rhs = array_from_ruby(other, std::nullopt);
  auto out = op(wrapper->array, rhs);
  return array_wrap(out);
}

static VALUE array_add(VALUE self, VALUE other) {
  return array_binary_op(self, other, [](mx::array a, mx::array b) { return mx::add(a, b); });
}

static VALUE array_subtract(VALUE self, VALUE other) {
  return array_binary_op(self, other, [](mx::array a, mx::array b) { return mx::subtract(a, b); });
}

static VALUE array_multiply(VALUE self, VALUE other) {
  return array_binary_op(self, other, [](mx::array a, mx::array b) { return mx::multiply(a, b); });
}

static VALUE array_divide(VALUE self, VALUE other) {
  return array_binary_op(self, other, [](mx::array a, mx::array b) { return mx::divide(a, b); });
}

static VALUE array_aref(VALUE self, VALUE index) {
  try {
    ArrayWrapper* wrapper = nullptr;
    TypedData_Get_Struct(self, ArrayWrapper, &array_data_type, wrapper);

    if (!RB_INTEGER_TYPE_P(index)) {
      rb_raise(rb_eTypeError, "index must be an integer in this phase");
    }
    if (wrapper->array.ndim() == 0) {
      rb_raise(rb_eArgError, "cannot index a scalar array");
    }

    int i = NUM2INT(index);
    const int axis_size = wrapper->array.shape(0);
    if (i < 0) {
      i += axis_size;
    }
    if (i < 0 || i >= axis_size) {
      rb_raise(rb_eIndexError, "index out of range");
    }

    mx::Shape start(wrapper->array.ndim(), 0);
    mx::Shape stop = wrapper->array.shape();
    start[0] = i;
    stop[0] = i + 1;

    auto sliced = mx::slice(wrapper->array, start, stop);
    auto squeezed = mx::squeeze(sliced, 0);
    return array_wrap(squeezed);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_array(int argc, VALUE* argv, VALUE) {
  try {
    VALUE value;
    VALUE dtype;
    rb_scan_args(argc, argv, "11", &value, &dtype);
    return array_wrap(array_from_ruby(value, optional_dtype_from_value(dtype)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_broadcast_shapes(int argc, VALUE* argv, VALUE) {
  try {
    if (argc == 0) {
      rb_raise(rb_eArgError, "broadcast_shapes expects at least one shape");
    }

    auto result = shape_from_ruby(argv[0]);
    for (int i = 1; i < argc; ++i) {
      result = mx::broadcast_shapes(result, shape_from_ruby(argv[i]));
    }

    VALUE out = rb_ary_new_capa(static_cast<long>(result.size()));
    for (auto dim : result) {
      rb_ary_push(out, INT2NUM(dim));
    }
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_add(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::add(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_subtract(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::subtract(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_multiply(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::multiply(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_divide(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::divide(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_power(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::power(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_remainder(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::remainder(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_divmod(VALUE, VALUE a, VALUE b) {
  try {
    auto result = mx::divmod(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt));
    VALUE out = rb_ary_new_capa(2);
    if (result.size() != 2) {
      rb_raise(rb_eRuntimeError, "divmod returned unexpected number of outputs");
    }
    rb_ary_push(out, array_wrap(result[0]));
    rb_ary_push(out, array_wrap(result[1]));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_slice(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE start;
    VALUE stop;
    VALUE strides;
    rb_scan_args(argc, argv, "31", &array, &start, &stop, &strides);

    auto a = array_unwrap(array);
    auto start_shape = shape_from_ruby(start);
    auto stop_shape = shape_from_ruby(stop);

    if (NIL_P(strides)) {
      return array_wrap(mx::slice(a, start_shape, stop_shape));
    }
    auto stride_shape = shape_from_ruby(strides);
    return array_wrap(mx::slice(a, start_shape, stop_shape, stride_shape));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_slice_update(int argc, VALUE* argv, VALUE) {
  try {
    VALUE src;
    VALUE update;
    VALUE start;
    VALUE stop;
    VALUE strides;
    rb_scan_args(argc, argv, "41", &src, &update, &start, &stop, &strides);

    auto src_a = array_unwrap(src);
    auto update_a = array_unwrap(update);
    auto start_shape = shape_from_ruby(start);
    auto stop_shape = shape_from_ruby(stop);
    if (NIL_P(strides)) {
      return array_wrap(mx::slice_update(src_a, update_a, start_shape, stop_shape));
    }
    auto stride_shape = shape_from_ruby(strides);
    return array_wrap(mx::slice_update(src_a, update_a, start_shape, stop_shape, stride_shape));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_take(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE indices;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &indices, &axis);
    auto a = array_unwrap(array);

    if (RB_INTEGER_TYPE_P(indices)) {
      const int idx = NUM2INT(indices);
      if (NIL_P(axis)) {
        return array_wrap(mx::take(a, idx));
      }
      return array_wrap(mx::take(a, idx, NUM2INT(axis)));
    }

    auto idx_array = RB_TYPE_P(indices, T_ARRAY)
        ? array_from_ruby(indices, std::make_optional(mx::int32))
        : array_from_ruby(indices, std::nullopt);
    if (NIL_P(axis)) {
      return array_wrap(mx::take(a, idx_array));
    }
    return array_wrap(mx::take(a, idx_array, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_take_along_axis(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE indices;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &indices, &axis);

    auto a = array_unwrap(array);
    auto idx = RB_TYPE_P(indices, T_ARRAY)
        ? array_from_ruby(indices, std::make_optional(mx::int32))
        : array_from_ruby(indices, std::nullopt);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      return array_wrap(mx::take_along_axis(flat, idx, 0));
    }
    return array_wrap(mx::take_along_axis(a, idx, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_put_along_axis(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE indices;
    VALUE values;
    VALUE axis;
    rb_scan_args(argc, argv, "31", &array, &indices, &values, &axis);

    auto a = array_unwrap(array);
    auto idx = RB_TYPE_P(indices, T_ARRAY)
        ? array_from_ruby(indices, std::make_optional(mx::int32))
        : array_from_ruby(indices, std::nullopt);
    auto vals = array_from_ruby(values, std::nullopt);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      auto updated = mx::put_along_axis(flat, idx, vals, 0);
      return array_wrap(mx::reshape(updated, a.shape()));
    }
    return array_wrap(mx::put_along_axis(a, idx, vals, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_pad(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE pad_width;
    VALUE mode;
    VALUE constant_values;
    rb_scan_args(argc, argv, "22", &array, &pad_width, &mode, &constant_values);

    auto a = array_unwrap(array);
    auto pad_value = NIL_P(constant_values)
        ? mx::array(0)
        : array_from_ruby(constant_values, std::nullopt);
    const std::string mode_v = NIL_P(mode) ? "constant" : StringValueCStr(mode);

    if (RB_INTEGER_TYPE_P(pad_width)) {
      return array_wrap(mx::pad(a, NUM2INT(pad_width), pad_value, mode_v));
    }

    if (!RB_TYPE_P(pad_width, T_ARRAY)) {
      rb_raise(
          rb_eTypeError,
          "pad_width must be an Integer, [before, after], or [[before, after], ...]");
    }

    const long len = RARRAY_LEN(pad_width);
    if (len == 0) {
      rb_raise(rb_eArgError, "pad_width must not be empty");
    }

    VALUE first = rb_ary_entry(pad_width, 0);
    if (RB_INTEGER_TYPE_P(first)) {
      if (len == 1) {
        return array_wrap(mx::pad(a, NUM2INT(first), pad_value, mode_v));
      }
      if (len != 2 || !RB_INTEGER_TYPE_P(rb_ary_entry(pad_width, 1))) {
        rb_raise(rb_eTypeError, "pad_width array must be [before, after]");
      }
      std::pair<int, int> pair{NUM2INT(first), NUM2INT(rb_ary_entry(pad_width, 1))};
      return array_wrap(mx::pad(a, pair, pad_value, mode_v));
    }

    std::vector<std::pair<int, int>> widths;
    widths.reserve(static_cast<size_t>(len));
    for (long i = 0; i < len; ++i) {
      VALUE entry = rb_ary_entry(pad_width, i);
      if (!RB_TYPE_P(entry, T_ARRAY) || RARRAY_LEN(entry) != 2) {
        rb_raise(
            rb_eTypeError,
            "pad_width nested form must be [[before, after], ...]");
      }
      VALUE lo = rb_ary_entry(entry, 0);
      VALUE hi = rb_ary_entry(entry, 1);
      if (!RB_INTEGER_TYPE_P(lo) || !RB_INTEGER_TYPE_P(hi)) {
        rb_raise(rb_eTypeError, "pad_width entries must be integers");
      }
      widths.push_back({NUM2INT(lo), NUM2INT(hi)});
    }
    if (widths.size() == 1) {
      return array_wrap(mx::pad(a, widths[0], pad_value, mode_v));
    }
    return array_wrap(mx::pad(a, widths, pad_value, mode_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_unflatten(VALUE, VALUE array, VALUE axis, VALUE shape) {
  try {
    return array_wrap(mx::unflatten(array_unwrap(array), NUM2INT(axis), shape_from_ruby(shape)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_as_strided(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE shape;
    VALUE strides;
    VALUE offset;
    rb_scan_args(argc, argv, "31", &array, &shape, &strides, &offset);
    const size_t offset_v = NIL_P(offset) ? 0 : static_cast<size_t>(NUM2ULL(offset));
    return array_wrap(mx::as_strided(
        array_unwrap(array),
        shape_from_ruby(shape),
        strides_from_ruby(strides),
        offset_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_concatenate(int argc, VALUE* argv, VALUE) {
  try {
    VALUE arrays;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &arrays, &axis);
    auto values = array_vector_from_ruby(arrays);
    if (NIL_P(axis)) {
      return array_wrap(mx::concatenate(std::move(values)));
    }
    return array_wrap(mx::concatenate(std::move(values), NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_concat(int argc, VALUE* argv, VALUE self) {
  return core_concatenate(argc, argv, self);
}

static VALUE core_stack(int argc, VALUE* argv, VALUE) {
  try {
    VALUE arrays;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &arrays, &axis);
    auto values = array_vector_from_ruby(arrays);
    if (NIL_P(axis)) {
      return array_wrap(mx::stack(values));
    }
    return array_wrap(mx::stack(values, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_split(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE split_spec;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &split_spec, &axis);

    auto a = array_unwrap(array);
    if (RB_INTEGER_TYPE_P(split_spec)) {
      const int parts = NUM2INT(split_spec);
      if (NIL_P(axis)) {
        return ruby_array_of_arrays(mx::split(a, parts));
      }
      return ruby_array_of_arrays(mx::split(a, parts, NUM2INT(axis)));
    }

    auto indices = shape_from_ruby(split_spec);
    if (NIL_P(axis)) {
      return ruby_array_of_arrays(mx::split(a, indices));
    }
    return ruby_array_of_arrays(mx::split(a, indices, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_repeat(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE repeats;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &repeats, &axis);

    auto a = array_unwrap(array);
    const int repeats_v = NUM2INT(repeats);
    if (NIL_P(axis)) {
      return array_wrap(mx::repeat(a, repeats_v));
    }
    return array_wrap(mx::repeat(a, repeats_v, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tile(VALUE, VALUE array, VALUE reps) {
  try {
    auto a = array_unwrap(array);
    std::vector<int> reps_v;
    if (RB_INTEGER_TYPE_P(reps)) {
      reps_v = {NUM2INT(reps)};
    } else {
      reps_v = int_vector_from_ruby(reps);
    }
    return array_wrap(mx::tile(a, reps_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_meshgrid(int argc, VALUE* argv, VALUE) {
  try {
    VALUE arrays;
    VALUE sparse;
    VALUE indexing;
    rb_scan_args(argc, argv, "12", &arrays, &sparse, &indexing);

    auto values = array_vector_from_ruby(arrays);
    const bool sparse_v = NIL_P(sparse) ? false : RTEST(sparse);
    const std::string indexing_v = NIL_P(indexing) ? "xy" : StringValueCStr(indexing);

    return ruby_array_of_arrays(mx::meshgrid(values, sparse_v, indexing_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_roll(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE shift;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &shift, &axis);

    auto a = array_unwrap(array);
    if (RB_INTEGER_TYPE_P(shift)) {
      const int shift_v = NUM2INT(shift);
      if (NIL_P(axis)) {
        return array_wrap(mx::roll(a, shift_v));
      }
      if (RB_INTEGER_TYPE_P(axis)) {
        return array_wrap(mx::roll(a, shift_v, NUM2INT(axis)));
      }
      return array_wrap(mx::roll(a, shift_v, int_vector_from_ruby(axis)));
    }

    auto shifts_i = int_vector_from_ruby(shift);
    mx::Shape shifts;
    shifts.reserve(shifts_i.size());
    for (int value : shifts_i) {
      shifts.push_back(static_cast<mx::ShapeElem>(value));
    }

    if (NIL_P(axis)) {
      return array_wrap(mx::roll(a, shifts));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::roll(a, shifts, NUM2INT(axis)));
    }
    return array_wrap(mx::roll(a, shifts, int_vector_from_ruby(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_stop_gradient(VALUE, VALUE array) {
  try {
    return array_wrap(mx::stop_gradient(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conjugate(VALUE, VALUE array) {
  try {
    return array_wrap(mx::conjugate(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_real(VALUE, VALUE array) {
  try {
    return array_wrap(mx::real(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_imag(VALUE, VALUE array) {
  try {
    return array_wrap(mx::imag(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_contiguous(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE allow_col_major;
    rb_scan_args(argc, argv, "11", &array, &allow_col_major);
    const bool allow_col_major_v = NIL_P(allow_col_major) ? false : RTEST(allow_col_major);
    return array_wrap(mx::contiguous(array_unwrap(array), allow_col_major_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_view(VALUE, VALUE array, VALUE dtype) {
  try {
    auto target_dtype = optional_dtype_from_value(dtype);
    if (!target_dtype.has_value()) {
      rb_raise(rb_eArgError, "view requires a dtype");
    }
    return array_wrap(mx::view(array_unwrap(array), target_dtype.value()));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_matmul(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::matmul(array_unwrap(a), array_unwrap(b)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_addmm(int argc, VALUE* argv, VALUE) {
  try {
    VALUE c;
    VALUE a;
    VALUE b;
    VALUE alpha;
    VALUE beta;
    rb_scan_args(argc, argv, "32", &c, &a, &b, &alpha, &beta);

    const float alpha_v = NIL_P(alpha) ? 1.0f : static_cast<float>(NUM2DBL(alpha));
    const float beta_v = NIL_P(beta) ? 1.0f : static_cast<float>(NUM2DBL(beta));
    return array_wrap(mx::addmm(
        array_from_ruby(c, std::nullopt),
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        alpha_v,
        beta_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_block_masked_mm(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2 || argc > 6) {
      rb_raise(rb_eArgError, "block_masked_mm expects 2 to 6 arguments");
    }

    const auto a = array_from_ruby(argv[0], std::nullopt);
    const auto b = array_from_ruby(argv[1], std::nullopt);
    const int block_size =
        (argc >= 3 && !NIL_P(argv[2])) ? NUM2INT(argv[2]) : 64;

    std::optional<mx::array> mask_out = std::nullopt;
    std::optional<mx::array> mask_lhs = std::nullopt;
    std::optional<mx::array> mask_rhs = std::nullopt;

    if (argc >= 4 && !NIL_P(argv[3])) {
      mask_out = array_from_ruby(argv[3], std::nullopt);
    }
    if (argc >= 5 && !NIL_P(argv[4])) {
      mask_lhs = array_from_ruby(argv[4], std::nullopt);
    }
    if (argc >= 6 && !NIL_P(argv[5])) {
      mask_rhs = array_from_ruby(argv[5], std::nullopt);
    }

    return array_wrap(mx::block_masked_mm(
        a,
        b,
        block_size,
        mask_out,
        mask_lhs,
        mask_rhs));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_gather_mm(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2 || argc > 5) {
      rb_raise(rb_eArgError, "gather_mm expects 2 to 5 arguments");
    }

    const auto a = array_from_ruby(argv[0], std::nullopt);
    const auto b = array_from_ruby(argv[1], std::nullopt);
    std::optional<mx::array> lhs_indices = std::nullopt;
    std::optional<mx::array> rhs_indices = std::nullopt;

    if (argc >= 3 && !NIL_P(argv[2])) {
      lhs_indices = array_from_ruby(argv[2], std::nullopt);
    }
    if (argc >= 4 && !NIL_P(argv[3])) {
      rhs_indices = array_from_ruby(argv[3], std::nullopt);
    }

    const bool sorted_indices = (argc >= 5) ? RTEST(argv[4]) : false;
    return array_wrap(mx::gather_mm(a, b, lhs_indices, rhs_indices, sorted_indices));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_segmented_mm(VALUE, VALUE a, VALUE b, VALUE segments) {
  try {
    return array_wrap(mx::segmented_mm(
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        array_from_ruby(segments, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_hadamard_transform(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE scale;
    rb_scan_args(argc, argv, "11", &a, &scale);
    std::optional<float> scale_v = NIL_P(scale)
        ? std::nullopt
        : std::make_optional(static_cast<float>(NUM2DBL(scale)));
    return array_wrap(mx::hadamard_transform(array_from_ruby(a, std::nullopt), scale_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_convolve(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE v;
    VALUE mode;
    rb_scan_args(argc, argv, "21", &a, &v, &mode);

    const std::string mode_v = NIL_P(mode) ? "full" : StringValueCStr(mode);
    auto lhs = array_from_ruby(a, std::nullopt);
    auto rhs = array_from_ruby(v, std::nullopt);
    if (lhs.ndim() != 1 || rhs.ndim() != 1) {
      rb_raise(rb_eArgError, "convolve inputs must be 1D");
    }
    if (lhs.size() == 0 || rhs.size() == 0) {
      rb_raise(rb_eArgError, "convolve inputs cannot be empty");
    }

    mx::array in = lhs.size() < rhs.size() ? rhs : lhs;
    mx::array wt = lhs.size() < rhs.size() ? lhs : rhs;

    const int wt_size = static_cast<int>(wt.shape(0));
    wt = mx::slice(wt, {wt_size - 1}, {-wt_size - 1}, {-1});
    in = mx::reshape(in, {1, -1, 1});
    wt = mx::reshape(wt, {1, -1, 1});

    int padding = 0;
    if (mode_v == "full") {
      padding = static_cast<int>(wt.size()) - 1;
    } else if (mode_v == "valid") {
      padding = 0;
    } else if (mode_v == "same") {
      if ((wt.size() % 2) != 0) {
        padding = static_cast<int>(wt.size() / 2);
      } else {
        const int pad_l = static_cast<int>(wt.size() / 2);
        const int pad_r = std::max(0, pad_l - 1);
        in = mx::pad(
            in,
            std::vector<std::pair<int, int>>{{0, 0}, {pad_l, pad_r}, {0, 0}},
            mx::array(0),
            "constant");
      }
    } else {
      rb_raise(rb_eArgError, "convolve mode must be one of: full, valid, same");
    }

    auto out = mx::conv1d(in, wt, 1, padding, 1, 1);
    return array_wrap(mx::reshape(out, {-1}));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv1d(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE dilation;
    VALUE groups;
    rb_scan_args(argc, argv, "24", &input, &weight, &stride, &padding, &dilation, &groups);

    const int stride_v = NIL_P(stride) ? 1 : NUM2INT(stride);
    const int padding_v = NIL_P(padding) ? 0 : NUM2INT(padding);
    const int dilation_v = NIL_P(dilation) ? 1 : NUM2INT(dilation);
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);

    return array_wrap(mx::conv1d(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        stride_v,
        padding_v,
        dilation_v,
        groups_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv2d(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE dilation;
    VALUE groups;
    rb_scan_args(argc, argv, "24", &input, &weight, &stride, &padding, &dilation, &groups);

    auto stride_v = int_pair_from_ruby_or_scalar(stride, {1, 1}, "stride");
    auto padding_v = int_pair_from_ruby_or_scalar(padding, {0, 0}, "padding");
    auto dilation_v = int_pair_from_ruby_or_scalar(dilation, {1, 1}, "dilation");
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);

    return array_wrap(mx::conv2d(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        stride_v,
        padding_v,
        dilation_v,
        groups_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv3d(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE dilation;
    VALUE groups;
    rb_scan_args(argc, argv, "24", &input, &weight, &stride, &padding, &dilation, &groups);

    auto stride_v = int_triple_from_ruby_or_scalar(stride, {1, 1, 1}, "stride");
    auto padding_v = int_triple_from_ruby_or_scalar(padding, {0, 0, 0}, "padding");
    auto dilation_v = int_triple_from_ruby_or_scalar(dilation, {1, 1, 1}, "dilation");
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);

    return array_wrap(mx::conv3d(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        stride_v,
        padding_v,
        dilation_v,
        groups_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv_transpose1d(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE dilation;
    VALUE output_padding;
    VALUE groups;
    rb_scan_args(
        argc,
        argv,
        "25",
        &input,
        &weight,
        &stride,
        &padding,
        &dilation,
        &output_padding,
        &groups);

    const int stride_v = NIL_P(stride) ? 1 : NUM2INT(stride);
    const int padding_v = NIL_P(padding) ? 0 : NUM2INT(padding);
    const int dilation_v = NIL_P(dilation) ? 1 : NUM2INT(dilation);
    const int output_padding_v = NIL_P(output_padding) ? 0 : NUM2INT(output_padding);
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);

    return array_wrap(mx::conv_transpose1d(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        stride_v,
        padding_v,
        dilation_v,
        output_padding_v,
        groups_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv_transpose2d(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE dilation;
    VALUE output_padding;
    VALUE groups;
    rb_scan_args(
        argc,
        argv,
        "25",
        &input,
        &weight,
        &stride,
        &padding,
        &dilation,
        &output_padding,
        &groups);

    auto stride_v = int_pair_from_ruby_or_scalar(stride, {1, 1}, "stride");
    auto padding_v = int_pair_from_ruby_or_scalar(padding, {0, 0}, "padding");
    auto dilation_v = int_pair_from_ruby_or_scalar(dilation, {1, 1}, "dilation");
    auto output_padding_v =
        int_pair_from_ruby_or_scalar(output_padding, {0, 0}, "output_padding");
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);

    return array_wrap(mx::conv_transpose2d(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        stride_v,
        padding_v,
        dilation_v,
        output_padding_v,
        groups_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv_transpose3d(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE dilation;
    VALUE output_padding;
    VALUE groups;
    rb_scan_args(
        argc,
        argv,
        "25",
        &input,
        &weight,
        &stride,
        &padding,
        &dilation,
        &output_padding,
        &groups);

    auto stride_v = int_triple_from_ruby_or_scalar(stride, {1, 1, 1}, "stride");
    auto padding_v = int_triple_from_ruby_or_scalar(padding, {0, 0, 0}, "padding");
    auto dilation_v = int_triple_from_ruby_or_scalar(dilation, {1, 1, 1}, "dilation");
    auto output_padding_v =
        int_triple_from_ruby_or_scalar(output_padding, {0, 0, 0}, "output_padding");
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);

    return array_wrap(mx::conv_transpose3d(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        stride_v,
        padding_v,
        dilation_v,
        output_padding_v,
        groups_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_conv_general(int argc, VALUE* argv, VALUE) {
  try {
    VALUE input;
    VALUE weight;
    VALUE stride;
    VALUE padding;
    VALUE kernel_dilation;
    VALUE input_dilation;
    VALUE groups;
    VALUE flip;
    rb_scan_args(
        argc,
        argv,
        "26",
        &input,
        &weight,
        &stride,
        &padding,
        &kernel_dilation,
        &input_dilation,
        &groups,
        &flip);

    auto stride_v = int_vector_from_ruby_or_scalar(stride, {1}, "stride");
    auto [padding_lo_v, padding_hi_v] = conv_general_padding_from_ruby(padding);
    auto kernel_dilation_v =
        int_vector_from_ruby_or_scalar(kernel_dilation, {1}, "kernel_dilation");
    auto input_dilation_v =
        int_vector_from_ruby_or_scalar(input_dilation, {1}, "input_dilation");
    const int groups_v = NIL_P(groups) ? 1 : NUM2INT(groups);
    const bool flip_v = NIL_P(flip) ? false : RTEST(flip);

    return array_wrap(mx::conv_general(
        array_from_ruby(input, std::nullopt),
        array_from_ruby(weight, std::nullopt),
        std::move(stride_v),
        std::move(padding_lo_v),
        std::move(padding_hi_v),
        std::move(kernel_dilation_v),
        std::move(input_dilation_v),
        groups_v,
        flip_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_quantized_matmul(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 3 || argc > 8) {
      rb_raise(rb_eArgError, "quantized_matmul expects 3 to 8 arguments");
    }

    const auto x = array_from_ruby(argv[0], std::nullopt);
    const auto w = array_from_ruby(argv[1], std::nullopt);
    const auto scales = array_from_ruby(argv[2], std::nullopt);

    std::optional<mx::array> biases = std::nullopt;
    bool transpose = true;
    std::optional<int> group_size = std::nullopt;
    std::optional<int> bits = std::nullopt;
    std::string mode = "affine";

    if (argc >= 4 && !NIL_P(argv[3])) {
      biases = array_from_ruby(argv[3], std::nullopt);
    }
    if (argc >= 5 && !NIL_P(argv[4])) {
      transpose = RTEST(argv[4]);
    }
    if (argc >= 6 && !NIL_P(argv[5])) {
      group_size = NUM2INT(argv[5]);
    }
    if (argc >= 7 && !NIL_P(argv[6])) {
      bits = NUM2INT(argv[6]);
    }
    if (argc >= 8 && !NIL_P(argv[7])) {
      mode = StringValueCStr(argv[7]);
    }

    return array_wrap(mx::quantized_matmul(
        x,
        w,
        scales,
        biases,
        transpose,
        group_size,
        bits,
        mode));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_quantize(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 1 || argc > 4) {
      rb_raise(rb_eArgError, "quantize expects 1 to 4 arguments");
    }

    const auto w = array_from_ruby(argv[0], std::nullopt);
    std::optional<int> group_size = std::nullopt;
    std::optional<int> bits = std::nullopt;
    std::string mode = "affine";

    if (argc >= 2 && !NIL_P(argv[1])) {
      group_size = NUM2INT(argv[1]);
    }
    if (argc >= 3 && !NIL_P(argv[2])) {
      bits = NUM2INT(argv[2]);
    }
    if (argc >= 4 && !NIL_P(argv[3])) {
      mode = StringValueCStr(argv[3]);
    }

    return ruby_array_of_arrays(mx::quantize(w, group_size, bits, mode));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_dequantize(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2 || argc > 7) {
      rb_raise(rb_eArgError, "dequantize expects 2 to 7 arguments");
    }

    const auto w = array_from_ruby(argv[0], std::nullopt);
    const auto scales = array_from_ruby(argv[1], std::nullopt);
    std::optional<mx::array> biases = std::nullopt;
    std::optional<int> group_size = std::nullopt;
    std::optional<int> bits = std::nullopt;
    std::string mode = "affine";
    std::optional<mx::Dtype> dtype = std::nullopt;

    if (argc >= 3 && !NIL_P(argv[2])) {
      biases = array_from_ruby(argv[2], std::nullopt);
    }
    if (argc >= 4 && !NIL_P(argv[3])) {
      group_size = NUM2INT(argv[3]);
    }
    if (argc >= 5 && !NIL_P(argv[4])) {
      bits = NUM2INT(argv[4]);
    }
    if (argc >= 6 && !NIL_P(argv[5])) {
      mode = StringValueCStr(argv[5]);
    }
    if (argc >= 7) {
      dtype = optional_dtype_from_value(argv[6]);
    }

    return array_wrap(mx::dequantize(w, scales, biases, group_size, bits, mode, dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_from_fp8(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE dtype;
    rb_scan_args(argc, argv, "11", &x, &dtype);
    auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::bfloat16);
    return array_wrap(mx::from_fp8(array_from_ruby(x, std::nullopt), target_dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_to_fp8(VALUE, VALUE x) {
  try {
    return array_wrap(mx::to_fp8(array_from_ruby(x, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_qqmm(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2 || argc > 6) {
      rb_raise(rb_eArgError, "qqmm expects 2 to 6 arguments");
    }

    const auto x = array_from_ruby(argv[0], std::nullopt);
    const auto w = array_from_ruby(argv[1], std::nullopt);
    std::optional<mx::array> scales = std::nullopt;
    std::optional<int> group_size = std::nullopt;
    std::optional<int> bits = std::nullopt;
    std::string mode = "nvfp4";

    if (argc >= 3 && !NIL_P(argv[2])) {
      scales = array_from_ruby(argv[2], std::nullopt);
    }
    if (argc >= 4 && !NIL_P(argv[3])) {
      group_size = NUM2INT(argv[3]);
    }
    if (argc >= 5 && !NIL_P(argv[4])) {
      bits = NUM2INT(argv[4]);
    }
    if (argc >= 6 && !NIL_P(argv[5])) {
      mode = StringValueCStr(argv[5]);
    }

    return array_wrap(mx::qqmm(x, w, scales, group_size, bits, mode));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_gather_qmm(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 3 || argc > 11) {
      rb_raise(rb_eArgError, "gather_qmm expects 3 to 11 arguments");
    }

    const auto x = array_from_ruby(argv[0], std::nullopt);
    const auto w = array_from_ruby(argv[1], std::nullopt);
    const auto scales = array_from_ruby(argv[2], std::nullopt);

    std::optional<mx::array> biases = std::nullopt;
    std::optional<mx::array> lhs_indices = std::nullopt;
    std::optional<mx::array> rhs_indices = std::nullopt;
    bool transpose = true;
    std::optional<int> group_size = std::nullopt;
    std::optional<int> bits = std::nullopt;
    std::string mode = "affine";
    bool sorted_indices = false;

    if (argc >= 4 && !NIL_P(argv[3])) {
      biases = array_from_ruby(argv[3], std::nullopt);
    }
    if (argc >= 5 && !NIL_P(argv[4])) {
      lhs_indices = array_from_ruby(argv[4], std::nullopt);
    }
    if (argc >= 6 && !NIL_P(argv[5])) {
      rhs_indices = array_from_ruby(argv[5], std::nullopt);
    }
    if (argc >= 7 && !NIL_P(argv[6])) {
      transpose = RTEST(argv[6]);
    }
    if (argc >= 8 && !NIL_P(argv[7])) {
      group_size = NUM2INT(argv[7]);
    }
    if (argc >= 9 && !NIL_P(argv[8])) {
      bits = NUM2INT(argv[8]);
    }
    if (argc >= 10 && !NIL_P(argv[9])) {
      mode = StringValueCStr(argv[9]);
    }
    if (argc >= 11 && !NIL_P(argv[10])) {
      sorted_indices = RTEST(argv[10]);
    }

    return array_wrap(mx::gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        transpose,
        group_size,
        bits,
        mode,
        sorted_indices));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_depends(VALUE, VALUE inputs, VALUE dependencies) {
  try {
    bool return_scalar = false;
    auto input_arrays = array_sequence_from_ruby(inputs, &return_scalar);
    auto dep_arrays = array_sequence_from_ruby(dependencies, nullptr);
    auto out = mx::depends(input_arrays, dep_arrays);
    if (return_scalar) {
      return array_wrap(out.at(0));
    }
    return ruby_array_of_arrays(out);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_save(VALUE, VALUE file, VALUE array) {
  try {
    mx::save(string_from_ruby(file), array_unwrap(array));
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static std::string infer_load_format(const std::string& file) {
  auto dot = file.find_last_of('.');
  if (dot == std::string::npos || dot == file.size() - 1) {
    rb_raise(rb_eArgError, "could not infer load format from file extension");
  }
  return file.substr(dot + 1);
}

static VALUE core_load(int argc, VALUE* argv, VALUE) {
  try {
    VALUE file;
    VALUE format;
    VALUE return_metadata;
    rb_scan_args(argc, argv, "12", &file, &format, &return_metadata);

    const std::string file_v = string_from_ruby(file);
    std::string format_v = NIL_P(format) ? infer_load_format(file_v) : string_from_ruby(format);
    const bool return_metadata_v = NIL_P(return_metadata) ? false : RTEST(return_metadata);

    if (format_v == "npy") {
      if (return_metadata_v) {
        rb_raise(rb_eArgError, "metadata not supported for format npy");
      }
      return array_wrap(mx::load(file_v));
    }
    if (format_v == "npz") {
      rb_raise(rb_eNotImpError, "npz load is not yet supported in the Ruby binding");
    }
    if (format_v == "safetensors") {
      auto [arrays, metadata] = mx::load_safetensors(file_v);
      VALUE ruby_arrays = ruby_hash_of_arrays(arrays);
      if (!return_metadata_v) {
        return ruby_arrays;
      }
      VALUE out = rb_ary_new_capa(2);
      rb_ary_push(out, ruby_arrays);
      rb_ary_push(out, ruby_hash_of_strings(metadata));
      return out;
    }
    if (format_v == "gguf") {
      auto [arrays, metadata] = mx::load_gguf(file_v);
      VALUE ruby_arrays = ruby_hash_of_arrays(arrays);
      if (!return_metadata_v) {
        return ruby_arrays;
      }
      VALUE out = rb_ary_new_capa(2);
      rb_ary_push(out, ruby_arrays);
      rb_ary_push(out, ruby_hash_of_gguf_metadata(metadata));
      return out;
    }

    rb_raise(rb_eArgError, "unknown load format");
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_save_safetensors(int argc, VALUE* argv, VALUE) {
  try {
    VALUE file;
    VALUE arrays;
    VALUE metadata;
    rb_scan_args(argc, argv, "21", &file, &arrays, &metadata);

    mx::save_safetensors(
        string_from_ruby(file),
        array_map_from_ruby_hash(arrays),
        string_map_from_ruby_hash(metadata));
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_save_gguf(int argc, VALUE* argv, VALUE) {
  try {
    VALUE file;
    VALUE arrays;
    VALUE metadata;
    rb_scan_args(argc, argv, "21", &file, &arrays, &metadata);

    mx::save_gguf(
        string_from_ruby(file),
        array_map_from_ruby_hash(arrays),
        gguf_meta_map_from_ruby_hash(metadata));
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_savez(int, VALUE*, VALUE) {
  rb_raise(rb_eNotImpError, "savez is not yet supported in the Ruby binding");
  return Qnil;
}

static VALUE core_savez_compressed(int, VALUE*, VALUE) {
  rb_raise(
      rb_eNotImpError,
      "savez_compressed is not yet supported in the Ruby binding");
  return Qnil;
}

static VALUE core_inner(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::inner(array_unwrap(a), array_unwrap(b)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_outer(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::outer(array_unwrap(a), array_unwrap(b)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tensordot(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE axes;
    rb_scan_args(argc, argv, "21", &a, &b, &axes);

    auto lhs = array_unwrap(a);
    auto rhs = array_unwrap(b);
    if (NIL_P(axes)) {
      return array_wrap(mx::tensordot(lhs, rhs, 2));
    }
    if (RB_INTEGER_TYPE_P(axes)) {
      return array_wrap(mx::tensordot(lhs, rhs, NUM2INT(axes)));
    }
    if (!RB_TYPE_P(axes, T_ARRAY) || RARRAY_LEN(axes) != 2) {
      rb_raise(rb_eTypeError, "axes must be an integer or [lhs_axes, rhs_axes]");
    }
    VALUE lhs_axes = rb_ary_entry(axes, 0);
    VALUE rhs_axes = rb_ary_entry(axes, 1);
    return array_wrap(mx::tensordot(lhs, rhs, int_vector_from_ruby(lhs_axes), int_vector_from_ruby(rhs_axes)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_einsum(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2) {
      rb_raise(rb_eArgError, "einsum expects a subscripts string and at least one operand");
    }

    const std::string subscripts = StringValueCStr(argv[0]);
    std::vector<mx::array> operands;
    operands.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; ++i) {
      operands.push_back(array_unwrap(argv[i]));
    }
    return array_wrap(mx::einsum(subscripts, operands));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_einsum_path(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2) {
      rb_raise(rb_eArgError, "einsum_path expects a subscripts string and at least one operand");
    }

    const std::string subscripts = StringValueCStr(argv[0]);
    std::vector<mx::array> operands;
    operands.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; ++i) {
      operands.push_back(array_unwrap(argv[i]));
    }

    auto [path, summary] = mx::einsum_path(subscripts, operands);
    VALUE ruby_path = rb_ary_new_capa(static_cast<long>(path.size()));
    for (const auto& step : path) {
      VALUE item = rb_ary_new_capa(static_cast<long>(step.size()));
      for (int index : step) {
        rb_ary_push(item, INT2NUM(index));
      }
      rb_ary_push(ruby_path, item);
    }

    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, ruby_path);
    rb_ary_push(out, rb_utf8_str_new(summary.c_str(), static_cast<long>(summary.size())));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_kron(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::kron(array_unwrap(a), array_unwrap(b)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_diagonal(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE offset;
    VALUE axis1;
    VALUE axis2;
    rb_scan_args(argc, argv, "13", &array, &offset, &axis1, &axis2);

    const int offset_v = NIL_P(offset) ? 0 : NUM2INT(offset);
    const int axis1_v = NIL_P(axis1) ? 0 : NUM2INT(axis1);
    const int axis2_v = NIL_P(axis2) ? 1 : NUM2INT(axis2);
    return array_wrap(mx::diagonal(array_unwrap(array), offset_v, axis1_v, axis2_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_diag(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE k;
    rb_scan_args(argc, argv, "11", &array, &k);
    const int k_v = NIL_P(k) ? 0 : NUM2INT(k);
    return array_wrap(mx::diag(array_unwrap(array), k_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_trace(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE offset;
    VALUE axis1;
    VALUE axis2;
    rb_scan_args(argc, argv, "13", &array, &offset, &axis1, &axis2);
    auto a = array_unwrap(array);

    if (NIL_P(offset) && NIL_P(axis1) && NIL_P(axis2)) {
      return array_wrap(mx::trace(a));
    }

    const int offset_v = NIL_P(offset) ? 0 : NUM2INT(offset);
    const int axis1_v = NIL_P(axis1) ? 0 : NUM2INT(axis1);
    const int axis2_v = NIL_P(axis2) ? 1 : NUM2INT(axis2);
    return array_wrap(mx::trace(a, offset_v, axis1_v, axis2_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_broadcast_to(VALUE, VALUE array, VALUE shape) {
  try {
    return array_wrap(mx::broadcast_to(array_unwrap(array), shape_from_ruby(shape)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_broadcast_arrays(VALUE, VALUE arrays) {
  try {
    auto values = array_vector_from_ruby(arrays);
    return ruby_array_of_arrays(mx::broadcast_arrays(values));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_reshape(VALUE, VALUE array, VALUE shape) {
  try {
    return array_wrap(mx::reshape(array_unwrap(array), shape_from_ruby(shape)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_flatten(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE start_axis;
    VALUE end_axis;
    rb_scan_args(argc, argv, "12", &array, &start_axis, &end_axis);

    auto a = array_unwrap(array);
    if (NIL_P(start_axis) && NIL_P(end_axis)) {
      return array_wrap(mx::flatten(a));
    }

    const int start_v = NIL_P(start_axis) ? 0 : NUM2INT(start_axis);
    const int end_v = NIL_P(end_axis) ? -1 : NUM2INT(end_axis);
    return array_wrap(mx::flatten(a, start_v, end_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_transpose(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axes;
    rb_scan_args(argc, argv, "11", &array, &axes);
    auto a = array_unwrap(array);
    if (NIL_P(axes)) {
      return array_wrap(mx::transpose(a));
    }
    return array_wrap(mx::transpose(a, int_vector_from_ruby(axes)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_permute_dims(int argc, VALUE* argv, VALUE self) {
  return core_transpose(argc, argv, self);
}

static VALUE core_squeeze(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &array, &axis);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      return array_wrap(mx::squeeze(a));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::squeeze(a, NUM2INT(axis)));
    }
    return array_wrap(mx::squeeze(a, int_vector_from_ruby(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_expand_dims(VALUE, VALUE array, VALUE axis) {
  try {
    auto a = array_unwrap(array);
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::expand_dims(a, NUM2INT(axis)));
    }
    return array_wrap(mx::expand_dims(a, int_vector_from_ruby(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_atleast_1d(VALUE, VALUE array) {
  try {
    return array_wrap(mx::atleast_1d(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_atleast_2d(VALUE, VALUE array) {
  try {
    return array_wrap(mx::atleast_2d(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_atleast_3d(VALUE, VALUE array) {
  try {
    return array_wrap(mx::atleast_3d(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_moveaxis(VALUE, VALUE array, VALUE source, VALUE destination) {
  try {
    return array_wrap(mx::moveaxis(array_unwrap(array), NUM2INT(source), NUM2INT(destination)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_swapaxes(VALUE, VALUE array, VALUE axis1, VALUE axis2) {
  try {
    return array_wrap(mx::swapaxes(array_unwrap(array), NUM2INT(axis1), NUM2INT(axis2)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sum(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &array, &axis);
    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::sum(a));
    }
    return array_wrap(mx::sum(a, NUM2INT(axis), false));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_mean(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &array, &axis);
    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::mean(a));
    }
    return array_wrap(mx::mean(a, NUM2INT(axis), false));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_all(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::all(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::all(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::all(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_any(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::any(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::any(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::any(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_softmax(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE precise;
    rb_scan_args(argc, argv, "12", &array, &axis, &precise);
    const bool precise_v = NIL_P(precise) ? false : RTEST(precise);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      return array_wrap(mx::softmax(a, precise_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::softmax(a, NUM2INT(axis), precise_v));
    }
    return array_wrap(mx::softmax(a, int_vector_from_ruby(axis), precise_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sort(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &array, &axis);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::sort(a));
    }
    return array_wrap(mx::sort(a, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_argsort(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    rb_scan_args(argc, argv, "11", &array, &axis);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::argsort(a));
    }
    return array_wrap(mx::argsort(a, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_topk(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE k;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &k, &axis);

    auto a = array_unwrap(array);
    const int k_v = NUM2INT(k);
    if (NIL_P(axis)) {
      return array_wrap(mx::topk(a, k_v));
    }
    return array_wrap(mx::topk(a, k_v, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_partition(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE kth;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &kth, &axis);

    auto a = array_unwrap(array);
    const int kth_v = NUM2INT(kth);
    if (NIL_P(axis)) {
      return array_wrap(mx::partition(a, kth_v));
    }
    return array_wrap(mx::partition(a, kth_v, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_argpartition(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE kth;
    VALUE axis;
    rb_scan_args(argc, argv, "21", &array, &kth, &axis);

    auto a = array_unwrap(array);
    const int kth_v = NUM2INT(kth);
    if (NIL_P(axis)) {
      return array_wrap(mx::argpartition(a, kth_v));
    }
    return array_wrap(mx::argpartition(a, kth_v, NUM2INT(axis)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_max(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::max(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::max(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::max(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_min(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::min(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::min(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::min(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_argmax(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::argmax(a, keepdims_v));
    }
    if (!RB_INTEGER_TYPE_P(axis)) {
      rb_raise(rb_eTypeError, "axis must be an integer for argmax");
    }
    return array_wrap(mx::argmax(a, NUM2INT(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_argmin(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::argmin(a, keepdims_v));
    }
    if (!RB_INTEGER_TYPE_P(axis)) {
      rb_raise(rb_eTypeError, "axis must be an integer for argmin");
    }
    return array_wrap(mx::argmin(a, NUM2INT(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_prod(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);

    auto a = array_unwrap(array);
    if (NIL_P(axis)) {
      return array_wrap(mx::prod(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::prod(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::prod(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cumsum(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE reverse;
    VALUE inclusive;
    rb_scan_args(argc, argv, "13", &array, &axis, &reverse, &inclusive);

    const bool reverse_v = NIL_P(reverse) ? false : RTEST(reverse);
    const bool inclusive_v = NIL_P(inclusive) ? true : RTEST(inclusive);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      return array_wrap(mx::cumsum(flat, 0, reverse_v, inclusive_v));
    }
    return array_wrap(mx::cumsum(a, NUM2INT(axis), reverse_v, inclusive_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cumprod(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE reverse;
    VALUE inclusive;
    rb_scan_args(argc, argv, "13", &array, &axis, &reverse, &inclusive);

    const bool reverse_v = NIL_P(reverse) ? false : RTEST(reverse);
    const bool inclusive_v = NIL_P(inclusive) ? true : RTEST(inclusive);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      return array_wrap(mx::cumprod(flat, 0, reverse_v, inclusive_v));
    }
    return array_wrap(mx::cumprod(a, NUM2INT(axis), reverse_v, inclusive_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cummax(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE reverse;
    VALUE inclusive;
    rb_scan_args(argc, argv, "13", &array, &axis, &reverse, &inclusive);

    const bool reverse_v = NIL_P(reverse) ? false : RTEST(reverse);
    const bool inclusive_v = NIL_P(inclusive) ? true : RTEST(inclusive);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      return array_wrap(mx::cummax(flat, 0, reverse_v, inclusive_v));
    }
    return array_wrap(mx::cummax(a, NUM2INT(axis), reverse_v, inclusive_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cummin(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE reverse;
    VALUE inclusive;
    rb_scan_args(argc, argv, "13", &array, &axis, &reverse, &inclusive);

    const bool reverse_v = NIL_P(reverse) ? false : RTEST(reverse);
    const bool inclusive_v = NIL_P(inclusive) ? true : RTEST(inclusive);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      return array_wrap(mx::cummin(flat, 0, reverse_v, inclusive_v));
    }
    return array_wrap(mx::cummin(a, NUM2INT(axis), reverse_v, inclusive_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_var(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    VALUE ddof;
    rb_scan_args(argc, argv, "13", &array, &axis, &keepdims, &ddof);

    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);
    const int ddof_v = NIL_P(ddof) ? 0 : NUM2INT(ddof);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      return array_wrap(mx::var(a, keepdims_v, ddof_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::var(a, NUM2INT(axis), keepdims_v, ddof_v));
    }
    return array_wrap(mx::var(a, int_vector_from_ruby(axis), keepdims_v, ddof_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_std(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    VALUE ddof;
    rb_scan_args(argc, argv, "13", &array, &axis, &keepdims, &ddof);

    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);
    const int ddof_v = NIL_P(ddof) ? 0 : NUM2INT(ddof);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      return array_wrap(mx::std(a, keepdims_v, ddof_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::std(a, NUM2INT(axis), keepdims_v, ddof_v));
    }
    return array_wrap(mx::std(a, int_vector_from_ruby(axis), keepdims_v, ddof_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_median(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);

    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      return array_wrap(mx::median(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::median(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::median(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_random_seed(VALUE, VALUE seed) {
  try {
    mx::random::seed(NUM2ULL(seed));
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_random_uniform(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE low;
    VALUE high;
    VALUE dtype;
    rb_scan_args(argc, argv, "13", &shape, &low, &high, &dtype);

    auto target_shape = shape_from_ruby(shape);
    auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);

    if (NIL_P(low) && NIL_P(high)) {
      return array_wrap(mx::random::uniform(target_shape, target_dtype));
    }

    const double lo = NIL_P(low) ? 0.0 : NUM2DBL(low);
    const double hi = NIL_P(high) ? 1.0 : NUM2DBL(high);
    return array_wrap(mx::random::uniform(lo, hi, target_shape, target_dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_seed(VALUE, VALUE seed) {
  try {
    mx::random::seed(NUM2ULL(seed));
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_key(VALUE, VALUE seed) {
  try {
    return array_wrap(mx::random::key(NUM2ULL(seed)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_random_split(int argc, VALUE* argv, VALUE) {
  try {
    VALUE key;
    VALUE num;
    rb_scan_args(argc, argv, "11", &key, &num);
    auto key_v = array_from_ruby(key, std::nullopt);

    if (NIL_P(num)) {
      auto [k1, k2] = mx::random::split(key_v);
      VALUE out = rb_ary_new_capa(2);
      rb_ary_push(out, array_wrap(k1));
      rb_ary_push(out, array_wrap(k2));
      return out;
    }
    return array_wrap(mx::random::split(key_v, NUM2INT(num)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_uniform(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE low;
    VALUE high;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "14", &shape, &low, &high, &dtype, &key);

    const auto target_shape = shape_from_ruby(shape);
    const auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);
    const auto key_v = optional_array_from_value(key);

    if (NIL_P(low) && NIL_P(high)) {
      return array_wrap(mx::random::uniform(target_shape, target_dtype, key_v));
    }

    auto low_v = array_from_ruby(NIL_P(low) ? DBL2NUM(0.0) : low, std::nullopt);
    auto high_v = array_from_ruby(NIL_P(high) ? DBL2NUM(1.0) : high, std::nullopt);
    return array_wrap(mx::random::uniform(low_v, high_v, target_shape, target_dtype, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_normal(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE loc;
    VALUE scale;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "14", &shape, &loc, &scale, &dtype, &key);

    const auto target_shape = shape_from_ruby(shape);
    const auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);
    const float loc_v = NIL_P(loc) ? 0.0f : static_cast<float>(NUM2DBL(loc));
    const float scale_v = NIL_P(scale) ? 1.0f : static_cast<float>(NUM2DBL(scale));
    const auto key_v = optional_array_from_value(key);
    return array_wrap(mx::random::normal(target_shape, target_dtype, loc_v, scale_v, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_randint(int argc, VALUE* argv, VALUE) {
  try {
    VALUE low;
    VALUE high;
    VALUE shape;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "32", &low, &high, &shape, &dtype, &key);

    const auto target_shape = shape_from_ruby(shape);
    const auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::int32);
    const auto key_v = optional_array_from_value(key);
    return array_wrap(mx::random::randint(
        array_from_ruby(low, std::nullopt),
        array_from_ruby(high, std::nullopt),
        target_shape,
        target_dtype,
        key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_bernoulli(int argc, VALUE* argv, VALUE) {
  try {
    VALUE p;
    VALUE shape;
    VALUE key;
    rb_scan_args(argc, argv, "03", &p, &shape, &key);
    const auto key_v = optional_array_from_value(key);

    if (NIL_P(p)) {
      return array_wrap(mx::random::bernoulli(key_v));
    }
    auto p_v = array_from_ruby(p, std::nullopt);
    if (NIL_P(shape)) {
      return array_wrap(mx::random::bernoulli(p_v, key_v));
    }
    return array_wrap(mx::random::bernoulli(p_v, shape_from_ruby(shape), key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_truncated_normal(int argc, VALUE* argv, VALUE) {
  try {
    VALUE lower;
    VALUE upper;
    VALUE shape;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "23", &lower, &upper, &shape, &dtype, &key);

    const auto lower_v = array_from_ruby(lower, std::nullopt);
    const auto upper_v = array_from_ruby(upper, std::nullopt);
    const auto dtype_v = optional_dtype_from_value(dtype).value_or(mx::float32);
    const auto key_v = optional_array_from_value(key);

    if (NIL_P(shape)) {
      return array_wrap(mx::random::truncated_normal(lower_v, upper_v, dtype_v, key_v));
    }
    return array_wrap(
        mx::random::truncated_normal(lower_v, upper_v, shape_from_ruby(shape), dtype_v, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_gumbel(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "12", &shape, &dtype, &key);
    auto shape_v = shape_from_ruby(shape);
    auto dtype_v = optional_dtype_from_value(dtype).value_or(mx::float32);
    auto key_v = optional_array_from_value(key);
    return array_wrap(mx::random::gumbel(shape_v, dtype_v, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_categorical(int argc, VALUE* argv, VALUE) {
  try {
    VALUE logits;
    VALUE axis;
    VALUE shape_or_num;
    VALUE key;
    rb_scan_args(argc, argv, "13", &logits, &axis, &shape_or_num, &key);
    const auto logits_v = array_from_ruby(logits, std::nullopt);
    const int axis_v = NIL_P(axis) ? -1 : NUM2INT(axis);
    const auto key_v = optional_array_from_value(key);

    if (NIL_P(shape_or_num)) {
      return array_wrap(mx::random::categorical(logits_v, axis_v, key_v));
    }
    if (RB_INTEGER_TYPE_P(shape_or_num)) {
      return array_wrap(mx::random::categorical(logits_v, axis_v, NUM2INT(shape_or_num), key_v));
    }
    return array_wrap(
        mx::random::categorical(logits_v, axis_v, shape_from_ruby(shape_or_num), key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_laplace(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE loc;
    VALUE scale;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "14", &shape, &loc, &scale, &dtype, &key);
    auto shape_v = shape_from_ruby(shape);
    auto dtype_v = optional_dtype_from_value(dtype).value_or(mx::float32);
    const float loc_v = NIL_P(loc) ? 0.0f : static_cast<float>(NUM2DBL(loc));
    const float scale_v = NIL_P(scale) ? 1.0f : static_cast<float>(NUM2DBL(scale));
    auto key_v = optional_array_from_value(key);
    return array_wrap(mx::random::laplace(shape_v, dtype_v, loc_v, scale_v, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_permutation(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE axis;
    VALUE key;
    rb_scan_args(argc, argv, "12", &x, &axis, &key);
    auto key_v = optional_array_from_value(key);
    if (RB_INTEGER_TYPE_P(x)) {
      return array_wrap(mx::random::permutation(NUM2INT(x), key_v));
    }
    int axis_v = NIL_P(axis) ? 0 : NUM2INT(axis);
    return array_wrap(mx::random::permutation(array_from_ruby(x, std::nullopt), axis_v, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_multivariate_normal(int argc, VALUE* argv, VALUE) {
  try {
    VALUE mean;
    VALUE cov;
    VALUE shape;
    VALUE dtype;
    VALUE key;
    rb_scan_args(argc, argv, "23", &mean, &cov, &shape, &dtype, &key);

    auto mean_v = array_from_ruby(mean, std::nullopt);
    auto cov_v = array_from_ruby(cov, std::nullopt);
    auto shape_v = NIL_P(shape) ? mx::Shape{} : shape_from_ruby(shape);
    auto dtype_v = optional_dtype_from_value(dtype).value_or(mean_v.dtype());
    auto key_v = optional_array_from_value(key);
    return array_wrap(mx::random::multivariate_normal(mean_v, cov_v, shape_v, dtype_v, key_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_fft(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axis;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axis, &stream);
    const int axis_v = NIL_P(axis) ? -1 : NUM2INT(axis);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n)) {
      return array_wrap(mxfft::fft(a_v, axis_v, stream_v));
    }
    return array_wrap(mxfft::fft(a_v, NUM2INT(n), axis_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ifft(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axis;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axis, &stream);
    const int axis_v = NIL_P(axis) ? -1 : NUM2INT(axis);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n)) {
      return array_wrap(mxfft::ifft(a_v, axis_v, stream_v));
    }
    return array_wrap(mxfft::ifft(a_v, NUM2INT(n), axis_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_fft2(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    auto axes_v = NIL_P(axes) ? std::vector<int>{-2, -1} : int_vector_from_ruby(axes);
    if (NIL_P(n)) {
      return array_wrap(mxfft::fftn(a_v, axes_v, stream_v));
    }
    return array_wrap(mxfft::fftn(a_v, shape_from_ruby(n), axes_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ifft2(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    auto axes_v = NIL_P(axes) ? std::vector<int>{-2, -1} : int_vector_from_ruby(axes);
    if (NIL_P(n)) {
      return array_wrap(mxfft::ifftn(a_v, axes_v, stream_v));
    }
    return array_wrap(mxfft::ifftn(a_v, shape_from_ruby(n), axes_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_fftn(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n) && NIL_P(axes)) {
      return array_wrap(mxfft::fftn(a_v, stream_v));
    }
    if (NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(mxfft::fftn(a_v, int_vector_from_ruby(axes), stream_v));
    }
    if (!NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(
          mxfft::fftn(a_v, shape_from_ruby(n), int_vector_from_ruby(axes), stream_v));
    }
    rb_raise(rb_eArgError, "fftn requires axes when n is provided");
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ifftn(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n) && NIL_P(axes)) {
      return array_wrap(mxfft::ifftn(a_v, stream_v));
    }
    if (NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(mxfft::ifftn(a_v, int_vector_from_ruby(axes), stream_v));
    }
    if (!NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(
          mxfft::ifftn(a_v, shape_from_ruby(n), int_vector_from_ruby(axes), stream_v));
    }
    rb_raise(rb_eArgError, "ifftn requires axes when n is provided");
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_rfft(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axis;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axis, &stream);
    const int axis_v = NIL_P(axis) ? -1 : NUM2INT(axis);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n)) {
      return array_wrap(mxfft::rfft(a_v, axis_v, stream_v));
    }
    return array_wrap(mxfft::rfft(a_v, NUM2INT(n), axis_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_irfft(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axis;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axis, &stream);
    const int axis_v = NIL_P(axis) ? -1 : NUM2INT(axis);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n)) {
      return array_wrap(mxfft::irfft(a_v, axis_v, stream_v));
    }
    return array_wrap(mxfft::irfft(a_v, NUM2INT(n), axis_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_rfft2(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    auto axes_v = NIL_P(axes) ? std::vector<int>{-2, -1} : int_vector_from_ruby(axes);
    if (NIL_P(n)) {
      return array_wrap(mxfft::rfftn(a_v, axes_v, stream_v));
    }
    return array_wrap(mxfft::rfftn(a_v, shape_from_ruby(n), axes_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_irfft2(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    auto axes_v = NIL_P(axes) ? std::vector<int>{-2, -1} : int_vector_from_ruby(axes);
    if (NIL_P(n)) {
      return array_wrap(mxfft::irfftn(a_v, axes_v, stream_v));
    }
    return array_wrap(mxfft::irfftn(a_v, shape_from_ruby(n), axes_v, stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_rfftn(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n) && NIL_P(axes)) {
      return array_wrap(mxfft::rfftn(a_v, stream_v));
    }
    if (NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(mxfft::rfftn(a_v, int_vector_from_ruby(axes), stream_v));
    }
    if (!NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(
          mxfft::rfftn(a_v, shape_from_ruby(n), int_vector_from_ruby(axes), stream_v));
    }
    rb_raise(rb_eArgError, "rfftn requires axes when n is provided");
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_irfftn(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE n;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &a, &n, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(n) && NIL_P(axes)) {
      return array_wrap(mxfft::irfftn(a_v, stream_v));
    }
    if (NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(mxfft::irfftn(a_v, int_vector_from_ruby(axes), stream_v));
    }
    if (!NIL_P(n) && !NIL_P(axes)) {
      return array_wrap(
          mxfft::irfftn(a_v, shape_from_ruby(n), int_vector_from_ruby(axes), stream_v));
    }
    rb_raise(rb_eArgError, "irfftn requires axes when n is provided");
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_fftshift(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(axes)) {
      return array_wrap(mxfft::fftshift(a_v, stream_v));
    }
    return array_wrap(mxfft::fftshift(a_v, int_vector_from_ruby(axes), stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ifftshift(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE axes;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &axes, &stream);
    auto a_v = array_from_ruby(a, std::nullopt);
    auto stream_v = stream_or_device_from_value(stream);
    if (NIL_P(axes)) {
      return array_wrap(mxfft::ifftshift(a_v, stream_v));
    }
    return array_wrap(mxfft::ifftshift(a_v, int_vector_from_ruby(axes), stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_norm(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE ord;
    VALUE axis;
    VALUE keepdims;
    VALUE stream;
    rb_scan_args(argc, argv, "14", &a, &ord, &axis, &keepdims, &stream);

    auto a_v = array_from_ruby(a, std::nullopt);
    auto axis_v = optional_axis_vector_from_value(axis);
    const bool keepdims_v = RTEST(keepdims);
    auto stream_v = stream_or_device_from_value(stream);

    if (NIL_P(ord)) {
      return array_wrap(mxlinalg::norm(a_v, axis_v, keepdims_v, stream_v));
    }
    if (SYMBOL_P(ord) || RB_TYPE_P(ord, T_STRING)) {
      return array_wrap(
          mxlinalg::norm(a_v, string_from_ruby(ord), axis_v, keepdims_v, stream_v));
    }
    if (RB_INTEGER_TYPE_P(ord) || RB_FLOAT_TYPE_P(ord)) {
      return array_wrap(mxlinalg::norm(a_v, NUM2DBL(ord), axis_v, keepdims_v, stream_v));
    }

    rb_raise(rb_eTypeError, "ord must be nil, integer, float, symbol, or string");
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_qr(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    auto stream_v = stream_or_device_from_value(stream);
    auto result = mxlinalg::qr(array_from_ruby(a, std::nullopt), stream_v);
    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, array_wrap(result.first));
    rb_ary_push(out, array_wrap(result.second));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_svd(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE compute_uv;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &compute_uv, &stream);
    const bool compute_uv_v = NIL_P(compute_uv) ? true : RTEST(compute_uv);
    auto stream_v = stream_or_device_from_value(stream);
    auto result =
        mxlinalg::svd(array_from_ruby(a, std::nullopt), compute_uv_v, stream_v);
    if (result.size() == 1) {
      return array_wrap(result.at(0));
    }
    return ruby_array_of_arrays(result);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_inv(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    return array_wrap(
        mxlinalg::inv(array_from_ruby(a, std::nullopt), stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tri_inv(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE upper;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &upper, &stream);
    const bool upper_v = RTEST(upper);
    return array_wrap(mxlinalg::tri_inv(
        array_from_ruby(a, std::nullopt), upper_v, stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cholesky(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE upper;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &upper, &stream);
    const bool upper_v = RTEST(upper);
    return array_wrap(mxlinalg::cholesky(
        array_from_ruby(a, std::nullopt), upper_v, stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cholesky_inv(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE upper;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &upper, &stream);
    const bool upper_v = RTEST(upper);
    return array_wrap(mxlinalg::cholesky_inv(
        array_from_ruby(a, std::nullopt), upper_v, stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_pinv(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    return array_wrap(
        mxlinalg::pinv(array_from_ruby(a, std::nullopt), stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_lu(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    auto result =
        mxlinalg::lu(array_from_ruby(a, std::nullopt), stream_or_device_from_value(stream));
    return ruby_array_of_arrays(result);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_lu_factor(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    auto result = mxlinalg::lu_factor(
        array_from_ruby(a, std::nullopt), stream_or_device_from_value(stream));
    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, array_wrap(result.first));
    rb_ary_push(out, array_wrap(result.second));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_solve(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE stream;
    rb_scan_args(argc, argv, "21", &a, &b, &stream);
    return array_wrap(mxlinalg::solve(
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_solve_triangular(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE upper;
    VALUE stream;
    rb_scan_args(argc, argv, "22", &a, &b, &upper, &stream);
    const bool upper_v = RTEST(upper);
    return array_wrap(mxlinalg::solve_triangular(
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        upper_v,
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cross(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE axis;
    VALUE stream;
    rb_scan_args(argc, argv, "22", &a, &b, &axis, &stream);
    const int axis_v = NIL_P(axis) ? -1 : NUM2INT(axis);
    return array_wrap(mxlinalg::cross(
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        axis_v,
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_eigvals(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    return array_wrap(mxlinalg::eigvals(
        array_from_ruby(a, std::nullopt), stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_eig(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE stream;
    rb_scan_args(argc, argv, "11", &a, &stream);
    auto result =
        mxlinalg::eig(array_from_ruby(a, std::nullopt), stream_or_device_from_value(stream));
    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, array_wrap(result.first));
    rb_ary_push(out, array_wrap(result.second));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_eigvalsh(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE uplo;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &uplo, &stream);
    std::string uplo_v = NIL_P(uplo) ? "L" : string_from_ruby(uplo);
    return array_wrap(mxlinalg::eigvalsh(
        array_from_ruby(a, std::nullopt), std::move(uplo_v), stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_eigh(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE uplo;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &a, &uplo, &stream);
    std::string uplo_v = NIL_P(uplo) ? "L" : string_from_ruby(uplo);
    auto result = mxlinalg::eigh(
        array_from_ruby(a, std::nullopt), std::move(uplo_v), stream_or_device_from_value(stream));
    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, array_wrap(result.first));
    rb_ary_push(out, array_wrap(result.second));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_rms_norm(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE weight;
    VALUE eps;
    VALUE stream;
    rb_scan_args(argc, argv, "13", &x, &weight, &eps, &stream);
    const float eps_v = NIL_P(eps) ? 1e-5f : static_cast<float>(NUM2DBL(eps));
    auto weight_v = optional_array_from_value(weight);
    return array_wrap(mxfast::rms_norm(
        array_from_ruby(x, std::nullopt), weight_v, eps_v, stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_layer_norm(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE weight;
    VALUE bias;
    VALUE eps;
    VALUE stream;
    rb_scan_args(argc, argv, "14", &x, &weight, &bias, &eps, &stream);
    const float eps_v = NIL_P(eps) ? 1e-5f : static_cast<float>(NUM2DBL(eps));
    auto weight_v = optional_array_from_value(weight);
    auto bias_v = optional_array_from_value(bias);
    return array_wrap(
        mxfast::layer_norm(
            array_from_ruby(x, std::nullopt),
            weight_v,
            bias_v,
            eps_v,
            stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_rope(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE dims;
    VALUE traditional;
    VALUE base;
    VALUE scale;
    VALUE offset;
    VALUE freqs;
    VALUE stream;
    rb_scan_args(
        argc, argv, "26", &x, &dims, &traditional, &base, &scale, &offset, &freqs, &stream);

    auto x_v = array_from_ruby(x, std::nullopt);
    const int dims_v = NUM2INT(dims);
    const bool traditional_v = RTEST(traditional);
    auto base_v = NIL_P(base)
        ? std::optional<float>{}
        : std::optional<float>{static_cast<float>(NUM2DBL(base))};
    const float scale_v = NIL_P(scale) ? 1.0f : static_cast<float>(NUM2DBL(scale));
    auto freqs_v = optional_array_from_value(freqs);
    auto stream_v = stream_or_device_from_value(stream);

    if (NIL_P(offset) || RB_INTEGER_TYPE_P(offset)) {
      const int offset_v = NIL_P(offset) ? 0 : NUM2INT(offset);
      return array_wrap(
          mxfast::rope(x_v, dims_v, traditional_v, base_v, scale_v, offset_v, freqs_v, stream_v));
    }

    return array_wrap(mxfast::rope(
        x_v,
        dims_v,
        traditional_v,
        base_v,
        scale_v,
        array_from_ruby(offset, std::nullopt),
        freqs_v,
        stream_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_scaled_dot_product_attention(int argc, VALUE* argv, VALUE) {
  try {
    VALUE q;
    VALUE k;
    VALUE v;
    VALUE scale;
    VALUE mask;
    VALUE sinks;
    VALUE stream;
    rb_scan_args(argc, argv, "34", &q, &k, &v, &scale, &mask, &sinks, &stream);

    std::string mask_mode;
    std::optional<mx::array> mask_arr = std::nullopt;
    if (!NIL_P(mask)) {
      if (RB_TYPE_P(mask, T_STRING) || SYMBOL_P(mask)) {
        mask_mode = string_from_ruby(mask);
      } else {
        mask_arr = array_from_ruby(mask, std::nullopt);
      }
    }

    auto sinks_v = optional_array_from_value(sinks);
    const float scale_v = NIL_P(scale) ? 1.0f : static_cast<float>(NUM2DBL(scale));

    return array_wrap(mxfast::scaled_dot_product_attention(
        array_from_ruby(q, std::nullopt),
        array_from_ruby(k, std::nullopt),
        array_from_ruby(v, std::nullopt),
        scale_v,
        mask_mode,
        mask_arr,
        sinks_v,
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_abs(VALUE, VALUE array) {
  try {
    return array_wrap(mx::abs(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_exp(VALUE, VALUE array) {
  try {
    return array_wrap(mx::exp(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sigmoid(VALUE, VALUE array) {
  try {
    return array_wrap(mx::sigmoid(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_log(VALUE, VALUE array) {
  try {
    return array_wrap(mx::log(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_logaddexp(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::logaddexp(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_logsumexp(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE keepdims;
    rb_scan_args(argc, argv, "12", &array, &axis, &keepdims);
    const bool keepdims_v = NIL_P(keepdims) ? false : RTEST(keepdims);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      return array_wrap(mx::logsumexp(a, keepdims_v));
    }
    if (RB_INTEGER_TYPE_P(axis)) {
      return array_wrap(mx::logsumexp(a, NUM2INT(axis), keepdims_v));
    }
    return array_wrap(mx::logsumexp(a, int_vector_from_ruby(axis), keepdims_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_logcumsumexp(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE axis;
    VALUE reverse;
    VALUE inclusive;
    rb_scan_args(argc, argv, "13", &array, &axis, &reverse, &inclusive);
    const bool reverse_v = NIL_P(reverse) ? false : RTEST(reverse);
    const bool inclusive_v = NIL_P(inclusive) ? true : RTEST(inclusive);
    auto a = array_unwrap(array);

    if (NIL_P(axis)) {
      auto flat = mx::reshape(a, mx::Shape{-1});
      return array_wrap(mx::logcumsumexp(flat, 0, reverse_v, inclusive_v));
    }
    return array_wrap(mx::logcumsumexp(a, NUM2INT(axis), reverse_v, inclusive_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sin(VALUE, VALUE array) {
  try {
    return array_wrap(mx::sin(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cos(VALUE, VALUE array) {
  try {
    return array_wrap(mx::cos(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tan(VALUE, VALUE array) {
  try {
    return array_wrap(mx::tan(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arcsin(VALUE, VALUE array) {
  try {
    return array_wrap(mx::arcsin(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arccos(VALUE, VALUE array) {
  try {
    return array_wrap(mx::arccos(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arctan(VALUE, VALUE array) {
  try {
    return array_wrap(mx::arctan(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arcsinh(VALUE, VALUE array) {
  try {
    return array_wrap(mx::arcsinh(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arccosh(VALUE, VALUE array) {
  try {
    return array_wrap(mx::arccosh(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arctanh(VALUE, VALUE array) {
  try {
    return array_wrap(mx::arctanh(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arctan2(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::arctan2(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_degrees(VALUE, VALUE array) {
  try {
    return array_wrap(mx::degrees(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_radians(VALUE, VALUE array) {
  try {
    return array_wrap(mx::radians(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sinh(VALUE, VALUE array) {
  try {
    return array_wrap(mx::sinh(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cosh(VALUE, VALUE array) {
  try {
    return array_wrap(mx::cosh(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tanh(VALUE, VALUE array) {
  try {
    return array_wrap(mx::tanh(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_negative(VALUE, VALUE array) {
  try {
    return array_wrap(mx::negative(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sign(VALUE, VALUE array) {
  try {
    return array_wrap(mx::sign(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_reciprocal(VALUE, VALUE array) {
  try {
    return array_wrap(mx::reciprocal(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_square(VALUE, VALUE array) {
  try {
    return array_wrap(mx::square(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_log1p(VALUE, VALUE array) {
  try {
    return array_wrap(mx::log1p(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_log2(VALUE, VALUE array) {
  try {
    return array_wrap(mx::log2(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_log10(VALUE, VALUE array) {
  try {
    return array_wrap(mx::log10(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_expm1(VALUE, VALUE array) {
  try {
    return array_wrap(mx::expm1(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_erf(VALUE, VALUE array) {
  try {
    return array_wrap(mx::erf(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_erfinv(VALUE, VALUE array) {
  try {
    return array_wrap(mx::erfinv(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_round(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE decimals;
    rb_scan_args(argc, argv, "11", &array, &decimals);
    const int decimals_v = NIL_P(decimals) ? 0 : NUM2INT(decimals);
    return array_wrap(mx::round(array_from_ruby(array, std::nullopt), decimals_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sqrt(VALUE, VALUE array) {
  try {
    return array_wrap(mx::sqrt(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_rsqrt(VALUE, VALUE array) {
  try {
    return array_wrap(mx::rsqrt(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_floor_divide(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::floor_divide(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_left_shift(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::left_shift(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_right_shift(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::right_shift(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_isfinite(VALUE, VALUE array) {
  try {
    return array_wrap(mx::isfinite(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_isnan(VALUE, VALUE array) {
  try {
    return array_wrap(mx::isnan(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_isinf(VALUE, VALUE array) {
  try {
    return array_wrap(mx::isinf(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_isposinf(VALUE, VALUE array) {
  try {
    return array_wrap(mx::isposinf(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_isneginf(VALUE, VALUE array) {
  try {
    return array_wrap(mx::isneginf(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_nan_to_num(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE nan;
    VALUE posinf;
    VALUE neginf;
    rb_scan_args(argc, argv, "13", &array, &nan, &posinf, &neginf);

    const float nan_v = NIL_P(nan) ? 0.0f : static_cast<float>(NUM2DBL(nan));
    std::optional<float> posinf_v = NIL_P(posinf)
        ? std::nullopt
        : std::make_optional(static_cast<float>(NUM2DBL(posinf)));
    std::optional<float> neginf_v = NIL_P(neginf)
        ? std::nullopt
        : std::make_optional(static_cast<float>(NUM2DBL(neginf)));

    return array_wrap(mx::nan_to_num(array_unwrap(array), nan_v, posinf_v, neginf_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_minimum(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::minimum(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_maximum(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::maximum(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_floor(VALUE, VALUE array) {
  try {
    return array_wrap(mx::floor(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ceil(VALUE, VALUE array) {
  try {
    return array_wrap(mx::ceil(array_from_ruby(array, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_clip(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE a_min;
    VALUE a_max;
    rb_scan_args(argc, argv, "12", &array, &a_min, &a_max);

    if (NIL_P(a_min) && NIL_P(a_max)) {
      rb_raise(rb_eArgError, "clip requires at least one bound");
    }

    auto input = array_from_ruby(array, std::nullopt);
    std::optional<mx::array> min_value =
        NIL_P(a_min) ? std::nullopt : std::make_optional(array_from_ruby(a_min, std::nullopt));
    std::optional<mx::array> max_value =
        NIL_P(a_max) ? std::nullopt : std::make_optional(array_from_ruby(a_max, std::nullopt));

    return array_wrap(mx::clip(input, min_value, max_value));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_logical_not(VALUE, VALUE a) {
  try {
    return array_wrap(mx::logical_not(array_from_ruby(a, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_logical_and(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::logical_and(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_logical_or(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::logical_or(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_bitwise_and(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::bitwise_and(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_bitwise_or(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::bitwise_or(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_bitwise_xor(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::bitwise_xor(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_bitwise_invert(VALUE, VALUE a) {
  try {
    return array_wrap(mx::bitwise_invert(array_from_ruby(a, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_allclose(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE rtol;
    VALUE atol;
    VALUE equal_nan;
    rb_scan_args(argc, argv, "23", &a, &b, &rtol, &atol, &equal_nan);

    const double rtol_v = NIL_P(rtol) ? 1e-5 : NUM2DBL(rtol);
    const double atol_v = NIL_P(atol) ? 1e-8 : NUM2DBL(atol);
    const bool equal_nan_v = NIL_P(equal_nan) ? false : RTEST(equal_nan);

    auto out = mx::allclose(array_unwrap(a), array_unwrap(b), rtol_v, atol_v, equal_nan_v);
    return out.item<bool>() ? Qtrue : Qfalse;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_equal(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::equal(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_not_equal(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::not_equal(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_greater(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::greater(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_greater_equal(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::greater_equal(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_less(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::less(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_less_equal(VALUE, VALUE a, VALUE b) {
  try {
    return array_wrap(mx::less_equal(array_from_ruby(a, std::nullopt), array_from_ruby(b, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_where(VALUE, VALUE condition, VALUE x, VALUE y) {
  try {
    return array_wrap(mx::where(
        array_from_ruby(condition, std::nullopt),
        array_from_ruby(x, std::nullopt),
        array_from_ruby(y, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_array_equal(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE equal_nan;
    rb_scan_args(argc, argv, "21", &a, &b, &equal_nan);
    const bool equal_nan_v = NIL_P(equal_nan) ? false : RTEST(equal_nan);

    auto out = mx::array_equal(
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        equal_nan_v);
    return out.item<bool>() ? Qtrue : Qfalse;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_isclose(int argc, VALUE* argv, VALUE) {
  try {
    VALUE a;
    VALUE b;
    VALUE rtol;
    VALUE atol;
    VALUE equal_nan;
    rb_scan_args(argc, argv, "23", &a, &b, &rtol, &atol, &equal_nan);

    const double rtol_v = NIL_P(rtol) ? 1e-5 : NUM2DBL(rtol);
    const double atol_v = NIL_P(atol) ? 1e-8 : NUM2DBL(atol);
    const bool equal_nan_v = NIL_P(equal_nan) ? false : RTEST(equal_nan);

    return array_wrap(mx::isclose(
        array_from_ruby(a, std::nullopt),
        array_from_ruby(b, std::nullopt),
        rtol_v,
        atol_v,
        equal_nan_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_arange(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 1 || argc > 4) {
      rb_raise(rb_eArgError, "arange expects 1 to 4 arguments");
    }

    double start = 0.0;
    double stop = 0.0;
    double step = 1.0;
    mx::Dtype dtype = mx::float32;

    if (argc == 1) {
      stop = NUM2DBL(argv[0]);
    } else if (argc == 2) {
      if (value_looks_like_dtype(argv[1])) {
        stop = NUM2DBL(argv[0]);
        dtype = optional_dtype_from_value(argv[1]).value_or(mx::float32);
      } else {
        start = NUM2DBL(argv[0]);
        stop = NUM2DBL(argv[1]);
      }
    } else if (argc == 3) {
      if (value_looks_like_dtype(argv[2])) {
        start = NUM2DBL(argv[0]);
        stop = NUM2DBL(argv[1]);
        dtype = optional_dtype_from_value(argv[2]).value_or(mx::float32);
      } else {
        start = NUM2DBL(argv[0]);
        stop = NUM2DBL(argv[1]);
        step = NUM2DBL(argv[2]);
      }
    } else {
      start = NUM2DBL(argv[0]);
      stop = NUM2DBL(argv[1]);
      step = NUM2DBL(argv[2]);
      dtype = optional_dtype_from_value(argv[3]).value_or(mx::float32);
    }

    return array_wrap(mx::arange(start, stop, step, dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_linspace(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2 || argc > 4) {
      rb_raise(rb_eArgError, "linspace expects 2 to 4 arguments");
    }

    const double start = NUM2DBL(argv[0]);
    const double stop = NUM2DBL(argv[1]);
    int num = 50;
    mx::Dtype dtype = mx::float32;

    if (argc == 3) {
      if (value_looks_like_dtype(argv[2])) {
        dtype = optional_dtype_from_value(argv[2]).value_or(mx::float32);
      } else {
        num = NUM2INT(argv[2]);
      }
    } else if (argc == 4) {
      num = NUM2INT(argv[2]);
      dtype = optional_dtype_from_value(argv[3]).value_or(mx::float32);
    }

    return array_wrap(mx::linspace(start, stop, num, dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_zeros(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE dtype;
    rb_scan_args(argc, argv, "11", &shape, &dtype);
    auto target_shape = shape_from_ruby(shape);
    auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);
    return array_wrap(mx::zeros(target_shape, target_dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ones(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE dtype;
    rb_scan_args(argc, argv, "11", &shape, &dtype);
    auto target_shape = shape_from_ruby(shape);
    auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);
    return array_wrap(mx::ones(target_shape, target_dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_full(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE value;
    VALUE dtype;
    rb_scan_args(argc, argv, "21", &shape, &value, &dtype);

    auto target_shape = shape_from_ruby(shape);
    auto target_dtype = optional_dtype_from_value(dtype);
    if (target_dtype.has_value()) {
      return array_wrap(mx::full(
          std::move(target_shape),
          array_from_ruby(value, target_dtype),
          target_dtype.value()));
    }
    return array_wrap(mx::full(std::move(target_shape), array_from_ruby(value, std::nullopt)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_zeros_like(VALUE, VALUE array) {
  try {
    return array_wrap(mx::zeros_like(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_ones_like(VALUE, VALUE array) {
  try {
    return array_wrap(mx::ones_like(array_unwrap(array)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_eye(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 1 || argc > 4) {
      rb_raise(rb_eArgError, "eye expects 1 to 4 arguments");
    }

    const int n = NUM2INT(argv[0]);
    int m = n;
    int k = 0;
    mx::Dtype dtype = mx::float32;

    if (argc == 2) {
      if (value_looks_like_dtype(argv[1])) {
        dtype = optional_dtype_from_value(argv[1]).value_or(mx::float32);
      } else {
        m = NUM2INT(argv[1]);
      }
    } else if (argc == 3) {
      if (value_looks_like_dtype(argv[2])) {
        m = NUM2INT(argv[1]);
        dtype = optional_dtype_from_value(argv[2]).value_or(mx::float32);
      } else {
        m = NUM2INT(argv[1]);
        k = NUM2INT(argv[2]);
      }
    } else if (argc == 4) {
      m = NUM2INT(argv[1]);
      k = NUM2INT(argv[2]);
      dtype = optional_dtype_from_value(argv[3]).value_or(mx::float32);
    }

    return array_wrap(mx::eye(n, m, k, dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_identity(int argc, VALUE* argv, VALUE) {
  try {
    VALUE n;
    VALUE dtype;
    rb_scan_args(argc, argv, "11", &n, &dtype);
    auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);
    return array_wrap(mx::identity(NUM2INT(n), target_dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tri(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 1 || argc > 4) {
      rb_raise(rb_eArgError, "tri expects 1 to 4 arguments");
    }

    const int n = NUM2INT(argv[0]);
    int m = n;
    int k = 0;
    mx::Dtype dtype = mx::float32;

    if (argc == 2) {
      if (value_looks_like_dtype(argv[1])) {
        dtype = optional_dtype_from_value(argv[1]).value_or(mx::float32);
      } else {
        m = NUM2INT(argv[1]);
      }
    } else if (argc == 3) {
      if (value_looks_like_dtype(argv[2])) {
        m = NUM2INT(argv[1]);
        dtype = optional_dtype_from_value(argv[2]).value_or(mx::float32);
      } else {
        m = NUM2INT(argv[1]);
        k = NUM2INT(argv[2]);
      }
    } else if (argc == 4) {
      m = NUM2INT(argv[1]);
      k = NUM2INT(argv[2]);
      dtype = optional_dtype_from_value(argv[3]).value_or(mx::float32);
    }

    return array_wrap(mx::tri(n, m, k, dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_tril(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE k;
    rb_scan_args(argc, argv, "11", &array, &k);
    const int k_v = NIL_P(k) ? 0 : NUM2INT(k);
    return array_wrap(mx::tril(array_unwrap(array), k_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_triu(int argc, VALUE* argv, VALUE) {
  try {
    VALUE array;
    VALUE k;
    rb_scan_args(argc, argv, "11", &array, &k);
    const int k_v = NIL_P(k) ? 0 : NUM2INT(k);
    return array_wrap(mx::triu(array_unwrap(array), k_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_astype(VALUE, VALUE array, VALUE dtype) {
  try {
    auto target_dtype = optional_dtype_from_value(dtype).value_or(mx::float32);
    return array_wrap(mx::astype(array_unwrap(array), target_dtype));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE native_loaded_p(VALUE) {
  return Qtrue;
}

static VALUE core_version(VALUE) {
  try {
    return rb_utf8_str_new_cstr(mx::version());
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_get_active_memory(VALUE) {
  try {
    return ULL2NUM(static_cast<unsigned long long>(mx::get_active_memory()));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_get_peak_memory(VALUE) {
  try {
    return ULL2NUM(static_cast<unsigned long long>(mx::get_peak_memory()));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_reset_peak_memory(VALUE) {
  try {
    mx::reset_peak_memory();
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_get_cache_memory(VALUE) {
  try {
    return ULL2NUM(static_cast<unsigned long long>(mx::get_cache_memory()));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_set_memory_limit(VALUE, VALUE limit) {
  try {
    const auto previous = mx::set_memory_limit(static_cast<size_t>(NUM2ULL(limit)));
    return ULL2NUM(static_cast<unsigned long long>(previous));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_set_cache_limit(VALUE, VALUE limit) {
  try {
    const auto previous = mx::set_cache_limit(static_cast<size_t>(NUM2ULL(limit)));
    return ULL2NUM(static_cast<unsigned long long>(previous));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_set_wired_limit(VALUE, VALUE limit) {
  try {
    const auto previous = mx::set_wired_limit(static_cast<size_t>(NUM2ULL(limit)));
    return ULL2NUM(static_cast<unsigned long long>(previous));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_clear_cache(VALUE) {
  try {
    mx::clear_cache();
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_metal_is_available(VALUE) {
  try {
    return mxmetal::is_available() ? Qtrue : Qfalse;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_metal_start_capture(VALUE, VALUE path) {
  try {
    mxmetal::start_capture(string_from_ruby(path));
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_metal_stop_capture(VALUE) {
  try {
    mxmetal::stop_capture();
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_metal_device_info(VALUE) {
  try {
    const auto& info = mxmetal::device_info();
    VALUE hash = rb_hash_new();
    for (const auto& [key, value] : info) {
      VALUE ruby_key = rb_utf8_str_new(key.c_str(), static_cast<long>(key.size()));
      VALUE ruby_value = Qnil;
      if (std::holds_alternative<std::string>(value)) {
        const auto& s = std::get<std::string>(value);
        ruby_value = rb_utf8_str_new(s.c_str(), static_cast<long>(s.size()));
      } else {
        ruby_value = ULL2NUM(static_cast<unsigned long long>(std::get<size_t>(value)));
      }
      rb_hash_aset(hash, ruby_key, ruby_value);
    }
    return hash;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_distributed_is_available(int argc, VALUE* argv, VALUE) {
  try {
    VALUE backend;
    rb_scan_args(argc, argv, "01", &backend);
    if (NIL_P(backend)) {
      return mxdist::is_available() ? Qtrue : Qfalse;
    }
    return mxdist::is_available(string_from_ruby(backend)) ? Qtrue : Qfalse;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_init(int argc, VALUE* argv, VALUE) {
  try {
    VALUE strict;
    VALUE backend;
    rb_scan_args(argc, argv, "02", &strict, &backend);
    const bool strict_v = RTEST(strict);
    const std::string backend_v = NIL_P(backend) ? "any" : string_from_ruby(backend);
    return group_wrap(mxdist::init(strict_v, backend_v));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_all_sum(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &x, &group, &stream);
    return array_wrap(
        mxdist::all_sum(
            array_from_ruby(x, std::nullopt),
            optional_group_from_value(group),
            stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_all_max(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &x, &group, &stream);
    return array_wrap(
        mxdist::all_max(
            array_from_ruby(x, std::nullopt),
            optional_group_from_value(group),
            stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_all_min(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &x, &group, &stream);
    return array_wrap(
        mxdist::all_min(
            array_from_ruby(x, std::nullopt),
            optional_group_from_value(group),
            stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_all_gather(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &x, &group, &stream);
    return array_wrap(
        mxdist::all_gather(
            array_from_ruby(x, std::nullopt),
            optional_group_from_value(group),
            stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_sum_scatter(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "12", &x, &group, &stream);
    return array_wrap(
        mxdist::sum_scatter(
            array_from_ruby(x, std::nullopt),
            optional_group_from_value(group),
            stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_send(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE dst;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "22", &x, &dst, &group, &stream);
    return array_wrap(mxdist::send(
        array_from_ruby(x, std::nullopt),
        NUM2INT(dst),
        optional_group_from_value(group),
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_recv(int argc, VALUE* argv, VALUE) {
  try {
    VALUE shape;
    VALUE dtype;
    VALUE src;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "32", &shape, &dtype, &src, &group, &stream);
    auto dtype_v = optional_dtype_from_value(dtype);
    if (!dtype_v.has_value()) {
      rb_raise(rb_eArgError, "dtype cannot be nil");
    }
    return array_wrap(mxdist::recv(
        shape_from_ruby(shape),
        dtype_v.value(),
        NUM2INT(src),
        optional_group_from_value(group),
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_recv_like(int argc, VALUE* argv, VALUE) {
  try {
    VALUE x;
    VALUE src;
    VALUE group;
    VALUE stream;
    rb_scan_args(argc, argv, "22", &x, &src, &group, &stream);
    return array_wrap(mxdist::recv_like(
        array_from_ruby(x, std::nullopt),
        NUM2INT(src),
        optional_group_from_value(group),
        stream_or_device_from_value(stream)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_default_device(VALUE) {
  try {
    return device_wrap(mx::default_device());
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_set_default_device(VALUE, VALUE device) {
  try {
    mx::set_default_device(device_from_object_or_type(device));
    return device;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cpu(VALUE) {
  return device_wrap(mx::Device(mx::Device::cpu, 0));
}

static VALUE core_gpu(VALUE) {
  return device_wrap(mx::Device(mx::Device::gpu, 0));
}

static VALUE core_is_available(VALUE, VALUE device_or_type) {
  try {
    const auto device = device_from_object_or_type(device_or_type);
    return mx::is_available(device) ? Qtrue : Qfalse;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_device_count(VALUE, VALUE type) {
  try {
    return INT2NUM(mx::device_count(device_type_from_value(type)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_device_info(int argc, VALUE* argv, VALUE) {
  try {
    VALUE device_arg;
    rb_scan_args(argc, argv, "01", &device_arg);

    mx::Device device = NIL_P(device_arg)
        ? mx::default_device()
        : device_from_object_or_type(device_arg);

    const auto& info = mx::device_info(device);
    VALUE hash = rb_hash_new();
    for (const auto& [key, value] : info) {
      VALUE ruby_key = rb_utf8_str_new(key.c_str(), static_cast<long>(key.size()));
      VALUE ruby_value = Qnil;
      if (std::holds_alternative<std::string>(value)) {
        const auto& s = std::get<std::string>(value);
        ruby_value = rb_utf8_str_new(s.c_str(), static_cast<long>(s.size()));
      } else {
        ruby_value = ULL2NUM(static_cast<unsigned long long>(std::get<size_t>(value)));
      }
      rb_hash_aset(hash, ruby_key, ruby_value);
    }

    return hash;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_default_stream(VALUE, VALUE device) {
  try {
    return stream_wrap(mx::default_stream(device_from_object_or_type(device)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_set_default_stream(VALUE, VALUE stream) {
  try {
    mx::set_default_stream(stream_unwrap(stream));
    return stream;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_new_stream(VALUE, VALUE device) {
  try {
    return stream_wrap(mx::new_stream(device_from_object_or_type(device)));
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_synchronize(int argc, VALUE* argv, VALUE) {
  try {
    VALUE stream;
    rb_scan_args(argc, argv, "01", &stream);

    if (NIL_P(stream)) {
      mx::synchronize();
    } else {
      mx::synchronize(stream_unwrap(stream));
    }
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

struct StreamRestoreContext {
  mx::Device device;
  mx::Stream stream;
};

static VALUE core_stream_restore(VALUE arg) {
  auto* restore = reinterpret_cast<StreamRestoreContext*>(arg);
  mx::set_default_device(restore->device);
  mx::set_default_stream(restore->stream);
  return Qnil;
}

static VALUE core_stream_yield(VALUE) {
  return rb_yield(Qnil);
}

static VALUE core_stream(VALUE, VALUE stream_or_device) {
  try {
    mx::Device target_device = mx::default_device();
    mx::Stream target_stream = mx::default_stream(target_device);
    if (rb_obj_is_kind_of(stream_or_device, cStream)) {
      target_stream = stream_unwrap(stream_or_device);
      target_device = target_stream.device;
    } else {
      target_device = device_from_object_or_type(stream_or_device);
      target_stream = mx::default_stream(target_device);
    }

    StreamRestoreContext restore{mx::default_device(), mx::default_stream(mx::default_device())};
    mx::set_default_device(target_device);
    mx::set_default_stream(target_stream);

    if (rb_block_given_p()) {
      return rb_ensure(
          core_stream_yield, Qnil, core_stream_restore, reinterpret_cast<VALUE>(&restore));
    }
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_metal_kernel(int argc, VALUE* argv, VALUE) {
  try {
    VALUE name;
    VALUE input_names;
    VALUE output_names;
    VALUE source;
    VALUE header;
    VALUE ensure_row_contiguous;
    VALUE atomic_outputs;
    rb_scan_args(
        argc,
        argv,
        "43",
        &name,
        &input_names,
        &output_names,
        &source,
        &header,
        &ensure_row_contiguous,
        &atomic_outputs);

    auto kernel = mxfast::metal_kernel(
        string_from_ruby(name),
        string_vector_from_ruby(input_names, "input_names"),
        string_vector_from_ruby(output_names, "output_names"),
        string_from_ruby(source),
        NIL_P(header) ? std::string{} : string_from_ruby(header),
        NIL_P(ensure_row_contiguous) ? true : RTEST(ensure_row_contiguous),
        RTEST(atomic_outputs));
    VALUE refs = rb_ary_new();
    for (int i = 0; i < argc; ++i) {
      rb_ary_push(refs, argv[i]);
    }
    return kernel_wrap(std::move(kernel), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_cuda_kernel(int argc, VALUE* argv, VALUE) {
  try {
    VALUE name;
    VALUE input_names;
    VALUE output_names;
    VALUE source;
    VALUE header;
    VALUE ensure_row_contiguous;
    VALUE shared_memory;
    rb_scan_args(
        argc,
        argv,
        "43",
        &name,
        &input_names,
        &output_names,
        &source,
        &header,
        &ensure_row_contiguous,
        &shared_memory);

    auto kernel = mxfast::cuda_kernel(
        string_from_ruby(name),
        string_vector_from_ruby(input_names, "input_names"),
        string_vector_from_ruby(output_names, "output_names"),
        string_from_ruby(source),
        NIL_P(header) ? std::string{} : string_from_ruby(header),
        NIL_P(ensure_row_contiguous) ? true : RTEST(ensure_row_contiguous),
        NIL_P(shared_memory) ? 0 : NUM2INT(shared_memory));
    VALUE refs = rb_ary_new();
    for (int i = 0; i < argc; ++i) {
      rb_ary_push(refs, argv[i]);
    }
    return kernel_wrap(std::move(kernel), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_precompiled_cuda_kernel(int argc, VALUE* argv, VALUE) {
  try {
    VALUE name;
    VALUE compiled_source;
    VALUE inputs;
    VALUE output_shapes;
    VALUE output_dtypes;
    VALUE scalars;
    VALUE grid;
    VALUE threadgroup;
    VALUE shared_memory;
    VALUE init_value;
    VALUE ensure_row_contiguous;
    VALUE stream;
    rb_scan_args(
        argc,
        argv,
        "84",
        &name,
        &compiled_source,
        &inputs,
        &output_shapes,
        &output_dtypes,
        &scalars,
        &grid,
        &threadgroup,
        &shared_memory,
        &init_value,
        &ensure_row_contiguous,
        &stream);

    std::optional<float> init_value_v = std::nullopt;
    if (!NIL_P(init_value)) {
      init_value_v = static_cast<float>(NUM2DBL(init_value));
    }

    auto outputs = mxfast::precompiled_cuda_kernel(
        string_from_ruby(name),
        string_from_ruby(compiled_source),
        array_inputs_from_ruby(inputs),
        shape_vector_from_ruby(output_shapes),
        dtype_vector_from_ruby(output_dtypes),
        scalar_args_from_ruby(scalars),
        int_triple_from_ruby_or_scalar(grid, {1, 1, 1}, "grid"),
        int_triple_from_ruby_or_scalar(threadgroup, {1, 1, 1}, "threadgroup"),
        NIL_P(shared_memory) ? 0 : NUM2INT(shared_memory),
        init_value_v,
        RTEST(ensure_row_contiguous),
        stream_or_device_from_value(stream));
    return ruby_array_of_arrays(outputs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_eval(int argc, VALUE* argv, VALUE) {
  try {
    std::vector<mx::array> arrays;
    arrays.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
      collect_arrays_from_tree(argv[i], arrays);
    }

    struct CoreEvalPayload {
      std::vector<mx::array>* arrays;
      std::exception_ptr error;
    };
    auto core_eval_without_gvl = [](void* arg) -> void* {
      auto* payload = reinterpret_cast<CoreEvalPayload*>(arg);
      try {
        mx::eval(*payload->arrays);
      } catch (...) {
        payload->error = std::current_exception();
      }
      return nullptr;
    };

    CoreEvalPayload payload{&arrays, nullptr};
    rb_thread_call_without_gvl(core_eval_without_gvl, &payload, RUBY_UBF_IO, nullptr);
    rethrow_captured_exception(payload.error);
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_async_eval(int argc, VALUE* argv, VALUE) {
  try {
    std::vector<mx::array> arrays;
    arrays.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
      collect_arrays_from_tree(argv[i], arrays);
    }

    struct CoreAsyncEvalPayload {
      std::vector<mx::array>* arrays;
      std::exception_ptr error;
    };
    auto core_async_eval_without_gvl = [](void* arg) -> void* {
      auto* payload = reinterpret_cast<CoreAsyncEvalPayload*>(arg);
      try {
        mx::async_eval(*payload->arrays);
      } catch (...) {
        payload->error = std::current_exception();
      }
      return nullptr;
    };

    CoreAsyncEvalPayload payload{&arrays, nullptr};
    rb_thread_call_without_gvl(core_async_eval_without_gvl, &payload, RUBY_UBF_IO, nullptr);
    rethrow_captured_exception(payload.error);
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_disable_compile(VALUE) {
  try {
    mx::disable_compile();
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_enable_compile(VALUE) {
  try {
    mx::enable_compile();
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_jvp(VALUE, VALUE fun, VALUE primals, VALUE tangents) {
  try {
    auto result = mx::jvp(
        vector_function_from_callable(fun),
        array_vector_from_ruby(primals),
        array_vector_from_ruby(tangents));
    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, ruby_array_of_arrays(result.first));
    rb_ary_push(out, ruby_array_of_arrays(result.second));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_vjp(VALUE, VALUE fun, VALUE primals, VALUE cotangents) {
  try {
    auto result = mx::vjp(
        vector_function_from_callable(fun),
        array_vector_from_ruby(primals),
        array_vector_from_ruby(cotangents));
    VALUE out = rb_ary_new_capa(2);
    rb_ary_push(out, ruby_array_of_arrays(result.first));
    rb_ary_push(out, ruby_array_of_arrays(result.second));
    return out;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_compile(int argc, VALUE* argv, VALUE) {
  try {
    VALUE fun;
    VALUE inputs;
    VALUE outputs;
    VALUE shapeless;
    rb_scan_args(argc, argv, "13", &fun, &inputs, &outputs, &shapeless);

    auto compiled = mx::compile(vector_function_from_callable(fun), RTEST(shapeless));

    VALUE refs = rb_ary_new();
    rb_ary_push(refs, fun);
    rb_ary_push(refs, inputs);
    rb_ary_push(refs, outputs);
    return function_wrap_vector(std::move(compiled), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_checkpoint(VALUE, VALUE fun) {
  try {
    auto checkpointed = mx::checkpoint(vector_function_from_callable(fun));
    VALUE refs = rb_ary_new();
    rb_ary_push(refs, fun);
    return function_wrap_vector(std::move(checkpointed), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_grad(int argc, VALUE* argv, VALUE) {
  try {
    VALUE fun;
    VALUE argnums;
    rb_scan_args(argc, argv, "11", &fun, &argnums);

    auto vector_fun = vector_function_from_callable(fun);
    auto scalar_fun = [vector_fun](const std::vector<mx::array>& inputs) {
      auto out = vector_fun(inputs);
      if (out.empty()) {
        throw std::invalid_argument("[grad] callable must return at least one array");
      }
      return out.at(0);
    };
    auto grad_fn = mx::grad(scalar_fun, argnums_from_value(argnums));

    VALUE refs = rb_ary_new();
    rb_ary_push(refs, fun);
    return function_wrap_vector(std::move(grad_fn), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_value_and_grad(int argc, VALUE* argv, VALUE) {
  try {
    VALUE fun;
    VALUE argnums;
    rb_scan_args(argc, argv, "11", &fun, &argnums);

    auto vector_fun = vector_function_from_callable(fun);
    auto scalar_fun = [vector_fun](const std::vector<mx::array>& inputs) {
      auto out = vector_fun(inputs);
      if (out.empty()) {
        throw std::invalid_argument("[value_and_grad] callable must return at least one array");
      }
      return out.at(0);
    };
    auto value_grad_fn = mx::value_and_grad(scalar_fun, argnums_from_value(argnums));

    auto wrapped =
        [value_grad_fn](const std::vector<mx::array>& inputs) -> std::pair<
            std::vector<mx::array>,
            std::vector<mx::array>> {
      auto result = value_grad_fn(inputs);
      return {{result.first}, result.second};
    };

    VALUE refs = rb_ary_new();
    rb_ary_push(refs, fun);
    return function_wrap_value_grad(std::move(wrapped), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_vmap(int argc, VALUE* argv, VALUE) {
  try {
    VALUE fun;
    VALUE in_axes;
    VALUE out_axes;
    rb_scan_args(argc, argv, "12", &fun, &in_axes, &out_axes);

    auto vmapped = mx::vmap(
        vector_function_from_callable(fun),
        vmap_axes_from_value(in_axes),
        vmap_axes_from_value(out_axes));

    VALUE refs = rb_ary_new();
    rb_ary_push(refs, fun);
    return function_wrap_vector(std::move(vmapped), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_export_function(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2) {
      rb_raise(rb_eArgError, "export_function expects at least file and callable");
    }
    VALUE file = argv[0];
    VALUE fun = argv[1];

    bool shapeless = false;
    int end = argc;
    if (argc > 2 && (argv[argc - 1] == Qtrue || argv[argc - 1] == Qfalse)) {
      shapeless = RTEST(argv[argc - 1]);
      end -= 1;
    }

    std::vector<VALUE> extras;
    extras.reserve(static_cast<size_t>(std::max(0, end - 2)));
    for (int i = 2; i < end; ++i) {
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
      if (rb_obj_is_kind_of(item, cArray)) {
        args.push_back(array_unwrap(item));
      } else if (RB_TYPE_P(item, T_ARRAY)) {
        args = array_vector_from_ruby(item);
      } else {
        args.push_back(array_from_ruby(item, std::nullopt));
      }
    } else {
      args.reserve(extras.size());
      for (VALUE item : extras) {
        args.push_back(array_from_ruby(item, std::nullopt));
      }
    }
    mx::Kwargs kwargs = NIL_P(kwargs_hash) ? mx::Kwargs{} : array_map_from_ruby_hash(kwargs_hash);
    if (args.empty() && kwargs.empty()) {
      rb_raise(
          rb_eArgError,
          "[export_function] Inputs must include at least one positional or keyword array");
    }

    mx::export_function(
        string_from_ruby(file),
        args_kwargs_function_from_callable(fun),
        args,
        kwargs,
        shapeless);
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_import_function(VALUE, VALUE file) {
  try {
    auto imported = mx::import_function(string_from_ruby(file));
    auto wrapped = [imported](const mx::Args& args, const mx::Kwargs& kwargs) mutable {
      return imported(args, kwargs);
    };
    VALUE refs = rb_ary_new();
    rb_ary_push(refs, file);
    return function_wrap_args_kwargs(std::move(wrapped), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_exporter(int argc, VALUE* argv, VALUE) {
  try {
    VALUE file;
    VALUE fun;
    VALUE shapeless;
    rb_scan_args(argc, argv, "21", &file, &fun, &shapeless);
    auto exporter =
        mx::exporter(string_from_ruby(file), args_kwargs_function_from_callable(fun), RTEST(shapeless));
    VALUE refs = rb_ary_new();
    rb_ary_push(refs, file);
    rb_ary_push(refs, fun);
    return function_exporter_wrap(std::move(exporter), refs);
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_export_to_dot(int argc, VALUE* argv, VALUE) {
  try {
    if (argc < 2) {
      rb_raise(rb_eArgError, "export_to_dot expects a path and at least one output");
    }

    std::vector<mx::array> outputs;
    outputs.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; ++i) {
      collect_arrays_from_tree(argv[i], outputs);
    }

    std::ofstream out(string_from_ruby(argv[0]));
    if (!out.is_open()) {
      rb_raise(rb_eRuntimeError, "failed to open output file");
    }
    mx::export_to_dot(out, outputs);
    return Qnil;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_bool_(VALUE) { return dtype_wrap(mx::bool_); }
static VALUE core_uint8(VALUE) { return dtype_wrap(mx::uint8); }
static VALUE core_uint16(VALUE) { return dtype_wrap(mx::uint16); }
static VALUE core_uint32(VALUE) { return dtype_wrap(mx::uint32); }
static VALUE core_uint64(VALUE) { return dtype_wrap(mx::uint64); }
static VALUE core_int8(VALUE) { return dtype_wrap(mx::int8); }
static VALUE core_int16(VALUE) { return dtype_wrap(mx::int16); }
static VALUE core_int32(VALUE) { return dtype_wrap(mx::int32); }
static VALUE core_int64(VALUE) { return dtype_wrap(mx::int64); }
static VALUE core_float16(VALUE) { return dtype_wrap(mx::float16); }
static VALUE core_float32(VALUE) { return dtype_wrap(mx::float32); }
static VALUE core_float64(VALUE) { return dtype_wrap(mx::float64); }
static VALUE core_bfloat16(VALUE) { return dtype_wrap(mx::bfloat16); }
static VALUE core_complex64(VALUE) { return dtype_wrap(mx::complex64); }

static VALUE core_complexfloating(VALUE) { return category_to_symbol(mx::complexfloating); }
static VALUE core_floating(VALUE) { return category_to_symbol(mx::floating); }
static VALUE core_inexact(VALUE) { return category_to_symbol(mx::inexact); }
static VALUE core_signedinteger(VALUE) { return category_to_symbol(mx::signedinteger); }
static VALUE core_unsignedinteger(VALUE) { return category_to_symbol(mx::unsignedinteger); }
static VALUE core_integer(VALUE) { return category_to_symbol(mx::integer); }
static VALUE core_number(VALUE) { return category_to_symbol(mx::number); }
static VALUE core_generic(VALUE) { return category_to_symbol(mx::generic); }

using DtypeOrCategory = std::variant<mx::Dtype, mx::Dtype::Category>;

static DtypeOrCategory dtype_or_category_from_value(VALUE value) {
  if (rb_obj_is_kind_of(value, cDtype)) {
    return dtype_unwrap(value);
  }

  VALUE symbol = value;
  if (RB_TYPE_P(value, T_STRING)) {
    symbol = rb_str_intern(value);
  }
  if (!SYMBOL_P(symbol)) {
    rb_raise(rb_eArgError, "expected dtype or dtype category symbol");
  }

  if (symbol_is_dtype(symbol)) {
    return dtype_from_symbol(symbol);
  }

  return category_from_symbol(symbol);
}

static VALUE core_issubdtype(VALUE, VALUE a, VALUE b) {
  try {
    auto lhs = dtype_or_category_from_value(a);
    auto rhs = dtype_or_category_from_value(b);

    bool result = false;
    if (std::holds_alternative<mx::Dtype>(lhs) && std::holds_alternative<mx::Dtype>(rhs)) {
      result = mx::issubdtype(std::get<mx::Dtype>(lhs), std::get<mx::Dtype>(rhs));
    } else if (std::holds_alternative<mx::Dtype::Category>(lhs) && std::holds_alternative<mx::Dtype>(rhs)) {
      result = mx::issubdtype(std::get<mx::Dtype::Category>(lhs), std::get<mx::Dtype>(rhs));
    } else if (std::holds_alternative<mx::Dtype>(lhs) && std::holds_alternative<mx::Dtype::Category>(rhs)) {
      result = mx::issubdtype(std::get<mx::Dtype>(lhs), std::get<mx::Dtype::Category>(rhs));
    } else {
      result = mx::issubdtype(
          std::get<mx::Dtype::Category>(lhs),
          std::get<mx::Dtype::Category>(rhs));
    }

    return result ? Qtrue : Qfalse;
  } catch (const std::exception& error) {
    raise_std_exception(error);
    return Qnil;
  }
}

static VALUE core_pi(VALUE) {
  return DBL2NUM(3.1415926535897932384626433);
}

static VALUE core_e(VALUE) {
  return DBL2NUM(2.71828182845904523536028747135266249775724709369995);
}

static VALUE core_euler_gamma(VALUE) {
  return DBL2NUM(0.5772156649015328606065120900824024310421);
}

static VALUE core_inf(VALUE) {
  return DBL2NUM(std::numeric_limits<double>::infinity());
}

static VALUE core_nan(VALUE) {
  return DBL2NUM(std::numeric_limits<double>::quiet_NaN());
}

static VALUE core_newaxis(VALUE) {
  return Qnil;
}

extern "C" void Init_native(void) {
  mMLX = rb_define_module("MLX");
  mNative = rb_define_module_under(mMLX, "Native");
  rb_define_singleton_method(mNative, "loaded?", RUBY_METHOD_FUNC(native_loaded_p), 0);

  mCore = rb_define_module_under(mMLX, "Core");
  rb_define_singleton_method(mCore, "version", RUBY_METHOD_FUNC(core_version), 0);

  rb_define_singleton_method(mCore, "get_active_memory", RUBY_METHOD_FUNC(core_get_active_memory), 0);
  rb_define_singleton_method(mCore, "get_peak_memory", RUBY_METHOD_FUNC(core_get_peak_memory), 0);
  rb_define_singleton_method(mCore, "reset_peak_memory", RUBY_METHOD_FUNC(core_reset_peak_memory), 0);
  rb_define_singleton_method(mCore, "get_cache_memory", RUBY_METHOD_FUNC(core_get_cache_memory), 0);
  rb_define_singleton_method(mCore, "set_memory_limit", RUBY_METHOD_FUNC(core_set_memory_limit), 1);
  rb_define_singleton_method(mCore, "set_cache_limit", RUBY_METHOD_FUNC(core_set_cache_limit), 1);
  rb_define_singleton_method(mCore, "set_wired_limit", RUBY_METHOD_FUNC(core_set_wired_limit), 1);
  rb_define_singleton_method(mCore, "clear_cache", RUBY_METHOD_FUNC(core_clear_cache), 0);
  rb_define_singleton_method(mCore, "metal_is_available", RUBY_METHOD_FUNC(core_metal_is_available), 0);
  rb_define_singleton_method(mCore, "metal_start_capture", RUBY_METHOD_FUNC(core_metal_start_capture), 1);
  rb_define_singleton_method(mCore, "metal_stop_capture", RUBY_METHOD_FUNC(core_metal_stop_capture), 0);
  rb_define_singleton_method(mCore, "metal_device_info", RUBY_METHOD_FUNC(core_metal_device_info), 0);
  rb_define_singleton_method(
      mCore, "distributed_is_available", RUBY_METHOD_FUNC(core_distributed_is_available), -1);
  rb_define_singleton_method(mCore, "init", RUBY_METHOD_FUNC(core_init), -1);
  rb_define_singleton_method(mCore, "all_sum", RUBY_METHOD_FUNC(core_all_sum), -1);
  rb_define_singleton_method(mCore, "all_max", RUBY_METHOD_FUNC(core_all_max), -1);
  rb_define_singleton_method(mCore, "all_min", RUBY_METHOD_FUNC(core_all_min), -1);
  rb_define_singleton_method(mCore, "all_gather", RUBY_METHOD_FUNC(core_all_gather), -1);
  rb_define_singleton_method(mCore, "sum_scatter", RUBY_METHOD_FUNC(core_sum_scatter), -1);
  rb_define_singleton_method(mCore, "send", RUBY_METHOD_FUNC(core_send), -1);
  rb_define_singleton_method(mCore, "recv", RUBY_METHOD_FUNC(core_recv), -1);
  rb_define_singleton_method(mCore, "recv_like", RUBY_METHOD_FUNC(core_recv_like), -1);

  cArray = rb_define_class_under(mCore, "Array", rb_cObject);
  rb_define_alloc_func(cArray, array_alloc);
  rb_define_method(cArray, "initialize", RUBY_METHOD_FUNC(array_initialize), -1);
  rb_define_method(cArray, "ndim", RUBY_METHOD_FUNC(array_ndim), 0);
  rb_define_method(cArray, "size", RUBY_METHOD_FUNC(array_size), 0);
  rb_define_method(cArray, "shape", RUBY_METHOD_FUNC(array_shape), 0);
  rb_define_method(cArray, "dtype", RUBY_METHOD_FUNC(array_dtype), 0);
  rb_define_method(cArray, "item", RUBY_METHOD_FUNC(array_item), 0);
  rb_define_method(cArray, "to_a", RUBY_METHOD_FUNC(array_to_a), 0);
  rb_define_method(cArray, "+", RUBY_METHOD_FUNC(array_add), 1);
  rb_define_method(cArray, "-", RUBY_METHOD_FUNC(array_subtract), 1);
  rb_define_method(cArray, "*", RUBY_METHOD_FUNC(array_multiply), 1);
  rb_define_method(cArray, "/", RUBY_METHOD_FUNC(array_divide), 1);
  rb_define_method(cArray, "[]", RUBY_METHOD_FUNC(array_aref), 1);
  rb_define_method(cArray, "to_s", RUBY_METHOD_FUNC(array_to_s), 0);
  rb_define_method(cArray, "inspect", RUBY_METHOD_FUNC(array_to_s), 0);

  rb_define_singleton_method(mCore, "array", RUBY_METHOD_FUNC(core_array), -1);
  rb_define_singleton_method(mCore, "asarray", RUBY_METHOD_FUNC(core_array), -1);
  rb_define_singleton_method(mCore, "broadcast_shapes", RUBY_METHOD_FUNC(core_broadcast_shapes), -1);
  rb_define_singleton_method(mCore, "add", RUBY_METHOD_FUNC(core_add), 2);
  rb_define_singleton_method(mCore, "subtract", RUBY_METHOD_FUNC(core_subtract), 2);
  rb_define_singleton_method(mCore, "multiply", RUBY_METHOD_FUNC(core_multiply), 2);
  rb_define_singleton_method(mCore, "divide", RUBY_METHOD_FUNC(core_divide), 2);
  rb_define_singleton_method(mCore, "power", RUBY_METHOD_FUNC(core_power), 2);
  rb_define_singleton_method(mCore, "remainder", RUBY_METHOD_FUNC(core_remainder), 2);
  rb_define_singleton_method(mCore, "divmod", RUBY_METHOD_FUNC(core_divmod), 2);
  rb_define_singleton_method(mCore, "slice", RUBY_METHOD_FUNC(core_slice), -1);
  rb_define_singleton_method(mCore, "slice_update", RUBY_METHOD_FUNC(core_slice_update), -1);
  rb_define_singleton_method(mCore, "as_strided", RUBY_METHOD_FUNC(core_as_strided), -1);
  rb_define_singleton_method(mCore, "take", RUBY_METHOD_FUNC(core_take), -1);
  rb_define_singleton_method(mCore, "take_along_axis", RUBY_METHOD_FUNC(core_take_along_axis), -1);
  rb_define_singleton_method(mCore, "put_along_axis", RUBY_METHOD_FUNC(core_put_along_axis), -1);
  rb_define_singleton_method(mCore, "unflatten", RUBY_METHOD_FUNC(core_unflatten), 3);
  rb_define_singleton_method(mCore, "concatenate", RUBY_METHOD_FUNC(core_concatenate), -1);
  rb_define_singleton_method(mCore, "concat", RUBY_METHOD_FUNC(core_concat), -1);
  rb_define_singleton_method(mCore, "stack", RUBY_METHOD_FUNC(core_stack), -1);
  rb_define_singleton_method(mCore, "random_split", RUBY_METHOD_FUNC(core_random_split), -1);
  rb_define_singleton_method(mCore, "repeat", RUBY_METHOD_FUNC(core_repeat), -1);
  rb_define_singleton_method(mCore, "tile", RUBY_METHOD_FUNC(core_tile), 2);
  rb_define_singleton_method(mCore, "meshgrid", RUBY_METHOD_FUNC(core_meshgrid), -1);
  rb_define_singleton_method(mCore, "roll", RUBY_METHOD_FUNC(core_roll), -1);
  rb_define_singleton_method(mCore, "stop_gradient", RUBY_METHOD_FUNC(core_stop_gradient), 1);
  rb_define_singleton_method(mCore, "conjugate", RUBY_METHOD_FUNC(core_conjugate), 1);
  rb_define_singleton_method(mCore, "conj", RUBY_METHOD_FUNC(core_conjugate), 1);
  rb_define_singleton_method(mCore, "real", RUBY_METHOD_FUNC(core_real), 1);
  rb_define_singleton_method(mCore, "imag", RUBY_METHOD_FUNC(core_imag), 1);
  rb_define_singleton_method(mCore, "contiguous", RUBY_METHOD_FUNC(core_contiguous), -1);
  rb_define_singleton_method(mCore, "view", RUBY_METHOD_FUNC(core_view), 2);
  rb_define_singleton_method(mCore, "matmul", RUBY_METHOD_FUNC(core_matmul), 2);
  rb_define_singleton_method(mCore, "addmm", RUBY_METHOD_FUNC(core_addmm), -1);
  rb_define_singleton_method(mCore, "block_masked_mm", RUBY_METHOD_FUNC(core_block_masked_mm), -1);
  rb_define_singleton_method(mCore, "gather_mm", RUBY_METHOD_FUNC(core_gather_mm), -1);
  rb_define_singleton_method(mCore, "segmented_mm", RUBY_METHOD_FUNC(core_segmented_mm), 3);
  rb_define_singleton_method(mCore, "hadamard_transform", RUBY_METHOD_FUNC(core_hadamard_transform), -1);
  rb_define_singleton_method(mCore, "convolve", RUBY_METHOD_FUNC(core_convolve), -1);
  rb_define_singleton_method(mCore, "conv1d", RUBY_METHOD_FUNC(core_conv1d), -1);
  rb_define_singleton_method(mCore, "conv2d", RUBY_METHOD_FUNC(core_conv2d), -1);
  rb_define_singleton_method(mCore, "conv3d", RUBY_METHOD_FUNC(core_conv3d), -1);
  rb_define_singleton_method(mCore, "conv_transpose1d", RUBY_METHOD_FUNC(core_conv_transpose1d), -1);
  rb_define_singleton_method(mCore, "conv_transpose2d", RUBY_METHOD_FUNC(core_conv_transpose2d), -1);
  rb_define_singleton_method(mCore, "conv_transpose3d", RUBY_METHOD_FUNC(core_conv_transpose3d), -1);
  rb_define_singleton_method(mCore, "conv_general", RUBY_METHOD_FUNC(core_conv_general), -1);
  rb_define_singleton_method(mCore, "quantized_matmul", RUBY_METHOD_FUNC(core_quantized_matmul), -1);
  rb_define_singleton_method(mCore, "quantize", RUBY_METHOD_FUNC(core_quantize), -1);
  rb_define_singleton_method(mCore, "dequantize", RUBY_METHOD_FUNC(core_dequantize), -1);
  rb_define_singleton_method(mCore, "from_fp8", RUBY_METHOD_FUNC(core_from_fp8), -1);
  rb_define_singleton_method(mCore, "to_fp8", RUBY_METHOD_FUNC(core_to_fp8), 1);
  rb_define_singleton_method(mCore, "qqmm", RUBY_METHOD_FUNC(core_qqmm), -1);
  rb_define_singleton_method(mCore, "gather_qmm", RUBY_METHOD_FUNC(core_gather_qmm), -1);
  rb_define_singleton_method(mCore, "depends", RUBY_METHOD_FUNC(core_depends), 2);
  rb_define_singleton_method(mCore, "save", RUBY_METHOD_FUNC(core_save), 2);
  rb_define_singleton_method(mCore, "load", RUBY_METHOD_FUNC(core_load), -1);
  rb_define_singleton_method(mCore, "save_safetensors", RUBY_METHOD_FUNC(core_save_safetensors), -1);
  rb_define_singleton_method(mCore, "save_gguf", RUBY_METHOD_FUNC(core_save_gguf), -1);
  rb_define_singleton_method(mCore, "savez", RUBY_METHOD_FUNC(core_savez), -1);
  rb_define_singleton_method(mCore, "savez_compressed", RUBY_METHOD_FUNC(core_savez_compressed), -1);
  rb_define_singleton_method(mCore, "inner", RUBY_METHOD_FUNC(core_inner), 2);
  rb_define_singleton_method(mCore, "outer", RUBY_METHOD_FUNC(core_outer), 2);
  rb_define_singleton_method(mCore, "tensordot", RUBY_METHOD_FUNC(core_tensordot), -1);
  rb_define_singleton_method(mCore, "einsum", RUBY_METHOD_FUNC(core_einsum), -1);
  rb_define_singleton_method(mCore, "einsum_path", RUBY_METHOD_FUNC(core_einsum_path), -1);
  rb_define_singleton_method(mCore, "kron", RUBY_METHOD_FUNC(core_kron), 2);
  rb_define_singleton_method(mCore, "diagonal", RUBY_METHOD_FUNC(core_diagonal), -1);
  rb_define_singleton_method(mCore, "diag", RUBY_METHOD_FUNC(core_diag), -1);
  rb_define_singleton_method(mCore, "trace", RUBY_METHOD_FUNC(core_trace), -1);
  rb_define_singleton_method(mCore, "broadcast_to", RUBY_METHOD_FUNC(core_broadcast_to), 2);
  rb_define_singleton_method(mCore, "broadcast_arrays", RUBY_METHOD_FUNC(core_broadcast_arrays), 1);
  rb_define_singleton_method(mCore, "reshape", RUBY_METHOD_FUNC(core_reshape), 2);
  rb_define_singleton_method(mCore, "flatten", RUBY_METHOD_FUNC(core_flatten), -1);
  rb_define_singleton_method(mCore, "transpose", RUBY_METHOD_FUNC(core_transpose), -1);
  rb_define_singleton_method(mCore, "permute_dims", RUBY_METHOD_FUNC(core_permute_dims), -1);
  rb_define_singleton_method(mCore, "squeeze", RUBY_METHOD_FUNC(core_squeeze), -1);
  rb_define_singleton_method(mCore, "expand_dims", RUBY_METHOD_FUNC(core_expand_dims), 2);
  rb_define_singleton_method(mCore, "atleast_1d", RUBY_METHOD_FUNC(core_atleast_1d), 1);
  rb_define_singleton_method(mCore, "atleast_2d", RUBY_METHOD_FUNC(core_atleast_2d), 1);
  rb_define_singleton_method(mCore, "atleast_3d", RUBY_METHOD_FUNC(core_atleast_3d), 1);
  rb_define_singleton_method(mCore, "moveaxis", RUBY_METHOD_FUNC(core_moveaxis), 3);
  rb_define_singleton_method(mCore, "swapaxes", RUBY_METHOD_FUNC(core_swapaxes), 3);
  rb_define_singleton_method(mCore, "sum", RUBY_METHOD_FUNC(core_sum), -1);
  rb_define_singleton_method(mCore, "mean", RUBY_METHOD_FUNC(core_mean), -1);
  rb_define_singleton_method(mCore, "all", RUBY_METHOD_FUNC(core_all), -1);
  rb_define_singleton_method(mCore, "any", RUBY_METHOD_FUNC(core_any), -1);
  rb_define_singleton_method(mCore, "softmax", RUBY_METHOD_FUNC(core_softmax), -1);
  rb_define_singleton_method(mCore, "sort", RUBY_METHOD_FUNC(core_sort), -1);
  rb_define_singleton_method(mCore, "argsort", RUBY_METHOD_FUNC(core_argsort), -1);
  rb_define_singleton_method(mCore, "topk", RUBY_METHOD_FUNC(core_topk), -1);
  rb_define_singleton_method(mCore, "partition", RUBY_METHOD_FUNC(core_partition), -1);
  rb_define_singleton_method(mCore, "argpartition", RUBY_METHOD_FUNC(core_argpartition), -1);
  rb_define_singleton_method(mCore, "max", RUBY_METHOD_FUNC(core_max), -1);
  rb_define_singleton_method(mCore, "min", RUBY_METHOD_FUNC(core_min), -1);
  rb_define_singleton_method(mCore, "argmax", RUBY_METHOD_FUNC(core_argmax), -1);
  rb_define_singleton_method(mCore, "argmin", RUBY_METHOD_FUNC(core_argmin), -1);
  rb_define_singleton_method(mCore, "prod", RUBY_METHOD_FUNC(core_prod), -1);
  rb_define_singleton_method(mCore, "cumsum", RUBY_METHOD_FUNC(core_cumsum), -1);
  rb_define_singleton_method(mCore, "cumprod", RUBY_METHOD_FUNC(core_cumprod), -1);
  rb_define_singleton_method(mCore, "cummax", RUBY_METHOD_FUNC(core_cummax), -1);
  rb_define_singleton_method(mCore, "cummin", RUBY_METHOD_FUNC(core_cummin), -1);
  rb_define_singleton_method(mCore, "var", RUBY_METHOD_FUNC(core_var), -1);
  rb_define_singleton_method(mCore, "std", RUBY_METHOD_FUNC(core_std), -1);
  rb_define_singleton_method(mCore, "median", RUBY_METHOD_FUNC(core_median), -1);
  rb_define_singleton_method(mCore, "abs", RUBY_METHOD_FUNC(core_abs), 1);
  rb_define_singleton_method(mCore, "exp", RUBY_METHOD_FUNC(core_exp), 1);
  rb_define_singleton_method(mCore, "sigmoid", RUBY_METHOD_FUNC(core_sigmoid), 1);
  rb_define_singleton_method(mCore, "log", RUBY_METHOD_FUNC(core_log), 1);
  rb_define_singleton_method(mCore, "logaddexp", RUBY_METHOD_FUNC(core_logaddexp), 2);
  rb_define_singleton_method(mCore, "logsumexp", RUBY_METHOD_FUNC(core_logsumexp), -1);
  rb_define_singleton_method(mCore, "logcumsumexp", RUBY_METHOD_FUNC(core_logcumsumexp), -1);
  rb_define_singleton_method(mCore, "sin", RUBY_METHOD_FUNC(core_sin), 1);
  rb_define_singleton_method(mCore, "cos", RUBY_METHOD_FUNC(core_cos), 1);
  rb_define_singleton_method(mCore, "tan", RUBY_METHOD_FUNC(core_tan), 1);
  rb_define_singleton_method(mCore, "arcsin", RUBY_METHOD_FUNC(core_arcsin), 1);
  rb_define_singleton_method(mCore, "arccos", RUBY_METHOD_FUNC(core_arccos), 1);
  rb_define_singleton_method(mCore, "arctan", RUBY_METHOD_FUNC(core_arctan), 1);
  rb_define_singleton_method(mCore, "arcsinh", RUBY_METHOD_FUNC(core_arcsinh), 1);
  rb_define_singleton_method(mCore, "arccosh", RUBY_METHOD_FUNC(core_arccosh), 1);
  rb_define_singleton_method(mCore, "arctanh", RUBY_METHOD_FUNC(core_arctanh), 1);
  rb_define_singleton_method(mCore, "arctan2", RUBY_METHOD_FUNC(core_arctan2), 2);
  rb_define_singleton_method(mCore, "degrees", RUBY_METHOD_FUNC(core_degrees), 1);
  rb_define_singleton_method(mCore, "radians", RUBY_METHOD_FUNC(core_radians), 1);
  rb_define_singleton_method(mCore, "sinh", RUBY_METHOD_FUNC(core_sinh), 1);
  rb_define_singleton_method(mCore, "cosh", RUBY_METHOD_FUNC(core_cosh), 1);
  rb_define_singleton_method(mCore, "tanh", RUBY_METHOD_FUNC(core_tanh), 1);
  rb_define_singleton_method(mCore, "negative", RUBY_METHOD_FUNC(core_negative), 1);
  rb_define_singleton_method(mCore, "sign", RUBY_METHOD_FUNC(core_sign), 1);
  rb_define_singleton_method(mCore, "reciprocal", RUBY_METHOD_FUNC(core_reciprocal), 1);
  rb_define_singleton_method(mCore, "square", RUBY_METHOD_FUNC(core_square), 1);
  rb_define_singleton_method(mCore, "log1p", RUBY_METHOD_FUNC(core_log1p), 1);
  rb_define_singleton_method(mCore, "log2", RUBY_METHOD_FUNC(core_log2), 1);
  rb_define_singleton_method(mCore, "log10", RUBY_METHOD_FUNC(core_log10), 1);
  rb_define_singleton_method(mCore, "expm1", RUBY_METHOD_FUNC(core_expm1), 1);
  rb_define_singleton_method(mCore, "erf", RUBY_METHOD_FUNC(core_erf), 1);
  rb_define_singleton_method(mCore, "erfinv", RUBY_METHOD_FUNC(core_erfinv), 1);
  rb_define_singleton_method(mCore, "round", RUBY_METHOD_FUNC(core_round), -1);
  rb_define_singleton_method(mCore, "sqrt", RUBY_METHOD_FUNC(core_sqrt), 1);
  rb_define_singleton_method(mCore, "rsqrt", RUBY_METHOD_FUNC(core_rsqrt), 1);
  rb_define_singleton_method(mCore, "floor_divide", RUBY_METHOD_FUNC(core_floor_divide), 2);
  rb_define_singleton_method(mCore, "left_shift", RUBY_METHOD_FUNC(core_left_shift), 2);
  rb_define_singleton_method(mCore, "right_shift", RUBY_METHOD_FUNC(core_right_shift), 2);
  rb_define_singleton_method(mCore, "isfinite", RUBY_METHOD_FUNC(core_isfinite), 1);
  rb_define_singleton_method(mCore, "isnan", RUBY_METHOD_FUNC(core_isnan), 1);
  rb_define_singleton_method(mCore, "isinf", RUBY_METHOD_FUNC(core_isinf), 1);
  rb_define_singleton_method(mCore, "isposinf", RUBY_METHOD_FUNC(core_isposinf), 1);
  rb_define_singleton_method(mCore, "isneginf", RUBY_METHOD_FUNC(core_isneginf), 1);
  rb_define_singleton_method(mCore, "nan_to_num", RUBY_METHOD_FUNC(core_nan_to_num), -1);
  rb_define_singleton_method(mCore, "allclose", RUBY_METHOD_FUNC(core_allclose), -1);
  rb_define_singleton_method(mCore, "equal", RUBY_METHOD_FUNC(core_equal), 2);
  rb_define_singleton_method(mCore, "not_equal", RUBY_METHOD_FUNC(core_not_equal), 2);
  rb_define_singleton_method(mCore, "greater", RUBY_METHOD_FUNC(core_greater), 2);
  rb_define_singleton_method(mCore, "greater_equal", RUBY_METHOD_FUNC(core_greater_equal), 2);
  rb_define_singleton_method(mCore, "less", RUBY_METHOD_FUNC(core_less), 2);
  rb_define_singleton_method(mCore, "less_equal", RUBY_METHOD_FUNC(core_less_equal), 2);
  rb_define_singleton_method(mCore, "where", RUBY_METHOD_FUNC(core_where), 3);
  rb_define_singleton_method(mCore, "array_equal", RUBY_METHOD_FUNC(core_array_equal), -1);
  rb_define_singleton_method(mCore, "isclose", RUBY_METHOD_FUNC(core_isclose), -1);
  rb_define_singleton_method(mCore, "minimum", RUBY_METHOD_FUNC(core_minimum), 2);
  rb_define_singleton_method(mCore, "maximum", RUBY_METHOD_FUNC(core_maximum), 2);
  rb_define_singleton_method(mCore, "floor", RUBY_METHOD_FUNC(core_floor), 1);
  rb_define_singleton_method(mCore, "ceil", RUBY_METHOD_FUNC(core_ceil), 1);
  rb_define_singleton_method(mCore, "clip", RUBY_METHOD_FUNC(core_clip), -1);
  rb_define_singleton_method(mCore, "pad", RUBY_METHOD_FUNC(core_pad), -1);
  rb_define_singleton_method(mCore, "logical_not", RUBY_METHOD_FUNC(core_logical_not), 1);
  rb_define_singleton_method(mCore, "logical_and", RUBY_METHOD_FUNC(core_logical_and), 2);
  rb_define_singleton_method(mCore, "logical_or", RUBY_METHOD_FUNC(core_logical_or), 2);
  rb_define_singleton_method(mCore, "bitwise_and", RUBY_METHOD_FUNC(core_bitwise_and), 2);
  rb_define_singleton_method(mCore, "bitwise_or", RUBY_METHOD_FUNC(core_bitwise_or), 2);
  rb_define_singleton_method(mCore, "bitwise_xor", RUBY_METHOD_FUNC(core_bitwise_xor), 2);
  rb_define_singleton_method(mCore, "bitwise_invert", RUBY_METHOD_FUNC(core_bitwise_invert), 1);
  rb_define_singleton_method(mCore, "random_seed", RUBY_METHOD_FUNC(core_random_seed), 1);
  rb_define_singleton_method(mCore, "random_uniform", RUBY_METHOD_FUNC(core_random_uniform), -1);
  rb_define_singleton_method(mCore, "seed", RUBY_METHOD_FUNC(core_seed), 1);
  rb_define_singleton_method(mCore, "key", RUBY_METHOD_FUNC(core_key), 1);
  rb_define_singleton_method(mCore, "split", RUBY_METHOD_FUNC(core_split), -1);
  rb_define_singleton_method(mCore, "uniform", RUBY_METHOD_FUNC(core_uniform), -1);
  rb_define_singleton_method(mCore, "normal", RUBY_METHOD_FUNC(core_normal), -1);
  rb_define_singleton_method(mCore, "randint", RUBY_METHOD_FUNC(core_randint), -1);
  rb_define_singleton_method(mCore, "bernoulli", RUBY_METHOD_FUNC(core_bernoulli), -1);
  rb_define_singleton_method(mCore, "truncated_normal", RUBY_METHOD_FUNC(core_truncated_normal), -1);
  rb_define_singleton_method(mCore, "gumbel", RUBY_METHOD_FUNC(core_gumbel), -1);
  rb_define_singleton_method(mCore, "categorical", RUBY_METHOD_FUNC(core_categorical), -1);
  rb_define_singleton_method(mCore, "laplace", RUBY_METHOD_FUNC(core_laplace), -1);
  rb_define_singleton_method(mCore, "permutation", RUBY_METHOD_FUNC(core_permutation), -1);
  rb_define_singleton_method(mCore, "multivariate_normal", RUBY_METHOD_FUNC(core_multivariate_normal), -1);
  rb_define_singleton_method(mCore, "fft", RUBY_METHOD_FUNC(core_fft), -1);
  rb_define_singleton_method(mCore, "ifft", RUBY_METHOD_FUNC(core_ifft), -1);
  rb_define_singleton_method(mCore, "fft2", RUBY_METHOD_FUNC(core_fft2), -1);
  rb_define_singleton_method(mCore, "ifft2", RUBY_METHOD_FUNC(core_ifft2), -1);
  rb_define_singleton_method(mCore, "fftn", RUBY_METHOD_FUNC(core_fftn), -1);
  rb_define_singleton_method(mCore, "ifftn", RUBY_METHOD_FUNC(core_ifftn), -1);
  rb_define_singleton_method(mCore, "rfft", RUBY_METHOD_FUNC(core_rfft), -1);
  rb_define_singleton_method(mCore, "irfft", RUBY_METHOD_FUNC(core_irfft), -1);
  rb_define_singleton_method(mCore, "rfft2", RUBY_METHOD_FUNC(core_rfft2), -1);
  rb_define_singleton_method(mCore, "irfft2", RUBY_METHOD_FUNC(core_irfft2), -1);
  rb_define_singleton_method(mCore, "rfftn", RUBY_METHOD_FUNC(core_rfftn), -1);
  rb_define_singleton_method(mCore, "irfftn", RUBY_METHOD_FUNC(core_irfftn), -1);
  rb_define_singleton_method(mCore, "fftshift", RUBY_METHOD_FUNC(core_fftshift), -1);
  rb_define_singleton_method(mCore, "ifftshift", RUBY_METHOD_FUNC(core_ifftshift), -1);
  rb_define_singleton_method(mCore, "norm", RUBY_METHOD_FUNC(core_norm), -1);
  rb_define_singleton_method(mCore, "qr", RUBY_METHOD_FUNC(core_qr), -1);
  rb_define_singleton_method(mCore, "svd", RUBY_METHOD_FUNC(core_svd), -1);
  rb_define_singleton_method(mCore, "inv", RUBY_METHOD_FUNC(core_inv), -1);
  rb_define_singleton_method(mCore, "tri_inv", RUBY_METHOD_FUNC(core_tri_inv), -1);
  rb_define_singleton_method(mCore, "cholesky", RUBY_METHOD_FUNC(core_cholesky), -1);
  rb_define_singleton_method(mCore, "cholesky_inv", RUBY_METHOD_FUNC(core_cholesky_inv), -1);
  rb_define_singleton_method(mCore, "pinv", RUBY_METHOD_FUNC(core_pinv), -1);
  rb_define_singleton_method(mCore, "lu", RUBY_METHOD_FUNC(core_lu), -1);
  rb_define_singleton_method(mCore, "lu_factor", RUBY_METHOD_FUNC(core_lu_factor), -1);
  rb_define_singleton_method(mCore, "solve", RUBY_METHOD_FUNC(core_solve), -1);
  rb_define_singleton_method(mCore, "solve_triangular", RUBY_METHOD_FUNC(core_solve_triangular), -1);
  rb_define_singleton_method(mCore, "cross", RUBY_METHOD_FUNC(core_cross), -1);
  rb_define_singleton_method(mCore, "eigvals", RUBY_METHOD_FUNC(core_eigvals), -1);
  rb_define_singleton_method(mCore, "eig", RUBY_METHOD_FUNC(core_eig), -1);
  rb_define_singleton_method(mCore, "eigvalsh", RUBY_METHOD_FUNC(core_eigvalsh), -1);
  rb_define_singleton_method(mCore, "eigh", RUBY_METHOD_FUNC(core_eigh), -1);
  rb_define_singleton_method(mCore, "rms_norm", RUBY_METHOD_FUNC(core_rms_norm), -1);
  rb_define_singleton_method(mCore, "layer_norm", RUBY_METHOD_FUNC(core_layer_norm), -1);
  rb_define_singleton_method(mCore, "rope", RUBY_METHOD_FUNC(core_rope), -1);
  rb_define_singleton_method(
      mCore,
      "scaled_dot_product_attention",
      RUBY_METHOD_FUNC(core_scaled_dot_product_attention),
      -1);
  rb_define_singleton_method(
      mCore, "scaled_dot_product_attention", RUBY_METHOD_FUNC(core_scaled_dot_product_attention), -1);
  rb_define_singleton_method(mCore, "scaled_dot_product_attention", RUBY_METHOD_FUNC(core_scaled_dot_product_attention), -1);
  rb_define_singleton_method(mCore, "arange", RUBY_METHOD_FUNC(core_arange), -1);
  rb_define_singleton_method(mCore, "linspace", RUBY_METHOD_FUNC(core_linspace), -1);
  rb_define_singleton_method(mCore, "zeros", RUBY_METHOD_FUNC(core_zeros), -1);
  rb_define_singleton_method(mCore, "ones", RUBY_METHOD_FUNC(core_ones), -1);
  rb_define_singleton_method(mCore, "full", RUBY_METHOD_FUNC(core_full), -1);
  rb_define_singleton_method(mCore, "zeros_like", RUBY_METHOD_FUNC(core_zeros_like), 1);
  rb_define_singleton_method(mCore, "ones_like", RUBY_METHOD_FUNC(core_ones_like), 1);
  rb_define_singleton_method(mCore, "eye", RUBY_METHOD_FUNC(core_eye), -1);
  rb_define_singleton_method(mCore, "identity", RUBY_METHOD_FUNC(core_identity), -1);
  rb_define_singleton_method(mCore, "tri", RUBY_METHOD_FUNC(core_tri), -1);
  rb_define_singleton_method(mCore, "tril", RUBY_METHOD_FUNC(core_tril), -1);
  rb_define_singleton_method(mCore, "triu", RUBY_METHOD_FUNC(core_triu), -1);
  rb_define_singleton_method(mCore, "astype", RUBY_METHOD_FUNC(core_astype), 2);

  cDtype = rb_define_class_under(mCore, "Dtype", rb_cObject);
  rb_define_alloc_func(cDtype, dtype_alloc);
  rb_define_method(cDtype, "initialize", RUBY_METHOD_FUNC(dtype_initialize), 1);
  rb_define_method(cDtype, "size", RUBY_METHOD_FUNC(dtype_size), 0);
  rb_define_method(cDtype, "name", RUBY_METHOD_FUNC(dtype_name), 0);
  rb_define_method(cDtype, "==", RUBY_METHOD_FUNC(dtype_equal), 1);
  rb_define_method(cDtype, "eql?", RUBY_METHOD_FUNC(dtype_equal), 1);
  rb_define_method(cDtype, "hash", RUBY_METHOD_FUNC(dtype_hash), 0);
  rb_define_method(cDtype, "to_s", RUBY_METHOD_FUNC(dtype_to_s), 0);
  rb_define_method(cDtype, "inspect", RUBY_METHOD_FUNC(dtype_to_s), 0);

  rb_define_singleton_method(mCore, "bool_", RUBY_METHOD_FUNC(core_bool_), 0);
  rb_define_singleton_method(mCore, "uint8", RUBY_METHOD_FUNC(core_uint8), 0);
  rb_define_singleton_method(mCore, "uint16", RUBY_METHOD_FUNC(core_uint16), 0);
  rb_define_singleton_method(mCore, "uint32", RUBY_METHOD_FUNC(core_uint32), 0);
  rb_define_singleton_method(mCore, "uint64", RUBY_METHOD_FUNC(core_uint64), 0);
  rb_define_singleton_method(mCore, "int8", RUBY_METHOD_FUNC(core_int8), 0);
  rb_define_singleton_method(mCore, "int16", RUBY_METHOD_FUNC(core_int16), 0);
  rb_define_singleton_method(mCore, "int32", RUBY_METHOD_FUNC(core_int32), 0);
  rb_define_singleton_method(mCore, "int64", RUBY_METHOD_FUNC(core_int64), 0);
  rb_define_singleton_method(mCore, "float16", RUBY_METHOD_FUNC(core_float16), 0);
  rb_define_singleton_method(mCore, "float32", RUBY_METHOD_FUNC(core_float32), 0);
  rb_define_singleton_method(mCore, "float64", RUBY_METHOD_FUNC(core_float64), 0);
  rb_define_singleton_method(mCore, "bfloat16", RUBY_METHOD_FUNC(core_bfloat16), 0);
  rb_define_singleton_method(mCore, "complex64", RUBY_METHOD_FUNC(core_complex64), 0);

  rb_define_singleton_method(mCore, "complexfloating", RUBY_METHOD_FUNC(core_complexfloating), 0);
  rb_define_singleton_method(mCore, "floating", RUBY_METHOD_FUNC(core_floating), 0);
  rb_define_singleton_method(mCore, "inexact", RUBY_METHOD_FUNC(core_inexact), 0);
  rb_define_singleton_method(mCore, "signedinteger", RUBY_METHOD_FUNC(core_signedinteger), 0);
  rb_define_singleton_method(mCore, "unsignedinteger", RUBY_METHOD_FUNC(core_unsignedinteger), 0);
  rb_define_singleton_method(mCore, "integer", RUBY_METHOD_FUNC(core_integer), 0);
  rb_define_singleton_method(mCore, "number", RUBY_METHOD_FUNC(core_number), 0);
  rb_define_singleton_method(mCore, "generic", RUBY_METHOD_FUNC(core_generic), 0);
  rb_define_singleton_method(mCore, "issubdtype", RUBY_METHOD_FUNC(core_issubdtype), 2);

  rb_define_singleton_method(mCore, "pi", RUBY_METHOD_FUNC(core_pi), 0);
  rb_define_singleton_method(mCore, "e", RUBY_METHOD_FUNC(core_e), 0);
  rb_define_singleton_method(mCore, "euler_gamma", RUBY_METHOD_FUNC(core_euler_gamma), 0);
  rb_define_singleton_method(mCore, "inf", RUBY_METHOD_FUNC(core_inf), 0);
  rb_define_singleton_method(mCore, "nan", RUBY_METHOD_FUNC(core_nan), 0);
  rb_define_singleton_method(mCore, "newaxis", RUBY_METHOD_FUNC(core_newaxis), 0);

  cDevice = rb_define_class_under(mCore, "Device", rb_cObject);
  rb_define_alloc_func(cDevice, device_alloc);
  rb_define_method(cDevice, "initialize", RUBY_METHOD_FUNC(device_initialize), -1);
  rb_define_method(cDevice, "type", RUBY_METHOD_FUNC(device_type), 0);
  rb_define_method(cDevice, "index", RUBY_METHOD_FUNC(device_index), 0);
  rb_define_method(cDevice, "==", RUBY_METHOD_FUNC(device_equal), 1);
  rb_define_method(cDevice, "eql?", RUBY_METHOD_FUNC(device_equal), 1);
  rb_define_method(cDevice, "to_s", RUBY_METHOD_FUNC(device_to_s), 0);
  rb_define_method(cDevice, "inspect", RUBY_METHOD_FUNC(device_to_s), 0);

  rb_define_singleton_method(mCore, "cpu", RUBY_METHOD_FUNC(core_cpu), 0);
  rb_define_singleton_method(mCore, "gpu", RUBY_METHOD_FUNC(core_gpu), 0);
  rb_define_singleton_method(mCore, "default_device", RUBY_METHOD_FUNC(core_default_device), 0);
  rb_define_singleton_method(mCore, "set_default_device", RUBY_METHOD_FUNC(core_set_default_device), 1);
  rb_define_singleton_method(mCore, "is_available", RUBY_METHOD_FUNC(core_is_available), 1);
  rb_define_singleton_method(mCore, "device_count", RUBY_METHOD_FUNC(core_device_count), 1);
  rb_define_singleton_method(mCore, "device_info", RUBY_METHOD_FUNC(core_device_info), -1);

  cGroup = rb_define_class_under(mCore, "Group", rb_cObject);
  rb_define_alloc_func(cGroup, group_alloc);
  rb_define_method(cGroup, "rank", RUBY_METHOD_FUNC(group_rank), 0);
  rb_define_method(cGroup, "size", RUBY_METHOD_FUNC(group_size), 0);
  rb_define_method(cGroup, "split", RUBY_METHOD_FUNC(group_split), -1);
  rb_define_method(cGroup, "to_s", RUBY_METHOD_FUNC(group_to_s), 0);
  rb_define_method(cGroup, "inspect", RUBY_METHOD_FUNC(group_to_s), 0);

  cStream = rb_define_class_under(mCore, "Stream", rb_cObject);
  rb_define_alloc_func(cStream, stream_alloc);
  rb_define_method(cStream, "initialize", RUBY_METHOD_FUNC(stream_initialize), -1);
  rb_define_method(cStream, "index", RUBY_METHOD_FUNC(stream_index), 0);
  rb_define_method(cStream, "device", RUBY_METHOD_FUNC(stream_device), 0);
  rb_define_method(cStream, "==", RUBY_METHOD_FUNC(stream_equal), 1);
  rb_define_method(cStream, "eql?", RUBY_METHOD_FUNC(stream_equal), 1);
  rb_define_method(cStream, "to_s", RUBY_METHOD_FUNC(stream_to_s), 0);
  rb_define_method(cStream, "inspect", RUBY_METHOD_FUNC(stream_to_s), 0);

  cFunction = rb_define_class_under(mCore, "Function", rb_cObject);
  rb_define_alloc_func(cFunction, function_alloc);
  rb_define_method(cFunction, "call", RUBY_METHOD_FUNC(function_call), -1);
  rb_define_method(cFunction, "[]", RUBY_METHOD_FUNC(function_call), -1);

  cFunctionExporter = rb_define_class_under(mCore, "FunctionExporter", rb_cObject);
  rb_define_alloc_func(cFunctionExporter, function_exporter_alloc);
  rb_define_method(cFunctionExporter, "call", RUBY_METHOD_FUNC(function_exporter_call), -1);
  rb_define_method(cFunctionExporter, "[]", RUBY_METHOD_FUNC(function_exporter_call), -1);
  rb_define_method(cFunctionExporter, "close", RUBY_METHOD_FUNC(function_exporter_close), 0);

  cKernel = rb_define_class_under(mCore, "Kernel", rb_cObject);
  rb_define_alloc_func(cKernel, kernel_alloc);
  rb_define_method(cKernel, "call", RUBY_METHOD_FUNC(kernel_call), -1);
  rb_define_method(cKernel, "[]", RUBY_METHOD_FUNC(kernel_call), -1);

  rb_define_singleton_method(mCore, "default_stream", RUBY_METHOD_FUNC(core_default_stream), 1);
  rb_define_singleton_method(mCore, "set_default_stream", RUBY_METHOD_FUNC(core_set_default_stream), 1);
  rb_define_singleton_method(mCore, "new_stream", RUBY_METHOD_FUNC(core_new_stream), 1);
  rb_define_singleton_method(mCore, "stream", RUBY_METHOD_FUNC(core_stream), 1);
  rb_define_singleton_method(mCore, "synchronize", RUBY_METHOD_FUNC(core_synchronize), -1);
  rb_define_singleton_method(mCore, "eval", RUBY_METHOD_FUNC(core_eval), -1);
  rb_define_singleton_method(mCore, "async_eval", RUBY_METHOD_FUNC(core_async_eval), -1);
  rb_define_singleton_method(mCore, "disable_compile", RUBY_METHOD_FUNC(core_disable_compile), 0);
  rb_define_singleton_method(mCore, "enable_compile", RUBY_METHOD_FUNC(core_enable_compile), 0);
  rb_define_singleton_method(mCore, "jvp", RUBY_METHOD_FUNC(core_jvp), 3);
  rb_define_singleton_method(mCore, "vjp", RUBY_METHOD_FUNC(core_vjp), 3);
  rb_define_singleton_method(mCore, "compile", RUBY_METHOD_FUNC(core_compile), -1);
  rb_define_singleton_method(mCore, "checkpoint", RUBY_METHOD_FUNC(core_checkpoint), 1);
  rb_define_singleton_method(mCore, "grad", RUBY_METHOD_FUNC(core_grad), -1);
  rb_define_singleton_method(mCore, "value_and_grad", RUBY_METHOD_FUNC(core_value_and_grad), -1);
  rb_define_singleton_method(mCore, "vmap", RUBY_METHOD_FUNC(core_vmap), -1);
  rb_define_singleton_method(mCore, "export_function", RUBY_METHOD_FUNC(core_export_function), -1);
  rb_define_singleton_method(mCore, "import_function", RUBY_METHOD_FUNC(core_import_function), 1);
  rb_define_singleton_method(mCore, "exporter", RUBY_METHOD_FUNC(core_exporter), -1);
  rb_define_singleton_method(mCore, "export_to_dot", RUBY_METHOD_FUNC(core_export_to_dot), -1);
  rb_define_singleton_method(mCore, "metal_kernel", RUBY_METHOD_FUNC(core_metal_kernel), -1);
  rb_define_singleton_method(mCore, "cuda_kernel", RUBY_METHOD_FUNC(core_cuda_kernel), -1);
  rb_define_singleton_method(
      mCore,
      "precompiled_cuda_kernel",
      RUBY_METHOD_FUNC(core_precompiled_cuda_kernel),
      -1);
  rb_define_singleton_method(mCore, "precompiled_cuda_kernel", RUBY_METHOD_FUNC(core_precompiled_cuda_kernel), -1);
}
