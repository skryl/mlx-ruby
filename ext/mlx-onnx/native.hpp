#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ruby.h>

#include "mlx/array.h"
#include "mlx/export.h"

mlx::core::array onnx_array_from_ruby(VALUE value);
std::vector<mlx::core::array> onnx_array_vector_from_ruby(VALUE value);
std::unordered_map<std::string, mlx::core::array> onnx_array_map_from_ruby_hash(VALUE value);
std::function<std::vector<mlx::core::array>(const mlx::core::Args&, const mlx::core::Kwargs&)>
onnx_args_kwargs_function_from_callable(VALUE callable);

extern "C" void init_onnx_native_bindings(VALUE mMLX);
