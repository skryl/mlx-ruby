#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ruby.h>

#include "mlx/array.h"

mlx::core::array graph_ir_array_from_ruby(VALUE value);
std::vector<mlx::core::array> graph_ir_array_vector_from_ruby(VALUE value);
std::unordered_map<std::string, mlx::core::array> graph_ir_array_map_from_ruby_hash(VALUE value);
std::function<std::vector<mlx::core::array>(const mlx::core::Args&, const mlx::core::Kwargs&)>
graph_ir_args_kwargs_function_from_callable(VALUE callable);

VALUE core_native_export_graph_ir(int argc, VALUE* argv, VALUE self);
VALUE core_native_export_graph_ir_json(int argc, VALUE* argv, VALUE self);

extern "C" void init_graph_ir_native_bindings(VALUE mMLX);
