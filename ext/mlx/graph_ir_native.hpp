#pragma once

#include <ruby.h>

VALUE core_native_export_graph_ir(int argc, VALUE* argv, VALUE self);
VALUE core_native_export_graph_ir_json(int argc, VALUE* argv, VALUE self);

extern "C" void init_graph_ir_native_bindings(VALUE mMLX);
