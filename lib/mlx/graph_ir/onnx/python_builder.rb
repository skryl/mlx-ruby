# frozen_string_literal: true

require "open3"

module MLX
  module GraphIR
    module ONNX
      module PythonBuilder
        module_function

        PY_BUILD_ONNX_FROM_STUB = <<~PY.freeze
          import json
          import sys

          import onnx
          from onnx import TensorProto, helper, numpy_helper

          stub_path = sys.argv[1]
          output_path = sys.argv[2]
          use_external_data = len(sys.argv) >= 4 and sys.argv[3] == "1"
          external_data_location = sys.argv[4] if len(sys.argv) >= 5 else "weights.bin"
          external_data_threshold = int(sys.argv[5]) if len(sys.argv) >= 6 else 1024
          if stub_path == "-":
              stub = json.load(sys.stdin)
          else:
              with open(stub_path, "r", encoding="utf-8") as f:
                  stub = json.load(f)

          dtype_map = {
              "bool": TensorProto.BOOL,
              "bool_": TensorProto.BOOL,
              "uint8": TensorProto.UINT8,
              "uint16": TensorProto.UINT16,
              "uint32": TensorProto.UINT32,
              "uint64": TensorProto.UINT64,
              "int8": TensorProto.INT8,
              "int16": TensorProto.INT16,
              "int32": TensorProto.INT32,
              "int64": TensorProto.INT64,
              "float16": TensorProto.FLOAT16,
              "float32": TensorProto.FLOAT,
              "float64": TensorProto.DOUBLE,
              "bfloat16": TensorProto.BFLOAT16,
              "complex64": TensorProto.COMPLEX64,
          }

          graph_spec = stub["graph"]
          bool_dtypes = {"bool", "bool_"}
          int_dtypes = {"uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"}
          float_dtypes = {"float16", "float32", "float64", "bfloat16"}

          def flatten_values(value):
              if isinstance(value, list):
                  out = []
                  for item in value:
                      out.extend(flatten_values(item))
                  return out
              return [value]

          def expected_value_count(dims):
              if not dims:
                  return 1
              total = 1
              for dim in dims:
                  total *= int(dim)
              return total

          def cast_initializer_values(values, dtype_name, bool_dtypes, int_dtypes, float_dtypes):
              if dtype_name in bool_dtypes:
                  return [bool(v) for v in values]
              if dtype_name in int_dtypes:
                  return [int(v) for v in values]
              if dtype_name in float_dtypes:
                  return [float(v) for v in values]
              if dtype_name == "complex64":
                  complex_values = []
                  for value in values:
                      if isinstance(value, dict) and "__mlx_complex__" in value:
                          pair = value["__mlx_complex__"]
                          if not isinstance(pair, list) or len(pair) != 2:
                              raise RuntimeError(
                                  f"invalid complex marker value: {value!r}; expected {{'__mlx_complex__': [real, imag]}}"
                              )
                          complex_values.append(complex(float(pair[0]), float(pair[1])))
                      elif isinstance(value, bool):
                          complex_values.append(complex(float(value), 0.0))
                      elif isinstance(value, (int, float)):
                          complex_values.append(complex(float(value), 0.0))
                      elif isinstance(value, str):
                          text = value.strip()
                          if text.endswith(("i", "I")):
                              text = f"{text[:-1]}j"
                          try:
                              complex_values.append(complex(text))
                          except ValueError as exc:
                              raise RuntimeError(
                                  f"invalid complex64 initializer string: {value!r}; expected Python complex format or Ruby trailing-i format"
                              ) from exc
                      else:
                          raise RuntimeError(f"unsupported complex64 initializer value: {value!r}")
                  return complex_values
              raise RuntimeError(f"initializer dtype not yet supported: {dtype_name!r}")

          def tensor_value_info(spec):
              elem_type = dtype_map[spec["dtype"]]
              shape = [int(dim) for dim in spec["shape"]]
              return helper.make_tensor_value_info(spec["name"], elem_type, shape)

          def initializer_tensor(spec, dtype_map, bool_dtypes, int_dtypes, float_dtypes):
              dtype_name = spec["dtype"]
              elem_type = dtype_map[dtype_name]
              dims = [int(dim) for dim in spec["shape"]]
              values = flatten_values(spec["values"])
              expected = expected_value_count(dims)
              if len(values) != expected:
                  raise RuntimeError(
                      f"initializer {spec['name']!r} has {len(values)} values but shape expects {expected}"
                  )
              cast_values = cast_initializer_values(values, dtype_name, bool_dtypes, int_dtypes, float_dtypes)
              return helper.make_tensor(spec["name"], elem_type, dims, cast_values)

          nodes = []
          for node in graph_spec["nodes"]:
              attrs = dict(node.get("attributes", {}))
              if node["op_type"] == "Cast" and isinstance(attrs.get("to"), str):
                  attrs["to"] = getattr(TensorProto, attrs["to"])
              nodes.append(
                  helper.make_node(
                      node["op_type"],
                      list(node["inputs"]),
                      list(node["outputs"]),
                      name=node.get("name", ""),
                      **attrs,
                  )
              )

          initializers = [
              initializer_tensor(spec, dtype_map, bool_dtypes, int_dtypes, float_dtypes)
              for spec in graph_spec.get("initializers", [])
          ]

          graph = helper.make_graph(
              nodes=nodes,
              name=graph_spec["name"],
              inputs=[tensor_value_info(spec) for spec in graph_spec["inputs"]],
              outputs=[tensor_value_info(spec) for spec in graph_spec["outputs"]],
              initializer=initializers,
          )

          model = helper.make_model(
              graph,
              producer_name=stub.get("producer_name", "mlx-ruby"),
              opset_imports=[helper.make_operatorsetid("", int(stub["opset"]))],
          )
          if use_external_data:
              # External data conversion relies on raw_data encoding.
              for index, initializer in enumerate(model.graph.initializer):
                  array = numpy_helper.to_array(initializer)
                  model.graph.initializer[index].CopyFrom(
                      numpy_helper.from_array(array, initializer.name)
                  )
          onnx.checker.check_model(model)
          if use_external_data:
              onnx.save_model(
                  model,
                  output_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=external_data_location,
                  size_threshold=external_data_threshold,
              )
          else:
              onnx.save(model, output_path)
        PY

        def build_from_stub!(
          stub_path:,
          output_path:,
          external_data:,
          external_data_file:,
          external_data_size_threshold:,
          python_bin: ENV.fetch("PYTHON", "python3")
        )
          run_python!(
            python_bin,
            "-c",
            PY_BUILD_ONNX_FROM_STUB,
            stub_path,
            output_path,
            external_data ? "1" : "0",
            external_data_file,
            external_data_size_threshold.to_s,
            stdin_data: nil
          )
        end

        def build_from_stub_json!(
          stub_json:,
          output_path:,
          external_data:,
          external_data_file:,
          external_data_size_threshold:,
          python_bin: ENV.fetch("PYTHON", "python3")
        )
          run_python!(
            python_bin,
            "-c",
            PY_BUILD_ONNX_FROM_STUB,
            "-",
            output_path,
            external_data ? "1" : "0",
            external_data_file,
            external_data_size_threshold.to_s,
            stdin_data: stub_json
          )
        end

        def run_python!(*argv, stdin_data:)
          stdout, stderr, status = Open3.capture3(*argv, stdin_data: stdin_data)
          return if status.success?

          raise RuntimeError, <<~MSG
            python command failed: #{argv.join(" ")}
            stdout:
            #{stdout}
            stderr:
            #{stderr}
          MSG
        end
        private_class_method :run_python!
      end
    end
  end
end
