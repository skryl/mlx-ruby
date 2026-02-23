# frozen_string_literal: true

require "json"
require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase337NativeOnnxBinaryPythonOracleParityTest < Minitest::Test
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

    def cast_initializer_values(values, dtype_name):
      if dtype_name in bool_dtypes:
        return [bool(v) for v in values]
      if dtype_name in int_dtypes:
        return [int(v) for v in values]
      if dtype_name in float_dtypes:
        return [float(v) for v in values]
      if dtype_name == "complex64":
        out = []
        for value in values:
          if isinstance(value, dict) and "__mlx_complex__" in value:
            pair = value["__mlx_complex__"]
            out.append(complex(float(pair[0]), float(pair[1])))
          elif isinstance(value, (bool, int, float)):
            out.append(complex(float(value), 0.0))
          else:
            raise RuntimeError(f"unsupported complex64 initializer value: {value!r}")
        return out
      raise RuntimeError(f"unsupported initializer dtype: {dtype_name!r}")

    def tensor_value_info(spec):
      return helper.make_tensor_value_info(
        spec["name"],
        dtype_map[spec["dtype"]],
        [int(dim) for dim in spec["shape"]],
      )

    def initializer_tensor(spec):
      dtype_name = spec["dtype"]
      elem_type = dtype_map[dtype_name]
      dims = [int(dim) for dim in spec["shape"]]
      values = flatten_values(spec["values"])
      expected = expected_value_count(dims)
      if len(values) != expected:
        raise RuntimeError(f"initializer {spec['name']!r} has wrong value count")
      cast_values = cast_initializer_values(values, dtype_name)
      return helper.make_tensor(spec["name"], elem_type, dims, cast_values)

    graph_spec = stub["graph"]
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

    graph = helper.make_graph(
      nodes=nodes,
      name=graph_spec["name"],
      inputs=[tensor_value_info(spec) for spec in graph_spec["inputs"]],
      outputs=[tensor_value_info(spec) for spec in graph_spec["outputs"]],
      initializer=[initializer_tensor(spec) for spec in graph_spec.get("initializers", [])],
    )

    model = helper.make_model(
      graph,
      producer_name=stub.get("producer_name", "mlx-ruby"),
      opset_imports=[helper.make_operatorsetid("", int(stub["opset"]))],
    )
    if use_external_data:
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

  PY_MODEL_SUMMARY = <<~PY.freeze
    import hashlib
    import json
    import sys

    import onnx
    from onnx import helper

    model_path = sys.argv[1]
    model = onnx.load(model_path, load_external_data=True)

    def norm(value):
      if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
      if isinstance(value, list):
        return [norm(item) for item in value]
      if isinstance(value, tuple):
        return [norm(item) for item in value]
      if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
      return value

    nodes = []
    for node in model.graph.node:
      attrs = []
      for attr in node.attribute:
        attrs.append([attr.name, norm(helper.get_attribute_value(attr))])
      nodes.append({
        "name": node.name,
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": sorted(attrs, key=lambda item: item[0]),
      })

    initializers = []
    for tensor in model.graph.initializer:
      initializers.append({
        "name": tensor.name,
        "data_type": int(tensor.data_type),
        "dims": list(tensor.dims),
        "raw_sha256": hashlib.sha256(tensor.raw_data).hexdigest(),
        "data_location": int(tensor.data_location),
        "external_data": sorted([[entry.key, entry.value] for entry in tensor.external_data]),
      })

    out = {
      "opset": int(model.opset_import[0].version),
      "producer_name": model.producer_name,
      "inputs": [
        {
          "name": info.name,
          "elem_type": int(info.type.tensor_type.elem_type),
          "shape": [int(dim.dim_value) for dim in info.type.tensor_type.shape.dim],
        }
        for info in model.graph.input
      ],
      "outputs": [
        {
          "name": info.name,
          "elem_type": int(info.type.tensor_type.elem_type),
          "shape": [int(dim.dim_value) for dim in info.type.tensor_type.shape.dim],
        }
        for info in model.graph.output
      ],
      "nodes": nodes,
      "initializers": sorted(initializers, key=lambda item: item["name"]),
    }
    print(json.dumps(out, sort_keys=True))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_native_binary_matches_python_oracle_without_external_data
    skip_unless_python_onnx!
    compare_native_and_python_oracle(external_data: false, threshold: 1024)
  end

  def test_native_binary_matches_python_oracle_with_external_data
    skip_unless_python_onnx!
    compare_native_and_python_oracle(external_data: true, threshold: 0)
  end

  private

  def skip_unless_python_onnx!
    skip "python onnx module is required for phase337 tests" unless python_module_available?("onnx")
  end

  def python_module_available?(name)
    python_bin = ENV.fetch("PYTHON", "python3")
    _out, _err, status = Open3.capture3(python_bin, "-c", "import #{name}")
    status.success?
  rescue Errno::ENOENT
    false
  end

  def compare_native_and_python_oracle(external_data:, threshold:)
    payload = exported_payload
    onnx_stub_json = MLX::GraphIR.graph_ir_to_onnx_json(payload, model_name: "phase337_oracle_case")
    python_bin = ENV.fetch("PYTHON", "python3")

    Dir.mktmpdir do |dir|
      stub_path = File.join(dir, "model.stub.json")
      native_path = File.join(dir, "native.onnx")
      python_path = File.join(dir, "python.onnx")
      File.binwrite(stub_path, onnx_stub_json)

      MLX::GraphIR.graph_ir_to_onnx(
        native_path,
        payload,
        model_name: "phase337_oracle_case",
        external_data: external_data,
        external_data_file: "model.data",
        external_data_size_threshold: threshold
      )

      run_python!(
        python_bin,
        "-c",
        PY_BUILD_ONNX_FROM_STUB,
        stub_path,
        python_path,
        (external_data ? "1" : "0"),
        "model.data",
        threshold.to_s
      )

      native_summary = model_summary_json(python_bin, native_path)
      python_summary = model_summary_json(python_bin, python_path)
      assert_equal python_summary, native_summary
    end
  end

  def model_summary_json(python_bin, model_path)
    stdout, _stderr = run_python!(python_bin, "-c", PY_MODEL_SUMMARY, model_path)
    JSON.parse(stdout)
  end

  def run_python!(*argv)
    stdout, stderr, status = Open3.capture3(*argv)
    return [stdout, stderr] if status.success?

    raise <<~MSG
      python command failed: #{argv.join(" ")}
      stdout:
      #{stdout}
      stderr:
      #{stderr}
    MSG
  end

  def exported_payload
    fun = lambda do |x, y:|
      MLX::Core.add(MLX::Core.exp(x), y)
    end
    x = MLX::Core.array([[1.0, 2.0], [3.0, 4.0]], MLX::Core.float32)
    y = MLX::Core.array([[0.5, 0.25], [0.125, 0.0625]], MLX::Core.float32)
    MLX::GraphIR.export_graph_ir(fun, x, y: y)
  end
end
