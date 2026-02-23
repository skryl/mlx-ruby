# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase305ConvolutionLoweringParityTest < Minitest::Test
  PY_RUN_ONNX = <<~PY.freeze
    import json
    import sys

    import numpy as np
    import onnxruntime as ort

    model_path = sys.argv[1]
    feeds = json.loads(sys.argv[2])
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    runtime_inputs = {}
    for inp in session.get_inputs():
      runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=np.float32)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
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

  def test_convolution_node_is_lowered_to_transpose_conv_transpose
    payload, = convolution2d_payload_with_values
    assert_equal ["Convolution"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = TestSupport.parse_onnx_stub(payload)
    onnx_nodes = stub.fetch("graph").fetch("nodes")
    assert_equal ["Transpose", "Transpose", "Conv", "Transpose"], onnx_nodes.map { |node| node.fetch("op_type") }

    conv_node = onnx_nodes[2]
    attrs = conv_node.fetch("attributes")
    assert_equal [1, 1], attrs.fetch("strides")
    assert_equal [0, 0, 0, 0], attrs.fetch("pads")
    assert_equal [1, 1], attrs.fetch("dilations")
    assert_equal 1, attrs.fetch("group")
  end

  def test_convolution_runtime_parity
    skip "python onnx module is required for phase305 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase305 tests" unless python_module_available?("onnxruntime")

    payload, x, weight = convolution2d_payload_with_values
    expected = MLX::Core.conv2d(
      MLX::Core.array(x, MLX::Core.float32),
      MLX::Core.array(weight, MLX::Core.float32)
    ).to_a

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => weight
    })
    assert_nested_close(expected, result.first, atol: 1e-4)
  end

  def test_conv_general_with_input_dilation_is_reported_unsupported
    payload = conv_general_input_dilation_payload
    report = JSON.parse(
      MLX::ONNX::Native.ir_compatibility_report_json(payload)
    )
    assert_equal 1, report.fetch("unsupported_nodes")
    assert_equal ["Convolution"], report.fetch("unsupported_ops")

    error = assert_raises(NotImplementedError) do
      TestSupport.parse_onnx_stub(payload)
    end
    assert_match("input_dilation", error.message)
  end

  def test_flip_convolution_propagates_dtype_to_downstream_add
    payload = flip_convolution_add_int32_payload

    stub = TestSupport.parse_onnx_stub(payload)
    onnx_nodes = stub.fetch("graph").fetch("nodes")
    assert_equal ["Transpose", "Transpose", "ConvTranspose", "Transpose", "Cast", "Add"],
                 onnx_nodes.map { |node| node.fetch("op_type") }

    cast = onnx_nodes[-2]
    assert_equal ["B"], cast.fetch("inputs")
    assert_equal({ "to" => "FLOAT" }, cast.fetch("attributes"))

    add = onnx_nodes[-1]
    assert_equal "conv", add.fetch("inputs").first
    refute_equal "B", add.fetch("inputs")[1]
  end

  private

  def python_module_available?(name)
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('#{name}') else 1)"
    )
    status.success?
  end

  def convolution2d_payload_with_values
    fun = lambda do |x, weight|
      MLX::Core.conv2d(x, weight)
    end

    x = [[[[1.0], [2.0]], [[3.0], [4.0]]]]
    weight = [[[[1.0], [1.0]], [[1.0], [1.0]]]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    weight_array = MLX::Core.array(weight, MLX::Core.float32)
    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x_array, weight_array))
    [payload, x, weight]
  end

  def conv_general_input_dilation_payload
    fun = lambda do |x, weight|
      MLX::Core.conv_general(x, weight, 1, 0, 1, 2, 1, false)
    end

    x = MLX::Core.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], MLX::Core.float32)
    weight = MLX::Core.array([[[[1.0], [1.0]], [[1.0], [1.0]]]], MLX::Core.float32)
    JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x, weight))
  end

  def flip_convolution_add_int32_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [
        { "name" => "A", "shape" => [1, 2, 2, 1], "dtype" => "float32" }
      ],
      "keyword_inputs" => [],
      "outputs" => [
        { "name" => "Y", "shape" => [1, 5, 4, 1], "dtype" => "float32" }
      ],
      "constants" => [
        {
          "name" => "W",
          "shape" => [1, 2, 2, 1],
          "dtype" => "float32",
          "values" => [[[[1.0], [0.5]], [[-0.5], [1.5]]]]
        },
        {
          "name" => "B",
          "shape" => [],
          "dtype" => "int32",
          "values" => 1
        }
      ],
      "nodes" => [
        {
          "op" => "Convolution",
          "inputs" => ["A", "W"],
          "outputs" => ["conv"],
          "arguments" => [[1, 1], [1, 1], [2, 1], [1, 1], [2, 2], 1, true]
        },
        {
          "op" => "Add",
          "inputs" => ["conv", "B"],
          "outputs" => ["Y"],
          "arguments" => []
        }
      ]
    }
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "convolution_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end

  def assert_nested_close(expected, actual, atol: 1e-5)
    if expected.is_a?(Array)
      assert_equal expected.length, actual.length
      expected.each_with_index do |item, index|
        assert_nested_close(item, actual[index], atol: atol)
      end
      return
    end

    assert_in_delta expected.to_f, actual.to_f, atol
  end
end
