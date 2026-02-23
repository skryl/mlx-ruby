# frozen_string_literal: true

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"

class Phase306ConvolutionTransposeLoweringParityTest < Minitest::Test
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

  def test_conv_transpose_node_is_lowered_to_onnx_convtranspose
    payload, = conv_transpose2d_payload_with_values
    assert_equal ["Convolution"], payload.fetch("nodes").map { |node| node.fetch("op") }

    stub = TestSupport.parse_onnx_stub(payload)
    onnx_nodes = stub.fetch("graph").fetch("nodes")
    assert_equal ["Transpose", "Transpose", "ConvTranspose", "Transpose"], onnx_nodes.map { |node| node.fetch("op_type") }

    conv_node = onnx_nodes[2]
    attrs = conv_node.fetch("attributes")
    assert_equal [2, 2], attrs.fetch("strides")
    assert_equal [1, 1], attrs.fetch("dilations")
    assert_equal [0, 0, 0, 0], attrs.fetch("pads")
    assert_equal [1, 0], attrs.fetch("output_padding")
    assert_equal 1, attrs.fetch("group")
  end

  def test_conv_transpose_runtime_parity
    skip "python onnx module is required for phase306 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase306 tests" unless python_module_available?("onnxruntime")

    payload, x, weight = conv_transpose2d_payload_with_values
    expected = MLX::Core.conv_transpose2d(
      MLX::Core.array(x, MLX::Core.float32),
      MLX::Core.array(weight, MLX::Core.float32),
      [2, 2],
      [0, 0],
      [1, 1],
      [1, 0],
      1
    ).to_a

    result = run_exported_onnx(payload, {
      payload.fetch("inputs")[0].fetch("name") => x,
      payload.fetch("inputs")[1].fetch("name") => weight
    })
    assert_nested_close(expected, result.first, atol: 1e-4)
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

  def conv_transpose2d_payload_with_values
    fun = lambda do |x, weight|
      MLX::Core.conv_transpose2d(x, weight, [2, 2], [0, 0], [1, 1], [1, 0], 1)
    end

    x = [[[[1.0], [2.0]], [[3.0], [4.0]]]]
    weight = [[[[1.0], [0.5]], [[-0.5], [1.5]]]]
    x_array = MLX::Core.array(x, MLX::Core.float32)
    weight_array = MLX::Core.array(weight, MLX::Core.float32)
    payload = JSON.parse(MLX::ONNX.export_graph_ir_json(fun, x_array, weight_array))
    [payload, x, weight]
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "conv_transpose_case")
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
