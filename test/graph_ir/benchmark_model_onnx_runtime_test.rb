# frozen_string_literal: true

ENV["MLX_TEST_TIMEOUT"] = "120"

require "json"
require "open3"
require "stringio"
require "tmpdir"
require_relative "test_helper"
require_relative "../../examples/benchmark/transformer_example"
require_relative "../../examples/benchmark/cnn_example"
require_relative "../../examples/benchmark/mlp_example"
require_relative "../../examples/benchmark/rnn_example"
require_relative "../../examples/benchmark/gpt2mini_example"

class Phase314BenchmarkModelOnnxRuntimeParityTest < Minitest::Test
  MODELS = %i[transformer cnn mlp rnn karpathy_gpt2].freeze

  PY_RUN_ONNX_TYPED = <<~PY.freeze
    import json
    import sys

    import numpy as np
    import onnxruntime as ort

    model_path = sys.argv[1]
    feeds = json.loads(sys.argv[2])

    dtype_map = {
      "tensor(bool)": np.bool_,
      "tensor(uint8)": np.uint8,
      "tensor(uint16)": np.uint16,
      "tensor(uint32)": np.uint32,
      "tensor(uint64)": np.uint64,
      "tensor(int8)": np.int8,
      "tensor(int16)": np.int16,
      "tensor(int32)": np.int32,
      "tensor(int64)": np.int64,
      "tensor(float16)": np.float16,
      "tensor(float)": np.float32,
      "tensor(double)": np.float64,
    }

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    runtime_inputs = {}
    for inp in session.get_inputs():
      if inp.name not in feeds:
        raise RuntimeError(f"missing feed for input {inp.name!r}")
      dtype = dtype_map.get(inp.type)
      if dtype is None:
        raise RuntimeError(f"unsupported runtime input dtype: {inp.type!r}")
      runtime_inputs[inp.name] = np.array(feeds[inp.name], dtype=dtype)

    outputs = session.run(None, runtime_inputs)
    print(json.dumps([out.tolist() for out in outputs]))
  PY

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase314 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase314 tests" unless python_module_available?("onnxruntime")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  MODELS.each do |model_name|
    define_method(:"test_#{model_name}_benchmark_model_onnx_runtime_parity") do
      fixture = benchmark_model_fixture(model_name)
      assert_fixture_onnx_runtime_parity!(model_name, fixture)
    end
  end

  private

  def assert_fixture_onnx_runtime_parity!(model_name, fixture)
    compatibility = JSON.parse(
      MLX::GraphIR::Native.graph_ir_compatibility_report_json(fixture.fetch(:payload))
    )
    assert_equal 0, compatibility.fetch("unsupported_nodes"), "unsupported ops: #{compatibility.fetch('unsupported_ops').inspect}"

    actual = run_exported_onnx(fixture.fetch(:payload), fixture.fetch(:feeds)).first
    expected = fixture.fetch(:expected)
    max_diff = max_abs_diff(expected, actual)
    max_ref = max_abs_value(expected)
    tolerance = [1.0, max_ref * 1e-5].max
    assert_operator max_diff, :<=, tolerance, "#{model_name} max_diff=#{max_diff}, tolerance=#{tolerance}"
  end

  def benchmark_model_fixture(model_name)
    seed = MLX::Core.array([0.0], MLX::Core.float32)

    case model_name
    when :transformer
      example = BenchmarkExamples::TransformerExample.new(
        batch_size: 1,
        sequence_length: 8,
        target_sequence_length: 4,
        dims: 32,
        num_heads: 4,
        num_layers: 1,
        dtype: MLX::Core.float32
      )
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :cnn
      example = BenchmarkExamples::CnnExample.new(batch_size: 1, dtype: MLX::Core.float32)
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :mlp
      example = BenchmarkExamples::MlpExample.new(batch_size: 1, dims: 32, dtype: MLX::Core.float32)
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :rnn
      example = BenchmarkExamples::RnnExample.new(batch_size: 1, sequence_length: 8, dims: 32, dtype: MLX::Core.float32)
      trace = ->(_seed) { example.run_step }
      expected = example.run_step.to_a
    when :karpathy_gpt2
      example = BenchmarkExamples::Gpt2MiniExample.new(
        batch_size: 1,
        sequence_length: 16,
        dims: 32,
        num_heads: 4,
        num_layers: 1,
        repo_root: File.expand_path("../..", __dir__)
      )
      model = example.instance_variable_get(:@model)
      sample_input, = example.send(:batch_for_step, 0)
      trace = ->(_seed) { model.call(sample_input) }
      expected = model.call(sample_input).to_a
    else
      raise ArgumentError, "unknown benchmark model fixture: #{model_name.inspect}"
    end

    payload = JSON.parse(MLX::GraphIR.export_graph_ir_json(trace, seed))
    input_name = payload.fetch("inputs").first.fetch("name")
    {
      payload: payload,
      feeds: { input_name => seed.to_a },
      expected: expected
    }
  end

  def python_module_available?(name)
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('#{name}') else 1)"
    )
    status.success?
  end

  def run_exported_onnx(payload, feeds)
    Dir.mktmpdir do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "benchmark_model_case")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX_TYPED, onnx_path, JSON.generate(feeds))
      raise "onnxruntime execution failed:\n#{err}" unless status.success?

      return JSON.parse(out)
    end
  end

  def max_abs_diff(expected, actual)
    if expected.is_a?(Array)
      return 0.0 if expected.empty? && actual.empty?

      expected.zip(actual).map { |exp_item, act_item| max_abs_diff(exp_item, act_item) }.max
    else
      (expected.to_f - actual.to_f).abs
    end
  end

  def max_abs_value(value)
    if value.is_a?(Array)
      return 0.0 if value.empty?

      value.map { |item| max_abs_value(item) }.max
    else
      value.to_f.abs
    end
  end
end
