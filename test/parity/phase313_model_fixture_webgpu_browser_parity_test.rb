# frozen_string_literal: true

ENV["MLX_TEST_TIMEOUT"] = "240"

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

class Phase313ModelFixtureWebgpuBrowserParityTest < Minitest::Test
  MODELS = %i[transformer cnn mlp rnn karpathy_gpt2].freeze

  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "node is required for phase313 tests" unless command_available?("node", "--version")
    skip "playwright module is required for phase313 tests" unless node_module_available?("playwright")
    skip "onnxruntime-web module is required for phase313 tests" unless node_module_available?("onnxruntime-web")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  MODELS.each do |model_name|
    define_method(:"test_#{model_name}_webgpu_benchmark_model_runtime_parity_without_wasm_fallback") do
      fixture = benchmark_model_fixture(model_name)
      assert_fixture_webgpu_parity!(fixture)
    rescue RuntimeError => e
      if webgpu_unavailable_error?(e.message)
        if require_webgpu?
          flunk("phase313 requires WebGPU in this environment: #{e.message.lines.first&.strip || e.message}")
        end

        warn "phase313 webgpu parity skipped: #{e.message.lines.first&.strip || e.message}"
        skip "webgpu provider is unavailable in this browser/runtime environment"
      end

      raise
    end
  end

  private

  def require_webgpu?
    ENV["MLX_TEST_REQUIRE_WEBGPU"] == "1"
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
      name: "webgpu_#{model_name}_case",
      payload: payload,
      feeds: { input_name => seed.to_a },
      expected: expected
    }
  end

  def assert_fixture_webgpu_parity!(fixture)
    compatibility = MLX::GraphIR.compatibility_report(fixture.fetch(:payload))
    assert_equal 0, compatibility.fetch("unsupported_nodes"), "unsupported ops: #{compatibility.fetch('unsupported_ops').inspect}"

    Dir.mktmpdir do |dir|
      harness_dir = File.join(dir, "#{fixture.fetch(:name)}_harness")
      MLX::GraphIR.export_onnx_webgpu_harness(
        harness_dir,
        fixture.fetch(:payload),
        model_name: fixture.fetch(:name),
        execution_providers: ["webgpu"],
        benchmark_warmup_runs: 0,
        benchmark_measure_runs: 1
      )

      File.binwrite(
        File.join(harness_dir, "inputs.example.json"),
        JSON.pretty_generate(fixture.fetch(:feeds))
      )

      telemetry = MLX::GraphIR.smoke_test_onnx_webgpu_harness(
        harness_dir,
        timeout_seconds: 180,
        mock_ort: false,
        local_ort: true
      )

      assert_equal "onnx_webgpu_telemetry_v1", telemetry.fetch("format")
      assert_equal "webgpu", telemetry.fetch("selected_provider")
      assert_equal ["webgpu"], telemetry.fetch("requested_providers")
      assert_equal false, telemetry.fetch("fallback_used")
      assert_equal 0.0, telemetry.fetch("fallback_partition_ratio")
      assert_equal [], telemetry.fetch("provider_init_errors")

      output_map = telemetry.fetch("sample_outputs")
      assert_kind_of Hash, output_map
      assert_equal 1, output_map.length

      expected = fixture.fetch(:expected)
      actual = output_map.values.first
      max_diff = max_abs_diff(expected, actual)
      max_ref = max_abs_value(expected)
      tolerance = [1.0, max_ref * 1e-5].max
      assert_operator max_diff, :<=, tolerance, "max_diff=#{max_diff}, tolerance=#{tolerance}"
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

  def webgpu_unavailable_error?(message)
    text = message.to_s
    patterns = [
      /navigator\.gpu/i,
      /webgpu[^\n]*not[^\n]*(available|supported|enabled)/i,
      /gpu adapter/i,
      /failed to create gpu/i,
      /no available backend found/i
    ]
    patterns.any? { |pattern| pattern.match?(text) }
  end

  def command_available?(*argv)
    _out, _err, status = Open3.capture3(*argv)
    status.success?
  rescue Errno::ENOENT
    false
  end

  def node_module_available?(name)
    _out, _err, status = Open3.capture3(
      "node",
      "-e",
      "import(process.argv[1]).then(() => process.exit(0)).catch(() => process.exit(1))",
      name,
      chdir: File.join(RUBY_ROOT, "web")
    )
    status.success?
  rescue Errno::ENOENT
    false
  end
end
