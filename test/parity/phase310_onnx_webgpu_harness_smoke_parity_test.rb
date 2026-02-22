# frozen_string_literal: true

require "json"
require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase310OnnxWebgpuHarnessSmokeParityTest < Minitest::Test
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

  def test_smoke_test_onnx_webgpu_harness_mock_mode_emits_telemetry
    skip "python onnx module is required for phase310 tests" unless python_module_available?("onnx")
    skip "node is required for phase310 tests" unless command_available?("node", "--version")
    skip "playwright module is required for phase310 tests" unless playwright_module_available?

    payload = sample_payload

    Dir.mktmpdir do |dir|
      harness_dir = File.join(dir, "web_harness")
      MLX::GraphIR.export_onnx_webgpu_harness(
        harness_dir,
        payload,
        benchmark_warmup_runs: 0,
        benchmark_measure_runs: 2
      )

      telemetry = MLX::GraphIR.smoke_test_onnx_webgpu_harness(
        harness_dir,
        mock_ort: true,
        timeout_seconds: 30
      )

      assert_equal "onnx_webgpu_telemetry_v1", telemetry.fetch("format")
      assert_equal "wasm", telemetry.fetch("selected_provider")
      assert_equal %w[webgpu wasm], telemetry.fetch("requested_providers")
      assert_equal true, telemetry.fetch("fallback_used")
      assert_equal 1.0, telemetry.fetch("fallback_partition_ratio")
      assert_kind_of Array, telemetry.fetch("provider_init_errors")
      assert_operator telemetry.fetch("provider_init_errors").length, :>=, 1
      assert_kind_of Array, telemetry.fetch("run_timings_ms")
      assert_equal 2, telemetry.fetch("run_timings_ms").length
    end
  end

  private

  def command_available?(*argv)
    _out, _err, status = Open3.capture3(*argv)
    status.success?
  rescue Errno::ENOENT
    false
  end

  def python_module_available?(name)
    python_bin = ENV.fetch("PYTHON", "python3")
    _out, _err, status = Open3.capture3(python_bin, "-c", "import #{name}")
    status.success?
  rescue Errno::ENOENT
    false
  end

  def playwright_module_available?
    _out, _err, status = Open3.capture3(
      "node",
      "-e",
      "import('playwright').then(() => process.exit(0)).catch(() => process.exit(1))",
      chdir: File.join(RUBY_ROOT, "web")
    )
    status.success?
  rescue Errno::ENOENT
    false
  end

  def sample_payload
    {
      "ir_version" => 1,
      "shapeless" => false,
      "inputs" => [{ "name" => "x", "shape" => [2], "dtype" => "float32" }],
      "keyword_inputs" => [],
      "outputs" => [{ "name" => "z", "shape" => [2], "dtype" => "float32" }],
      "constants" => [
        {
          "name" => "c",
          "shape" => [2],
          "dtype" => "float32",
          "values" => [1.0, -1.0]
        }
      ],
      "nodes" => [{ "op" => "Add", "inputs" => %w[x c], "outputs" => ["z"] }]
    }
  end
end
