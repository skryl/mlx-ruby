# frozen_string_literal: true

require "json"
require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase311OnnxWebgpuHarnessRealRuntimeSmokeParityTest < Minitest::Test
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

  def test_smoke_test_onnx_webgpu_harness_real_wasm_runtime
    unless ENV["MLX_TEST_WEB_SMOKE_REAL"] == "1"
      skip "set MLX_TEST_WEB_SMOKE_REAL=1 to enable phase311 real-runtime smoke test"
    end
    skip "node is required for phase311 tests" unless command_available?("node", "--version")
    skip "playwright module is required for phase311 tests" unless node_module_available?("playwright")
    skip "onnxruntime-web module is required for phase311 tests" unless node_module_available?("onnxruntime-web")

    Dir.mktmpdir do |dir|
      harness_dir = File.join(dir, "web_harness")
      MLX::Core.export_onnx_webgpu_harness(
        harness_dir,
        sample_payload,
        execution_providers: ["wasm"],
        benchmark_warmup_runs: 0,
        benchmark_measure_runs: 2
      )

      telemetry = MLX::Core.smoke_test_onnx_webgpu_harness(
        harness_dir,
        timeout_seconds: 45,
        mock_ort: false,
        local_ort: true
      )

      assert_equal "onnx_webgpu_telemetry_v1", telemetry.fetch("format")
      assert_equal "wasm", telemetry.fetch("selected_provider")
      assert_equal ["wasm"], telemetry.fetch("requested_providers")
      assert_equal false, telemetry.fetch("fallback_used")
      assert_equal 0.0, telemetry.fetch("fallback_partition_ratio")
      assert_equal [], telemetry.fetch("provider_init_errors")
      assert_kind_of Array, telemetry.fetch("run_timings_ms")
      assert_equal 2, telemetry.fetch("run_timings_ms").length
      assert telemetry.fetch("model_load_latency_ms").is_a?(Numeric)
      assert telemetry.fetch("first_inference_latency_ms").is_a?(Numeric)
      assert telemetry.fetch("steady_state_inference_latency_ms").is_a?(Numeric)
    end
  end

  private

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
