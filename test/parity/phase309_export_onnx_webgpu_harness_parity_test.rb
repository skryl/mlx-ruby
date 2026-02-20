# frozen_string_literal: true

require "json"
require "open3"
require "tmpdir"
require_relative "test_helper"

class Phase309ExportOnnxWebgpuHarnessParityTest < Minitest::Test
  def setup
    TestSupport.build_native_extension!
    $LOAD_PATH.unshift(File.join(RUBY_ROOT, "lib"))
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
    skip "python onnx module is required for phase309 tests" unless python_module_available?("onnx")
  end

  def teardown
    MLX::Core.set_default_device(@previous_device) if defined?(@previous_device) && @previous_device
    $LOAD_PATH.delete(File.join(RUBY_ROOT, "lib"))
  end

  def test_export_onnx_webgpu_harness_writes_browser_assets_and_manifest
    payload = sample_payload

    Dir.mktmpdir do |dir|
      out_dir = File.join(dir, "web_harness")
      manifest = MLX::GraphIR.export_onnx_webgpu_harness(
        out_dir,
        payload,
        model_name: "webgpu_case",
        benchmark_warmup_runs: 1,
        benchmark_measure_runs: 3
      )

      assert_equal "onnx_webgpu_harness_v1", manifest.fetch("format")
      assert_equal %w[webgpu wasm], manifest.fetch("execution_providers")
      assert_equal "model.onnx", manifest.fetch("model")

      assert File.file?(File.join(out_dir, "model.onnx"))
      assert File.file?(File.join(out_dir, "harness.manifest.json"))
      assert File.file?(File.join(out_dir, "inputs.example.json"))
      assert File.file?(File.join(out_dir, "index.html"))
      assert File.file?(File.join(out_dir, "harness.js"))

      manifest_on_disk = JSON.parse(File.binread(File.join(out_dir, "harness.manifest.json")))
      assert_equal "onnx_webgpu_harness_v1", manifest_on_disk.fetch("format")
      assert_equal [1], manifest_on_disk.fetch("benchmark").values_at("warmup_runs")
      assert_equal [3], manifest_on_disk.fetch("benchmark").values_at("measure_runs")

      html = File.binread(File.join(out_dir, "index.html"))
      js = File.binread(File.join(out_dir, "harness.js"))
      assert_includes html, "./harness.js"
      assert_includes js, "onnxruntime-web"
      assert_includes js, "webgpu"
      assert_includes js, "wasm"
    end
  end

  private

  def python_module_available?(name)
    _out, _err, status = Open3.capture3(
      "python3",
      "-c",
      "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('#{name}') else 1)"
    )
    status.success?
  rescue StandardError
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
