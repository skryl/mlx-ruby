# frozen_string_literal: true

require "json"
require "open3"
require "rbconfig"
require "stringio"
require "tempfile"
require "tmpdir"
require_relative "test_helper"

class Phase324ExamplesSubmoduleOnnxRuntimeParityTest < Minitest::Test
  def self.current_test_timeout_seconds
    1200
  end

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
    named = {}
    for meta, value in zip(session.get_outputs(), outputs):
      named[meta.name] = value.tolist()
    print(json.dumps(named))
  PY

  WEBGPU_OUTPUT_ABS_TOLERANCE = 1.0
  WEBGPU_OUTPUT_REL_TOLERANCE = 1e-5

  def setup
    TestSupport.build_native_extension!
    @repo_root = RUBY_ROOT
    @submodule_root = File.join(@repo_root, "submodules", "mlx-ruby-examples")
    @submodule_runner = File.join(@submodule_root, "benchmark", "runner.rb")
    @force_cpu = File.join(@submodule_root, "benchmark", "ruby", "force_cpu.rb")
    @capture_hook = File.join(@repo_root, "examples", "benchmark", "benchmark_mlx_examples", "onnx_capture_hook.rb")
    @lib_root = File.join(@repo_root, "lib")

    skip "mlx-ruby-examples submodule is not available" unless File.exist?(@submodule_runner)
    skip "python onnx module is required for phase324 tests" unless python_module_available?("onnx")
    skip "python onnxruntime module is required for phase324 tests" unless python_module_available?("onnxruntime")

    $LOAD_PATH.unshift(@lib_root)
    require "mlx"
    @previous_device = MLX::Core.default_device
    MLX::Core.set_default_device(MLX::Core.cpu)
  end

  def teardown
    if defined?(@previous_device) && @previous_device
      MLX::Core.set_default_device(@previous_device)
    end
    $LOAD_PATH.delete(@lib_root) if defined?(@lib_root)
  end

  def test_all_examples_submodule_models_export_and_run_in_onnxruntime
    specs = load_submodule_specs
    assert_operator specs.length, :>=, 20, "expected full examples model set, got #{specs.length}"

    specs.each do |spec|
      model_id = spec.fetch("id")
      capture = capture_onnx_fixture(spec)
      payload = capture.fetch("payload")
      compatibility = JSON.parse(
        MLX::ONNX::Native.ir_compatibility_report_json(payload)
      )
      assert_equal 0, compatibility.fetch("unsupported_nodes"), "#{model_id} unsupported ops: #{compatibility.fetch('unsupported_ops').inspect}"

      output_map = run_exported_onnx(payload, capture.fetch("feeds"), model_id: model_id)
      output_names = capture.fetch("output_names")
      expected_outputs = capture.fetch("expected_outputs")
      resolved_pairs = resolve_output_pairs(output_names, expected_outputs, output_map)
      max_diff = resolved_pairs.map { |pair| max_abs_diff(pair.fetch(:expected), pair.fetch(:actual)) }.max || 0.0
      max_ref = resolved_pairs.map { |pair| max_abs_value(pair.fetch(:expected)) }.max || 0.0
      tolerance = [WEBGPU_OUTPUT_ABS_TOLERANCE, max_ref * WEBGPU_OUTPUT_REL_TOLERANCE].max
      assert_operator max_diff, :<=, tolerance, "#{model_id} max_diff=#{max_diff}, tolerance=#{tolerance}"
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
  end

  def load_submodule_specs
    specs = []
    current = {}

    File.foreach(@submodule_runner) do |line|
      stripped = line.strip
      if stripped == "{"
        current = {}
        next
      end

      if (id_match = line.match(/^\s*id:\s*"([^"]+)",\s*$/))
        current["id"] = id_match[1]
        next
      end

      if (script_match = line.match(/^\s*ruby_script:\s*"([^"]+)",\s*$/))
        current["ruby_script"] = script_match[1]
        next
      end

      if (stripped == "}," || stripped == "}") && current.key?("id") && current.key?("ruby_script")
        specs << current
        current = {}
      end
    end

    specs
  end

  def capture_onnx_fixture(spec)
    Tempfile.create(["phase324_examples_capture", ".json"]) do |capture|
      env = unbundled_env(
        "MLX_BENCHMARK" => "1",
        "MLX_BENCHMARK_DEVICE" => "cpu",
        "MLX_BENCHMARK_DRYRUN" => "1",
        "MLX_EXAMPLES_SUBMODULE_ROOT" => @submodule_root,
        "MLX_EXAMPLES_ONNX_CAPTURE_FILE" => capture.path
      )

      command = [
        RbConfig.ruby,
        "-I#{@lib_root}",
        "-r",
        @force_cpu,
        "-r",
        @capture_hook,
        spec.fetch("ruby_script")
      ]

      stdout, stderr, status = Open3.capture3(env, *command, chdir: @submodule_root)
      assert status.success?, "capture failed for #{spec.fetch('id')}\nstdout:\n#{stdout}\nstderr:\n#{stderr}"
      assert File.exist?(capture.path), "capture file missing for #{spec.fetch('id')}"
      refute File.zero?(capture.path), "capture file empty for #{spec.fetch('id')}"

      payload = JSON.parse(File.read(capture.path))
      if payload.key?("error")
        flunk("capture hook failed for #{spec.fetch('id')}: #{payload.fetch('error')}")
      end

      graph_payload = payload.fetch("payload")
      node_count = graph_payload.fetch("nodes", []).length
      assert_operator node_count, :>, 0, "empty graph payload for #{spec.fetch('id')}"
      return payload
    end
  end

  def run_exported_onnx(payload, feeds, model_id:)
    Dir.mktmpdir("phase324-onnx-#{model_id.tr('/', '_')}-") do |dir|
      onnx_path = File.join(dir, "graph.onnx")
      TestSupport.export_onnx_from_graph_ir_source(onnx_path, payload, model_name: "phase324_#{model_id.tr('/', '_')}")
      out, err, status = Open3.capture3("python3", "-c", PY_RUN_ONNX_TYPED, onnx_path, JSON.generate(feeds))
      assert status.success?, "onnxruntime execution failed for #{model_id}\nstdout:\n#{out}\nstderr:\n#{err}"
      return JSON.parse(out)
    end
  end

  def resolve_output_pairs(output_names, expected_outputs, sample_outputs)
    sample_keys = sample_outputs.keys
    output_names.each_with_index.map do |name, index|
      expected = expected_outputs.fetch(name)
      actual_key = if sample_outputs.key?(name)
        name
      elsif index < sample_keys.length
        sample_keys[index]
      else
        nil
      end
      raise "missing runtime output for #{name}" if actual_key.nil?

      {
        name: name,
        expected: expected,
        actual: sample_outputs.fetch(actual_key)
      }
    end
  end

  def max_abs_diff(expected, actual)
    if expected.is_a?(Array) && actual.is_a?(Array)
      return 0.0 if expected.empty? && actual.empty?

      length = [expected.length, actual.length].max
      (0...length).map do |index|
        left = expected[index]
        right = actual[index]
        if left.nil? || right.nil?
          max_abs_value(left.nil? ? right : left)
        else
          max_abs_diff(left, right)
        end
      end.max || 0.0
    elsif expected.is_a?(Hash) && actual.is_a?(Hash)
      keys = expected.keys | actual.keys
      keys.map do |key|
        if !expected.key?(key) || !actual.key?(key)
          max_abs_value(expected[key] || actual[key])
        else
          max_abs_diff(expected.fetch(key), actual.fetch(key))
        end
      end.max || 0.0
    else
      return max_abs_value(actual) if expected.nil?
      return max_abs_value(expected) if actual.nil?

      if expected.is_a?(Array) || expected.is_a?(Hash) || actual.is_a?(Array) || actual.is_a?(Hash)
        return [max_abs_value(expected), max_abs_value(actual)].max
      end

      if (expected == true || expected == false || actual == true || actual == false)
        return expected == actual ? 0.0 : 1.0
      end

      (expected.to_f - actual.to_f).abs
    end
  end

  def max_abs_value(value)
    if value.is_a?(Array)
      return 0.0 if value.empty?

      value.map { |item| max_abs_value(item) }.max || 0.0
    elsif value.is_a?(Hash)
      return 0.0 if value.empty?

      value.values.map { |item| max_abs_value(item) }.max || 0.0
    elsif value == true || value == false
      value ? 1.0 : 0.0
    else
      value.to_f.abs
    end
  end

  def unbundled_env(overrides = {})
    env = {}
    ENV.each_key do |key|
      env[key] = nil if key.start_with?("BUNDLE_") || key.start_with?("BUNDLER_")
    end

    rubyopt_tokens = ENV.fetch("RUBYOPT", "").split
    rubyopt_tokens.reject! { |token| token.include?("bundler/setup") }
    env["RUBYOPT"] = rubyopt_tokens.empty? ? nil : rubyopt_tokens.join(" ")
    env["RUBYGEMS_GEMDEPS"] = nil if ENV.key?("RUBYGEMS_GEMDEPS")

    overrides.each do |key, value|
      env[key] = value
    end
    env
  end
end
